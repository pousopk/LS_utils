#include "manager/yolo_inference.hpp"

#include "manager/rotated_box_geometry.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

// Rotates `point` clockwise around `pivot` by `degrees` (y-down image
// space) -- a local reimplementation of the same primitive
// labeling_state.cpp's rotatePointClockwise already validates, used here
// only to build an independent expected-corners check (see below), not
// as the thing under test.
cv::Point2f rotateForTest(cv::Point2f point, cv::Point2f pivot, float degrees) {
    const double theta = static_cast<double>(degrees) * CV_PI / 180.0;
    const double dx = static_cast<double>(point.x) - static_cast<double>(pivot.x);
    const double dy = static_cast<double>(point.y) - static_cast<double>(pivot.y);
    const double cosT = std::cos(theta);
    const double sinT = std::sin(theta);
    const double rx = dx * cosT - dy * sinT;
    const double ry = dx * sinT + dy * cosT;
    return cv::Point2f(
        static_cast<float>(static_cast<double>(pivot.x) + rx), static_cast<float>(static_cast<double>(pivot.y) + ry));
}

void test_centerObbToTopLeftPivotBox_zeroAngleMatchesUnrotatedBox() {
    const ObbBox result = centerObbToTopLeftPivotBox(50.0f, 40.0f, 20.0f, 10.0f, 0.0f);
    // center (50,40), size (20,10), angle 0 -> top-left at (40,35).
    CHECK(result.box.x == 40);
    CHECK(result.box.y == 35);
    CHECK(result.box.width == 20);
    CHECK(result.box.height == 10);
    CHECK(std::abs(result.rotationDegrees - 0.0f) < 0.01f);
}

void test_centerObbToTopLeftPivotBox_matchesRotatedRectCorners() {
    // Ground truth: OpenCV's own RotatedRect::points() for the same
    // center/size/angle, independent of this function's own derivation.
    const cv::Point2f center(50.0f, 40.0f);
    const cv::Size2f size(20.0f, 10.0f);
    const float angleDegrees = 35.0f;
    const cv::RotatedRect rotatedRect(center, size, angleDegrees);
    cv::Point2f expectedCorners[4];
    rotatedRect.points(expectedCorners);

    const ObbBox result = centerObbToTopLeftPivotBox(center.x, center.y, size.width, size.height, angleDegrees);

    // Re-rotate the converted (top-left-pivot) box the same way
    // rotatedBoxCorners (phase B) does, and confirm it reproduces the
    // same 4 corners RotatedRect gives directly -- this doesn't depend on
    // trusting centerObbToTopLeftPivotBox's own derivation.
    const cv::Point2f pivot(static_cast<float>(result.box.x), static_cast<float>(result.box.y));
    const cv::Point2f actualCorners[4] = {
        rotateForTest(pivot, pivot, result.rotationDegrees),
        rotateForTest(cv::Point2f(pivot.x + result.box.width, pivot.y), pivot, result.rotationDegrees),
        rotateForTest(cv::Point2f(pivot.x + result.box.width, pivot.y + result.box.height), pivot, result.rotationDegrees),
        rotateForTest(cv::Point2f(pivot.x, pivot.y + result.box.height), pivot, result.rotationDegrees),
    };

    // RotatedRect::points() order is bottom-left, top-left, top-right,
    // bottom-right (OpenCV's own documented order) -- match each against
    // whichever of our 4 corners is closest, since we only care that the
    // *set* of 4 corners agrees, not that array indices line up.
    for (const auto& expected : expectedCorners) {
        bool foundMatch = false;
        for (const auto& actual : actualCorners) {
            if (std::abs(expected.x - actual.x) < 0.5f && std::abs(expected.y - actual.y) < 0.5f) {
                foundMatch = true;
                break;
            }
        }
        CHECK(foundMatch);
    }
}

void test_decodeYoloObbOutput_decodesSingleBoxWithRotation() {
    // One row: [cx, cy, w, h, class0, class1, angle_radians].
    // class1 wins (0.9 > 0.1), angle = pi/2 (90 deg clockwise).
    cv::Mat output(1, 7, CV_32F);
    float* row = output.ptr<float>(0);
    row[0] = 100.0f; // cx
    row[1] = 80.0f;  // cy
    row[2] = 40.0f;  // w
    row[3] = 20.0f;  // h
    row[4] = 0.1f;   // class0 score
    row[5] = 0.9f;   // class1 score
    row[6] = static_cast<float>(CV_PI) / 2.0f; // angle

    LetterboxTransform identityTransform; // scale=1, padX=padY=0
    const std::vector<std::string> classNames = {"cat", "dog"};

    const auto detections = decodeYoloObbOutput(output, classNames, identityTransform, 200, 200, 0.25f, 0.45f);

    CHECK(detections.size() == 1);
    CHECK(detections[0].className == "dog");
    CHECK(std::abs(detections[0].confidence - 0.9f) < 0.01f);
    CHECK(std::abs(detections[0].rotationDegrees - 90.0f) < 0.5f);
    // center (100,80), size (40,20), angle 90 -> top-left at (100+10, 80-20) = (110, 60)
    // (same derivation as test_centerObbToTopLeftPivotBox_zeroAngleMatchesUnrotatedBox's
    // formula, just with a nonzero angle rotating the (-w/2,-h/2) offset).
    CHECK(std::abs(detections[0].box.x - 110) <= 1);
    CHECK(std::abs(detections[0].box.y - 60) <= 1);
}

void test_decodeYoloObbOutput_belowConfidenceThresholdIsDropped() {
    cv::Mat output(1, 7, CV_32F);
    float* row = output.ptr<float>(0);
    row[0] = 100.0f;
    row[1] = 80.0f;
    row[2] = 40.0f;
    row[3] = 20.0f;
    row[4] = 0.1f;
    row[5] = 0.15f; // below confThreshold
    row[6] = 0.0f;

    LetterboxTransform identityTransform;
    const std::vector<std::string> classNames = {"cat", "dog"};
    const auto detections = decodeYoloObbOutput(output, classNames, identityTransform, 200, 200, 0.25f, 0.45f);
    CHECK(detections.empty());
}

void test_computeRotatedIoU_matchesComputeIoUForAxisAlignedBoxes() {
    const cv::Rect boxA(0, 0, 10, 10);
    const cv::Rect boxB(5, 0, 10, 10);
    // Fast path: both rotations 0.0 -- must delegate to computeIoU exactly,
    // not just approximately (it's a straight pass-through, not a
    // reimplementation).
    CHECK(computeRotatedIoU(boxA, 0.0f, boxB, 0.0f) == computeIoU(boxA, boxB));
}

void test_computeRotatedIoU_identicalRotatedBoxesGiveIoUOne() {
    const cv::Rect box(5, 5, 30, 15);
    const float iou = computeRotatedIoU(box, 25.0f, box, 25.0f);
    CHECK(std::abs(iou - 1.0f) < 0.001f);
}

void test_computeRotatedIoU_disjointRotatedBoxesGiveIoUZero() {
    const cv::Rect boxA(0, 0, 20, 20);
    const cv::Rect boxB(1000, 1000, 20, 20);
    CHECK(computeRotatedIoU(boxA, 45.0f, boxB, 30.0f) == 0.0f);
}

void test_computeRotatedIoU_partialOverlapMatchesIntersectConvexConvex() {
    // Same box, rotated 45 degrees around its own top-left pivot vs. not
    // rotated at all -- partial overlap. Ground truth computed
    // independently in this test via rotatedBoxCorners (Task 1's
    // already-tested primitive) + cv::intersectConvexConvex directly,
    // rather than trusting computeRotatedIoU's own internals.
    const cv::Rect boxA(0, 0, 100, 50);
    const cv::Rect boxB(0, 0, 100, 50);
    const float rotationA = 0.0f;
    const float rotationB = 45.0f;

    const auto cornersA = rotatedBoxCorners(boxA, rotationA);
    const auto cornersB = rotatedBoxCorners(boxB, rotationB);
    const std::vector<cv::Point2f> polyA(cornersA.begin(), cornersA.end());
    const std::vector<cv::Point2f> polyB(cornersB.begin(), cornersB.end());
    std::vector<cv::Point2f> intersection;
    const float expectedIntersectionArea = cv::intersectConvexConvex(polyA, polyB, intersection);
    const float areaA = static_cast<float>(boxA.width) * static_cast<float>(boxA.height);
    const float areaB = static_cast<float>(boxB.width) * static_cast<float>(boxB.height);
    const float expectedIoU = expectedIntersectionArea / (areaA + areaB - expectedIntersectionArea);

    const float actualIoU = computeRotatedIoU(boxA, rotationA, boxB, rotationB);
    CHECK(std::abs(actualIoU - expectedIoU) < 0.001f);
    // Sanity: this case must actually be a nontrivial partial overlap, not
    // an accidental 0 or 1 -- otherwise the test wouldn't distinguish a
    // correct implementation from a broken one.
    CHECK(actualIoU > 0.05f && actualIoU < 0.95f);
}

void test_computeBoxAgreement_rotationAwareIoUPreventsFalseMatch() {
    // Both detections share the exact same *unrotated* box -- a naive
    // axis-aligned comparison of raw .box fields would call this a
    // perfect match (IoU 1.0). But detectionB is rotated 90 degrees
    // around its own top-left pivot, so its true footprint occupies a
    // completely different region (touching detectionA's footprint only
    // along a zero-area shared edge) -- the true rotation-aware IoU is 0,
    // and computeBoxAgreement must not count this as a matched pair.
    const std::vector<Detection> a = {
        Detection{cv::Rect(0, 0, 40, 10), 0, "cat", 0.9f, 0.0f},
    };
    const std::vector<Detection> b = {
        Detection{cv::Rect(0, 0, 40, 10), 0, "cat", 0.9f, 90.0f},
    };

    const BoxAgreement agreement = computeBoxAgreement(a, b, 0.5f);
    CHECK(agreement.matchedPairs == 0);
    CHECK(agreement.totalA == 1);
    CHECK(agreement.totalB == 1);
}

} // namespace

int main() {
    test_centerObbToTopLeftPivotBox_zeroAngleMatchesUnrotatedBox();
    test_centerObbToTopLeftPivotBox_matchesRotatedRectCorners();
    test_decodeYoloObbOutput_decodesSingleBoxWithRotation();
    test_decodeYoloObbOutput_belowConfidenceThresholdIsDropped();
    test_computeRotatedIoU_matchesComputeIoUForAxisAlignedBoxes();
    test_computeRotatedIoU_identicalRotatedBoxesGiveIoUOne();
    test_computeRotatedIoU_disjointRotatedBoxesGiveIoUZero();
    test_computeRotatedIoU_partialOverlapMatchesIntersectConvexConvex();
    test_computeBoxAgreement_rotationAwareIoUPreventsFalseMatch();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

#include "manager/detection_metrics.hpp"

#include <cmath>
#include <cstdio>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}

bool approxEqual(float a, float b, float tolerance = 0.01f) {
    return std::fabs(a - b) <= tolerance;
}

} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

void test_computeDetectionMetrics_onePositiveOneFalsePositive() {
    // 2 images, each with 1 ground-truth "cat" box.
    // Image 0's prediction matches perfectly (conf 0.9) -> TP.
    // Image 1's prediction doesn't overlap at all (conf 0.8) -> FP.
    // precision=[1.0, 0.5], recall=[0.5, 0.5] -> AP = (0.5-0)*1.0 + (0.5-0.5)*0.5 = 0.5.
    std::vector<DetectionEvaluationItem> items(2);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.9f}};
    items[1].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[1].predictions = {Detection{cv::Rect(100, 100, 10, 10), 0, "cat", 0.8f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(metrics.perClass.size() == 1);
    CHECK(metrics.perClass[0].className == "cat");
    CHECK(approxEqual(metrics.perClass[0].averagePrecision, 0.5f));
    CHECK(approxEqual(metrics.meanAveragePrecision, 0.5f));
    CHECK(metrics.perClass[0].truePositives == 1);
    CHECK(metrics.perClass[0].falsePositives == 1);
    CHECK(metrics.perClass[0].numGroundTruth == 2);
}

void test_computeDetectionMetrics_perfectDetectorIsApOne() {
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.99f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(approxEqual(metrics.meanAveragePrecision, 1.0f));
}

void test_computeDetectionMetrics_noOverlapIsApZero() {
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(100, 100, 10, 10), 0, "cat", 0.9f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(approxEqual(metrics.meanAveragePrecision, 0.0f));
}

void test_computeDetectionMetrics_excludesClassWithNoGroundTruth() {
    // "cat" has a perfect match (AP=1.0); "dog" has predictions but zero
    // ground truth, so it must not dilute the mean.
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {
        Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.9f},
        Detection{cv::Rect(50, 50, 10, 10), 1, "dog", 0.7f},
    };

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(metrics.perClass.size() == 2);
    CHECK(approxEqual(metrics.meanAveragePrecision, 1.0f));
}

void test_computeDetectionMetrics_rotationAwareIoURejectsFalseMatch() {
    // Ground truth and prediction share the exact same *unrotated* box --
    // a naive axis-aligned IoU would call this a perfect match (AP=1.0).
    // The prediction is rotated 90 degrees around its own top-left pivot,
    // so its true footprint barely touches the ground truth's (a
    // zero-area shared edge) -- the true rotation-aware IoU is 0, so this
    // must be scored as a false positive, giving AP=0.
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 40, 10), "widget", 0.0f}};
    items[0].predictions = {Detection{cv::Rect(0, 0, 40, 10), 0, "widget", 0.9f, 90.0f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(metrics.perClass.size() == 1);
    CHECK(metrics.perClass[0].truePositives == 0);
    CHECK(metrics.perClass[0].falsePositives == 1);
    CHECK(approxEqual(metrics.perClass[0].averagePrecision, 0.0f));
}

} // namespace

int main() {
    test_computeDetectionMetrics_onePositiveOneFalsePositive();
    test_computeDetectionMetrics_perfectDetectorIsApOne();
    test_computeDetectionMetrics_noOverlapIsApZero();
    test_computeDetectionMetrics_excludesClassWithNoGroundTruth();
    test_computeDetectionMetrics_rotationAwareIoURejectsFalseMatch();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

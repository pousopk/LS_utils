#include "manager/yolo_inference.hpp"

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
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

void test_computeLetterboxTransform_wideFrame() {
    const LetterboxTransform t = computeLetterboxTransform(1280, 720, 640);
    CHECK(approxEqual(t.scale, 0.5f));
    CHECK(approxEqual(t.padX, 0.0f));
    CHECK(approxEqual(t.padY, 140.0f));
}

void test_computeLetterboxTransform_tallFrame() {
    const LetterboxTransform t = computeLetterboxTransform(720, 1280, 640);
    CHECK(approxEqual(t.scale, 0.5f));
    CHECK(approxEqual(t.padX, 140.0f));
    CHECK(approxEqual(t.padY, 0.0f));
}

void test_computeIoU_identical() {
    const cv::Rect r(0, 0, 10, 10);
    CHECK(approxEqual(computeIoU(r, r), 1.0f));
}

void test_computeIoU_disjoint() {
    CHECK(approxEqual(computeIoU(cv::Rect(0, 0, 10, 10), cv::Rect(20, 20, 10, 10)), 0.0f));
}

void test_computeIoU_partialOverlap() {
    // A: (0,0)-(10,10), B: (5,0)-(15,10) -> intersection 5x10=50, union 150
    const float iou = computeIoU(cv::Rect(0, 0, 10, 10), cv::Rect(5, 0, 10, 10));
    CHECK(approxEqual(iou, 50.0f / 150.0f));
}

} // namespace

int main() {
    test_computeLetterboxTransform_wideFrame();
    test_computeLetterboxTransform_tallFrame();
    test_computeIoU_identical();
    test_computeIoU_disjoint();
    test_computeIoU_partialOverlap();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

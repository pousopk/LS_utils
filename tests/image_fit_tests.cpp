#include "ui_common/image_fit.hpp"

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
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

bool near(float a, float b) {
    return std::fabs(a - b) < 0.01f;
}

void test_fitImageToRegion_wideImageFillsWidth() {
    const ImVec2 size = fitImageToRegion(1920, 1080, 480.0f, 480.0f);
    CHECK(near(size.x, 480.0f));
    CHECK(near(size.y, 270.0f));
}

void test_fitImageToRegion_tallImageFillsHeight() {
    const ImVec2 size = fitImageToRegion(1000, 2000, 400.0f, 300.0f);
    CHECK(near(size.x, 150.0f));
    CHECK(near(size.y, 300.0f));
}

void test_fitImageToRegion_emptyImageReturnsRegion() {
    const ImVec2 size = fitImageToRegion(0, 0, 320.0f, 200.0f);
    CHECK(near(size.x, 320.0f));
    CHECK(near(size.y, 200.0f));
}

void test_fitImageToRegion_neverSmallerThanOnePixel() {
    const ImVec2 size = fitImageToRegion(100000, 1, 10.0f, 10.0f);
    CHECK(size.x >= 1.0f);
    CHECK(size.y >= 1.0f);
}

} // namespace

int main() {
    test_fitImageToRegion_wideImageFillsWidth();
    test_fitImageToRegion_tallImageFillsHeight();
    test_fitImageToRegion_emptyImageReturnsRegion();
    test_fitImageToRegion_neverSmallerThanOnePixel();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

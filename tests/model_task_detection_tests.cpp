#include "manager/model_metadata_detection.hpp"

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

DetectedTaskMode resolve(const std::string& hint, const std::vector<int64_t>& shape, size_t classCount) {
    return resolveModelTask(hint, shape, 640, 480, classCount).mode;
}

void testTaskHintWins() {
    // A recognized hint decides regardless of (even missing) output shape.
    CHECK(resolve("detect", {}, 3) == DetectedTaskMode::Detection);
    CHECK(resolve("obb", {}, 3) == DetectedTaskMode::ObbDetection);
    CHECK(resolve("classify", {}, 3) == DetectedTaskMode::Classification);
    CHECK(resolve("vad", {}, 0) == DetectedTaskMode::Anomaly);
    CHECK(resolve("obb", {1, 7, 8400}, 3) == DetectedTaskMode::ObbDetection);
}

void testUnsupportedTasksRejected() {
    const ResolvedModelTask segment = resolveModelTask("segment", {1, 39, 8400}, 640, 480, 3);
    CHECK(segment.mode == DetectedTaskMode::Unknown);
    CHECK(!segment.error.empty());
    CHECK(resolve("pose", {1, 56, 8400}, 1) == DetectedTaskMode::Unknown);
}

void testDetectionFallbackByChannelCount() {
    // 4 box coords + 3 classes -> axis-aligned; one more (angle) -> OBB.
    CHECK(resolve("", {1, 7, 8400}, 3) == DetectedTaskMode::Detection);
    CHECK(resolve("", {1, 8, 8400}, 3) == DetectedTaskMode::ObbDetection);
    // The ambiguous case the old manual OBB checkbox existed for: 6
    // channels is 2-class detection or 1-class OBB -- the class count
    // from metadata settles it.
    CHECK(resolve("", {1, 6, 8400}, 2) == DetectedTaskMode::Detection);
    CHECK(resolve("", {1, 6, 8400}, 1) == DetectedTaskMode::ObbDetection);
    const ResolvedModelTask mismatch = resolveModelTask("", {1, 20, 8400}, 640, 480, 3);
    CHECK(mismatch.mode == DetectedTaskMode::Unknown);
    CHECK(!mismatch.error.empty());
}

void testOtherFallbackShapes() {
    CHECK(resolve("", {1, 10}, 10) == DetectedTaskMode::Classification);
    CHECK(resolve("", {1, 1, 480, 640}, 0) == DetectedTaskMode::Anomaly);
    CHECK(resolve("", {1, 1, 640, 480}, 0) == DetectedTaskMode::Unknown);  // H/W swapped
    CHECK(resolve("", {}, 3) == DetectedTaskMode::Unknown);                // probe failed
    // An unrecognized hint falls back to the shape guess.
    CHECK(resolve("something-else", {1, 10}, 10) == DetectedTaskMode::Classification);
}

} // namespace

int main() {
    testTaskHintWins();
    testUnsupportedTasksRejected();
    testDetectionFallbackByChannelCount();
    testOtherFallbackShapes();
    if (g_failures == 0) {
        std::printf("All model task detection tests passed.\n");
        return 0;
    }
    return 1;
}

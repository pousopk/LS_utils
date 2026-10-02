#include "manager/benchmark_state.hpp"

#include <cstdio>
#include <set>

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

// Tasks 1..count; odd ids are labeled (annotation + ground truth), even
// ids are unlabeled.
void fillData(SharedLabelStudioProjectData& data, int count) {
    for (int id = 1; id <= count; ++id) {
        DatasetTaskSummary summary;
        summary.taskId = id;
        summary.imagePath = "/data/upload/1/img" + std::to_string(id) + ".png";
        summary.hasAnnotation = id % 2 == 1;
        data.summaries.push_back(summary);
        if (summary.hasAnnotation) {
            ImageGroundTruth groundTruth;
            groundTruth.imageFilename = "img" + std::to_string(id) + ".png";
            groundTruth.classificationLabel = "label" + std::to_string(id);
            groundTruth.hasClassificationAnnotation = true;
            data.groundTruthByTaskId[id] = groundTruth;
        }
    }
}

void testKeepsOnlyLabeledTasksWithGroundTruthRenamed() {
    SharedLabelStudioProjectData data;
    fillData(data, 6);
    // Annotated but no parsed ground truth (e.g. no image field) -> skipped.
    data.groundTruthByTaskId.erase(5);

    const BenchmarkTaskSelection selection = selectBenchmarkTasks(data, 0, 1);
    CHECK(selection.labeledTaskCount == 2);
    CHECK(selection.tasks.size() == 2);
    CHECK(selection.groundTruth.images.size() == 2);
    if (selection.tasks.size() == 2 && selection.groundTruth.images.size() == 2) {
        CHECK(selection.tasks[0].taskId == 1);
        CHECK(selection.tasks[1].taskId == 3);
        // Ground truth is matched by the cached file's name, not the original basename.
        CHECK(selection.groundTruth.images[0].imageFilename == "1.png");
        CHECK(selection.groundTruth.images[1].imageFilename == "3.png");
        CHECK(selection.groundTruth.images[1].classificationLabel == "label3");
    }
}

void testSamplesBeforeDownloadAndSortsById() {
    SharedLabelStudioProjectData data;
    fillData(data, 200);  // 100 labeled
    const BenchmarkTaskSelection selection = selectBenchmarkTasks(data, 10, 42);
    CHECK(selection.labeledTaskCount == 100);
    CHECK(selection.tasks.size() == 10);
    CHECK(selection.groundTruth.images.size() == 10);
    std::set<int> ids;
    for (size_t i = 0; i < selection.tasks.size(); ++i) {
        CHECK(selection.tasks[i].taskId % 2 == 1);
        if (i > 0) {
            CHECK(selection.tasks[i - 1].taskId < selection.tasks[i].taskId);
        }
        ids.insert(selection.tasks[i].taskId);
    }
    CHECK(ids.size() == 10);

    // Same seed -> same sample.
    const BenchmarkTaskSelection again = selectBenchmarkTasks(data, 10, 42);
    bool same = again.tasks.size() == selection.tasks.size();
    for (size_t i = 0; same && i < again.tasks.size(); ++i) {
        same = again.tasks[i].taskId == selection.tasks[i].taskId;
    }
    CHECK(same);
}

void testSampleLargerThanLabeledKeepsAll() {
    SharedLabelStudioProjectData data;
    fillData(data, 10);  // 5 labeled
    const BenchmarkTaskSelection selection = selectBenchmarkTasks(data, 50, 7);
    CHECK(selection.tasks.size() == 5);
    CHECK(selection.labeledTaskCount == 5);
}

} // namespace

int main() {
    testKeepsOnlyLabeledTasksWithGroundTruthRenamed();
    testSamplesBeforeDownloadAndSortsById();
    testSampleLargerThanLabeledKeepsAll();
    if (g_failures == 0) {
        std::printf("All benchmark task selection tests passed.\n");
        return 0;
    }
    return 1;
}

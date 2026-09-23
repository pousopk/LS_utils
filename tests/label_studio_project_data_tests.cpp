#include "manager/label_studio_project_data.hpp"

#include <cstdio>
#include <optional>

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

void test_applySharedTaskListResult_successReplacesDataAndBumpsVersion() {
    SharedLabelStudioProjectData state;
    state.version = 3;
    state.loading = true;

    DatasetTaskListResult result;
    result.success = true;
    result.rawTasksJson = nlohmann::json::array({nlohmann::json{{"id", 1}}});
    result.summaries = {DatasetTaskSummary{1, "/a.jpg", true, false, {}, std::nullopt, std::nullopt}};
    result.summaryIndexByTaskId = {{1, 0}};

    applySharedTaskListResult(state, result);

    CHECK(!state.loading);
    CHECK(state.error.empty());
    CHECK(state.loaded);
    CHECK(state.version == 4);
    CHECK(state.summaries.size() == 1);
    CHECK(state.summaries[0].taskId == 1);
    CHECK(state.summaryIndexByTaskId.at(1) == 0);
}

void test_applySharedTaskListResult_failurePreservesPriorDataAndSetsError() {
    SharedLabelStudioProjectData state;
    state.loaded = true;
    state.version = 2;
    state.summaries = {DatasetTaskSummary{5, "/old.jpg", false, false, {}, std::nullopt, std::nullopt}};

    DatasetTaskListResult result;
    result.success = false;
    result.error = "network error";

    applySharedTaskListResult(state, result);

    CHECK(!state.loading);
    CHECK(state.error == "network error");
    CHECK(state.loaded);
    CHECK(state.version == 2);
    CHECK(state.summaries.size() == 1);
    CHECK(state.summaries[0].taskId == 5);
}

void test_applySharedProjectConfigResult_successSetsConfigAndControlTagNames() {
    SharedLabelStudioProjectData state;

    LabelStudioProjectConfig config;
    config.dataImageKey = "image";
    config.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Cat", "Dog"}},
        LabelStudioControlTag{LabelStudioControlTagType::BrushLabels, "mask", "image", {"Cat"}},
    };

    applySharedProjectConfigResult(state, config);

    CHECK(state.error.empty());
    CHECK(state.projectConfig.dataImageKey == "image");
    CHECK(state.rectangleLabelsFromName == "label");
    CHECK(state.brushLabelsFromName == "mask");
}

void test_applySharedProjectConfigResult_failurePreservesPriorConfigAndSetsError() {
    SharedLabelStudioProjectData state;
    state.projectConfig.dataImageKey = "image";
    state.rectangleLabelsFromName = "label";

    LabelStudioProjectConfig config;
    config.error = "HTTP 401: invalid token";

    applySharedProjectConfigResult(state, config);

    CHECK(state.error == "Project config: HTTP 401: invalid token");
    CHECK(state.projectConfig.dataImageKey == "image");
    CHECK(state.rectangleLabelsFromName == "label");
}

} // namespace

int main() {
    test_applySharedTaskListResult_successReplacesDataAndBumpsVersion();
    test_applySharedTaskListResult_failurePreservesPriorDataAndSetsError();
    test_applySharedProjectConfigResult_successSetsConfigAndControlTagNames();
    test_applySharedProjectConfigResult_failurePreservesPriorConfigAndSetsError();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

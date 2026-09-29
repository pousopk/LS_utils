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

DatasetTaskSummary summary(int id) {
    DatasetTaskSummary s;
    s.taskId = id;
    s.imagePath = "/" + std::to_string(id) + ".jpg";
    return s;
}

DatasetTaskListPage pageOf(std::initializer_list<int> ids) {
    DatasetTaskListPage page;
    for (const int id : ids) {
        page.summaries.push_back(summary(id));
        page.boxesByTaskId[id] = DatasetBoxesToDraw{};
    }
    return page;
}

void test_firstLoadStreamsPagesIntoLiveData() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    CHECK(state.loading);
    CHECK(state.streamingIntoLive);
    const uint64_t v0 = state.version;

    applySharedTaskListPage(state, pageOf({1, 2}));
    CHECK(state.summaries.size() == 2);
    CHECK(state.version > v0);
    CHECK(!state.loaded);

    applySharedTaskListPage(state, pageOf({3}));
    CHECK(state.summaries.size() == 3);
    CHECK(state.summaryIndexByTaskId.count(3) == 1 && state.summaryIndexByTaskId.at(3) == 2);
    CHECK(state.boxesByTaskId.count(3) == 1);

    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    CHECK(state.loaded);
    CHECK(!state.loading);
    CHECK(state.error.empty());
}

void test_applySharedTaskListPage_skipsDuplicateTaskIds() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1, 2}));
    applySharedTaskListPage(state, pageOf({2, 3}));   // offset shift re-delivered task 2
    CHECK(state.summaries.size() == 3);
    CHECK(state.summaries.size() == 3 && state.summaries[2].taskId == 3);
    CHECK(state.summaryIndexByTaskId.count(2) == 1 && state.summaryIndexByTaskId.at(2) == 1);
}

void test_refreshStagesPagesAndSwapsOnCompletion() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1, 2}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    const uint64_t vLoaded = state.version;

    beginSharedTaskListLoad(state);
    CHECK(!state.streamingIntoLive);
    applySharedTaskListPage(state, pageOf({7, 8, 9}));
    CHECK(state.summaries.size() == 2);   // live untouched mid-refresh
    CHECK(state.version == vLoaded);

    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    CHECK(state.summaries.size() == 3);
    CHECK(!state.summaries.empty() && state.summaries[0].taskId == 7);
    CHECK(state.summaryIndexByTaskId.count(1) == 0);
    CHECK(state.version > vLoaded);
    CHECK(state.staging.summaries.empty());
}

void test_refreshFailureKeepsPreviousLiveDataAndDiscardsStaging() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1, 2}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    const uint64_t vLoaded = state.version;

    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({7}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Failed, "timeout"});

    CHECK(state.loaded);
    CHECK(state.summaries.size() == 2);
    CHECK(!state.summaries.empty() && state.summaries[0].taskId == 1);
    CHECK(state.version == vLoaded);
    CHECK(state.staging.summaries.empty());
    CHECK(state.error.find("timeout") != std::string::npos);
}

void test_firstLoadFailureKeepsPartialDataAndStaysNotLoaded() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    state.loadProgress = DatasetTaskListProgress{2, 150000};
    applySharedTaskListPage(state, pageOf({1, 2}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Failed, "HTTP 502"});

    CHECK(!state.loaded);
    CHECK(!state.loading);
    CHECK(state.summaries.size() == 2);
    CHECK(state.error == "Loaded 2 of 150000 tasks, then: HTTP 502");
}

void test_firstLoadCancelledReportsCancelled() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    state.loadProgress = DatasetTaskListProgress{1, std::nullopt};
    applySharedTaskListPage(state, pageOf({1}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Cancelled, ""});
    CHECK(state.error == "Loaded 1 tasks, then: cancelled");
}

void test_beginAfterFailedFirstLoadClearsPartialData() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Failed, "x"});

    beginSharedTaskListLoad(state);   // retry: still not loaded -> streams live again, from scratch
    CHECK(state.streamingIntoLive);
    CHECK(state.summaries.empty());
    CHECK(state.summaryIndexByTaskId.empty());
    CHECK(state.error.empty());
}

void test_buildDatasetTaskListPage_summarizesAndKeepsOnlyCompactData() {
    const nlohmann::json pageTasks = nlohmann::json::array({
        {{"id", 10}, {"data", {{"image", "/10.jpg"}}}, {"created_at", "2026-09-10T10:00:00Z"},
         {"total_annotations", 1}, {"total_predictions", 0}},
        {{"id", 11}, {"data", {{"other", "/11.jpg"}}}},
    });
    DatasetTaskListConfig config;
    config.dataImageKey = "image";
    config.rectangleLabelsFromName = "box";
    const DatasetTaskListPage page = buildDatasetTaskListPage(pageTasks, config);
    CHECK(page.summaries.size() == 1);
    if (page.summaries.size() == 1) {
        CHECK(page.summaries[0].taskId == 10);
        CHECK(page.summaries[0].hasAnnotation);
        CHECK(page.summaries[0].createdAt == "2026-09-10T10:00:00Z");
    }
    CHECK(page.masksByTaskId.empty());
}

void test_describeSharedTaskListLoadProgress() {
    SharedLabelStudioProjectData state;
    state.loading = true;
    state.loadProgress = DatasetTaskListProgress{12400, 150000};
    CHECK(describeSharedTaskListLoadProgress(state) == "Loading tasks... 12400 / 150000");
    state.loadProgress = DatasetTaskListProgress{400, std::nullopt};
    CHECK(describeSharedTaskListLoadProgress(state) == "Loading tasks... 400");
}

void test_buildDatasetTaskListPage_dropsTasksOutsideCreatedAtBounds() {
    const nlohmann::json pageTasks = nlohmann::json::array({
        {{"id", 1}, {"data", {{"image", "/1.jpg"}}}, {"created_at", "2026-08-31T23:00:00Z"}},
        {{"id", 2}, {"data", {{"image", "/2.jpg"}}}, {"created_at", "2026-09-01T10:00:00Z"}},
        {{"id", 3}, {"data", {{"image", "/3.jpg"}}}, {"created_at", "2026-09-02T00:00:00Z"}},
    });
    DatasetTaskListConfig config;
    config.dataImageKey = "image";
    config.createdAtBounds.fromInclusive = *parseIso8601Utc("2026-09-01T00:00:00Z");
    config.createdAtBounds.toExclusive = *parseIso8601Utc("2026-09-02T00:00:00Z");
    const DatasetTaskListPage page = buildDatasetTaskListPage(pageTasks, config);
    CHECK(page.summaries.size() == 1);
    CHECK(!page.summaries.empty() && page.summaries[0].taskId == 2);
    CHECK(page.droppedOutOfRange == 2);
}

void test_toCreatedAtBounds_fromMidnightToEndOfToDay() {
    TaskImportDateRange range;
    range.from = CalendarDate{2026, 9, 1};
    range.to = CalendarDate{2026, 9, 15};
    const TaskCreatedAtBounds bounds = toCreatedAtBounds(range);
    CHECK(bounds.fromInclusive == localMidnight(CalendarDate{2026, 9, 1}));
    CHECK(bounds.toExclusive == localMidnight(CalendarDate{2026, 9, 16}));   // whole "to" day included
    CHECK(toCreatedAtBounds(TaskImportDateRange{}).unbounded());
}

void test_isValidTaskImportDateRange() {
    CHECK(isValidTaskImportDateRange(TaskImportDateRange{}));
    CHECK(isValidTaskImportDateRange(TaskImportDateRange{CalendarDate{2026, 9, 1}, CalendarDate{2026, 9, 1}}));
    CHECK(!isValidTaskImportDateRange(TaskImportDateRange{CalendarDate{2026, 9, 2}, CalendarDate{2026, 9, 1}}));
    CHECK(isValidTaskImportDateRange(TaskImportDateRange{std::nullopt, CalendarDate{2026, 9, 1}}));
}

void test_describeTaskImportDateRange() {
    CHECK(describeTaskImportDateRange(TaskImportDateRange{}).empty());
    CHECK(describeTaskImportDateRange(TaskImportDateRange{CalendarDate{2026, 9, 1}, CalendarDate{2026, 9, 15}})
          == "Tasks imported 2026/09/01 - 2026/09/15");
    CHECK(describeTaskImportDateRange(TaskImportDateRange{CalendarDate{2026, 9, 1}, std::nullopt})
          == "Tasks imported from 2026/09/01");
    CHECK(describeTaskImportDateRange(TaskImportDateRange{std::nullopt, CalendarDate{2026, 9, 15}})
          == "Tasks imported up to 2026/09/15");
}

void test_setSharedTaskImportDateRange_newRangeClearsLoadedList() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1, 2}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    const uint64_t vLoaded = state.version;

    const TaskImportDateRange range{CalendarDate{2026, 9, 1}, std::nullopt};
    CHECK(setSharedTaskImportDateRange(state, range));
    CHECK(state.appliedImportDateRange == range);
    CHECK(!state.loaded);
    CHECK(!state.loading);
    CHECK(state.summaries.empty());
    CHECK(state.summaryIndexByTaskId.empty());
    CHECK(state.version > vLoaded);
}

void test_setSharedTaskImportDateRange_sameRangeIsNoOp() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    applySharedTaskListPage(state, pageOf({1}));
    applySharedTaskListCompletion(state, DatasetTaskListCompletion{DatasetTaskListOutcome::Completed, ""});
    CHECK(!setSharedTaskImportDateRange(state, TaskImportDateRange{}));
    CHECK(state.loaded);
    CHECK(state.summaries.size() == 1);
}

void test_applySharedTaskListPage_accumulatesDroppedOutOfRange() {
    SharedLabelStudioProjectData state;
    beginSharedTaskListLoad(state);
    DatasetTaskListPage page = pageOf({1});
    page.droppedOutOfRange = 5;
    applySharedTaskListPage(state, page);
    applySharedTaskListPage(state, pageOf({2}));
    CHECK(state.droppedOutOfRangeTasks == 5);
    beginSharedTaskListLoad(state);
    CHECK(state.droppedOutOfRangeTasks == 0);
}

} // namespace

int main() {
    test_applySharedProjectConfigResult_successSetsConfigAndControlTagNames();
    test_applySharedProjectConfigResult_failurePreservesPriorConfigAndSetsError();

    test_firstLoadStreamsPagesIntoLiveData();
    test_applySharedTaskListPage_skipsDuplicateTaskIds();
    test_refreshStagesPagesAndSwapsOnCompletion();
    test_refreshFailureKeepsPreviousLiveDataAndDiscardsStaging();
    test_firstLoadFailureKeepsPartialDataAndStaysNotLoaded();
    test_firstLoadCancelledReportsCancelled();
    test_beginAfterFailedFirstLoadClearsPartialData();
    test_buildDatasetTaskListPage_summarizesAndKeepsOnlyCompactData();
    test_describeSharedTaskListLoadProgress();
    test_buildDatasetTaskListPage_dropsTasksOutsideCreatedAtBounds();
    test_toCreatedAtBounds_fromMidnightToEndOfToDay();
    test_isValidTaskImportDateRange();
    test_describeTaskImportDateRange();
    test_setSharedTaskImportDateRange_newRangeClearsLoadedList();
    test_setSharedTaskImportDateRange_sameRangeIsNoOp();
    test_applySharedTaskListPage_accumulatesDroppedOutOfRange();
    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

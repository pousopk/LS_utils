#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

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

void test_summarizeDatasetTasks_basicFieldsAndAnnotationPredictionPresence() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"}, "total_annotations": 1, "total_predictions": 0},
        {"id": 2, "data": {"image": "/a/2.jpg"}, "total_annotations": 0, "total_predictions": 1}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries.size() == 2);
    CHECK(summaries[0].taskId == 1);
    CHECK(summaries[0].imagePath == "/a/1.jpg");
    CHECK(summaries[0].hasAnnotation);
    CHECK(!summaries[0].hasPrediction);
    CHECK(!summaries[1].hasAnnotation);
    CHECK(summaries[1].hasPrediction);
}

void test_summarizeDatasetTasks_fallsBackToArrayLengthWithoutTotalsFields() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"}, "annotations": [{"id": 9, "result": []}], "predictions": []}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries[0].hasAnnotation);
    CHECK(!summaries[0].hasPrediction);
}

void test_summarizeDatasetTasks_collectsClassNamesFromAnnotationAndPrediction() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"},
         "annotations": [{"result": [{"value": {"rectanglelabels": ["Cat"]}}]}],
         "predictions": [{"score": 0.5, "result": [{"value": {"rectanglelabels": ["Dog"]}}]}]}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries.size() == 1);
    CHECK(summaries[0].classNames.size() == 2);
    CHECK(summaries[0].classNames[0] == "Cat");
    CHECK(summaries[0].classNames[1] == "Dog");
}

void test_summarizeDatasetTasks_collectsClassNamesFromChoicesAndBrushlabels() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [
             {"result": [{"value": {"choices": ["Blurry"]}}]},
             {"result": [{"value": {"brushlabels": ["Scratch"]}}]}
         ]}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries[0].classNames.size() == 2);
    CHECK(summaries[0].classNames[0] == "Blurry");
    CHECK(summaries[0].classNames[1] == "Scratch");
}

void test_summarizeDatasetTasks_dedupesClassNames() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"},
         "annotations": [{"result": [
             {"value": {"rectanglelabels": ["Cat"]}},
             {"value": {"rectanglelabels": ["Cat"]}}
         ]}]}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries[0].classNames.size() == 1);
    CHECK(summaries[0].classNames[0] == "Cat");
}

void test_summarizeDatasetTasks_confidenceRangeAcrossPredictions() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"},
         "predictions": [{"score": 0.2, "result": []}, {"score": 0.9, "result": []}]}
    ])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(summaries[0].minConfidence.has_value());
    CHECK(*summaries[0].minConfidence == 0.2f);
    CHECK(*summaries[0].maxConfidence == 0.9f);
}

void test_summarizeDatasetTasks_noConfidenceWhenNoPredictionsHaveScore() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a/1.jpg"}}])");
    const auto summaries = summarizeDatasetTasks(json, "image");
    CHECK(!summaries[0].minConfidence.has_value());
    CHECK(!summaries[0].maxConfidence.has_value());
}

void test_summarizeDatasetTasks_skipsTaskMissingIdOrImageKey() {
    const auto json = nlohmann::json::parse(R"([
        {"data": {"image": "/a/1.jpg"}},
        {"id": 2, "data": {}}
    ])");
    CHECK(summarizeDatasetTasks(json, "image").empty());
}

void test_summarizeDatasetTasks_wrappedInTasksKey() {
    const auto json = nlohmann::json::parse(R"({"tasks": [{"id": 1, "data": {"image": "/a/1.jpg"}}], "total": 1})");
    CHECK(summarizeDatasetTasks(json, "image").size() == 1);
}

void test_filterDatasetTasks_annotationHasExcludesTasksWithoutAnnotation() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", true, false, {}, std::nullopt, std::nullopt},
        DatasetTaskSummary{2, "/b.jpg", false, false, {}, std::nullopt, std::nullopt},
    };
    DatasetFilterSpec filter;
    filter.annotationFilter = DatasetPresenceFilter::Has;
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 1);
}

void test_filterDatasetTasks_predictionLacksExcludesTasksWithPrediction() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", false, true, {}, std::nullopt, std::nullopt},
        DatasetTaskSummary{2, "/b.jpg", false, false, {}, std::nullopt, std::nullopt},
    };
    DatasetFilterSpec filter;
    filter.predictionFilter = DatasetPresenceFilter::Lacks;
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 2);
}

void test_filterDatasetTasks_classNameFilterRequiresExactMatch() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", true, false, {"Cat", "Dog"}, std::nullopt, std::nullopt},
        DatasetTaskSummary{2, "/b.jpg", true, false, {"Dog"}, std::nullopt, std::nullopt},
    };
    DatasetFilterSpec filter;
    filter.classNameFilter = "Cat";
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 1);
}

void test_filterDatasetTasks_confidenceLessThanUsesMinConfidence() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", false, true, {}, 0.2f, 0.8f},
        DatasetTaskSummary{2, "/b.jpg", false, true, {}, 0.6f, 0.9f},
    };
    DatasetFilterSpec filter;
    filter.confidenceFilterMode = DatasetConfidenceFilterMode::LessThan;
    filter.confidenceThreshold = 0.5f;
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 1);
}

void test_filterDatasetTasks_confidenceGreaterThanUsesMaxConfidence() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", false, true, {}, 0.1f, 0.4f},
        DatasetTaskSummary{2, "/b.jpg", false, true, {}, 0.6f, 0.9f},
    };
    DatasetFilterSpec filter;
    filter.confidenceFilterMode = DatasetConfidenceFilterMode::GreaterThan;
    filter.confidenceThreshold = 0.5f;
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 2);
}

void test_filterDatasetTasks_confidenceFilterExcludesTasksWithNoPrediction() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", false, false, {}, std::nullopt, std::nullopt},
    };
    DatasetFilterSpec filter;
    filter.confidenceFilterMode = DatasetConfidenceFilterMode::GreaterThan;
    filter.confidenceThreshold = 0.0f;
    CHECK(filterDatasetTasks(summaries, filter).empty());
}

void test_filterDatasetTasks_combinesMultipleDimensions() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", true, true, {"Cat"}, 0.7f, 0.9f},
        DatasetTaskSummary{2, "/b.jpg", true, true, {"Dog"}, 0.7f, 0.9f},
    };
    DatasetFilterSpec filter;
    filter.annotationFilter = DatasetPresenceFilter::Has;
    filter.classNameFilter = "Cat";
    const auto matching = filterDatasetTasks(summaries, filter);
    CHECK(matching.size() == 1);
    CHECK(matching[0] == 1);
}

void test_filterDatasetTasks_allDefaultsMatchesEverything() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{1, "/a.jpg", true, true, {}, std::nullopt, std::nullopt},
        DatasetTaskSummary{2, "/b.jpg", false, false, {}, std::nullopt, std::nullopt},
    };
    CHECK(filterDatasetTasks(summaries, DatasetFilterSpec{}).size() == 2);
}

void test_buildDatasetExportJson_returnsOnlyMatchingTasks() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"}},
        {"id": 2, "data": {"image": "/b.jpg"}},
        {"id": 3, "data": {"image": "/c.jpg"}}
    ])");
    const auto exported = buildDatasetExportJson(json, {1, 3});
    CHECK(exported.size() == 2);
    CHECK(exported[0]["id"] == 1);
    CHECK(exported[1]["id"] == 3);
}

void test_buildDatasetExportJson_emptyMatchingIdsReturnsEmptyArray() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a.jpg"}}])");
    CHECK(buildDatasetExportJson(json, {}).empty());
}

void test_buildDatasetExportJson_preservesTaskObjectUnmodified() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"}, "annotations": [{"id": 9, "result": []}]}
    ])");
    const auto exported = buildDatasetExportJson(json, {1});
    CHECK(exported[0]["annotations"][0]["id"] == 9);
}

} // namespace

int main() {
    test_summarizeDatasetTasks_basicFieldsAndAnnotationPredictionPresence();
    test_summarizeDatasetTasks_fallsBackToArrayLengthWithoutTotalsFields();
    test_summarizeDatasetTasks_collectsClassNamesFromAnnotationAndPrediction();
    test_summarizeDatasetTasks_collectsClassNamesFromChoicesAndBrushlabels();
    test_summarizeDatasetTasks_dedupesClassNames();
    test_summarizeDatasetTasks_confidenceRangeAcrossPredictions();
    test_summarizeDatasetTasks_noConfidenceWhenNoPredictionsHaveScore();
    test_summarizeDatasetTasks_skipsTaskMissingIdOrImageKey();
    test_summarizeDatasetTasks_wrappedInTasksKey();
    test_filterDatasetTasks_annotationHasExcludesTasksWithoutAnnotation();
    test_filterDatasetTasks_predictionLacksExcludesTasksWithPrediction();
    test_filterDatasetTasks_classNameFilterRequiresExactMatch();
    test_filterDatasetTasks_confidenceLessThanUsesMinConfidence();
    test_filterDatasetTasks_confidenceGreaterThanUsesMaxConfidence();
    test_filterDatasetTasks_confidenceFilterExcludesTasksWithNoPrediction();
    test_filterDatasetTasks_combinesMultipleDimensions();
    test_filterDatasetTasks_allDefaultsMatchesEverything();
    test_buildDatasetExportJson_returnsOnlyMatchingTasks();
    test_buildDatasetExportJson_emptyMatchingIdsReturnsEmptyArray();
    test_buildDatasetExportJson_preservesTaskObjectUnmodified();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

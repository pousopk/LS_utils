#include "manager/dataset_thumbnail_lru.hpp"
#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <cstdio>
#include <sstream>
#include <unordered_set>

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

void test_idsToEvict_noEvictionWhenUnderCap() {
    CHECK(idsToEvict({1, 2, 3}, {}, 5).empty());
}

void test_idsToEvict_evictsLeastRecentlyTouchedFirst() {
    const auto toEvict = idsToEvict({1, 2, 3, 4}, {}, 2);
    CHECK(toEvict.size() == 2);
    CHECK(toEvict[0] == 1);
    CHECK(toEvict[1] == 2);
}

void test_idsToEvict_neverEvictsCurrentlyVisibleIds() {
    const auto toEvict = idsToEvict({1, 2, 3, 4}, {1, 2}, 2);
    CHECK(toEvict.size() == 2);
    CHECK(toEvict[0] == 3);
    CHECK(toEvict[1] == 4);
}

void test_idsToEvict_visibleIdsAloneOverCapEvictsOnlyNonVisible() {
    CHECK(idsToEvict({1, 2, 3}, {1, 2, 3}, 1).empty());
}

void test_idsToEvict_exactlyAtCapEvictsNothing() {
    CHECK(idsToEvict({1, 2}, {}, 2).empty());
}

void test_boxesToDrawForTask_separatesAnnotationAndPredictionBoxes() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [{"result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "rectanglelabels": ["Cat"]}}
         ]}],
         "predictions": [{"score": 0.9, "result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 50.0, "y": 50.0, "width": 10.0, "height": 10.0, "rectanglelabels": ["Dog"]}}
         ]}]}
    ])");
    const auto boxes = boxesToDrawForTask(json, "label", 1);
    CHECK(boxes.annotationBoxes.size() == 1);
    CHECK(boxes.annotationBoxes[0].className == "Cat");
    CHECK(boxes.predictionBoxes.size() == 1);
    CHECK(boxes.predictionBoxes[0].className == "Dog");
}

void test_boxesToDrawForTask_ignoresResultsWithDifferentFromName() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [{"result": [
             {"type": "rectanglelabels", "from_name": "other_label", "original_width": 100, "original_height": 100,
              "value": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "rectanglelabels": ["Cat"]}}
         ]}]}
    ])");
    const auto boxes = boxesToDrawForTask(json, "label", 1);
    CHECK(boxes.annotationBoxes.empty());
    CHECK(boxes.predictionBoxes.empty());
}

void test_boxesToDrawForTask_emptyFromNameReturnsNothing() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [{"result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "rectanglelabels": ["Cat"]}}
         ]}]}
    ])");
    const auto boxes = boxesToDrawForTask(json, "", 1);
    CHECK(boxes.annotationBoxes.empty());
    CHECK(boxes.predictionBoxes.empty());
}

void test_boxesToDrawForTask_taskNotFoundReturnsEmpty() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a.jpg"}}])");
    const auto boxes = boxesToDrawForTask(json, "label", 999);
    CHECK(boxes.annotationBoxes.empty());
    CHECK(boxes.predictionBoxes.empty());
}

void test_boxesToDrawForTask_preservesRotation() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [{"result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "rotation": 45.0, "rectanglelabels": ["Cat"]}}
         ]}]}
    ])");
    const auto boxes = boxesToDrawForTask(json, "label", 1).annotationBoxes;
    CHECK(boxes.size() == 1);
    CHECK(boxes[0].rotationDegrees == 45.0f);
}

nlohmann::json brushResultItem(const std::vector<int>& rle, const std::string& fromName, const std::string& className) {
    nlohmann::json item;
    item["type"] = "brushlabels";
    item["from_name"] = fromName;
    item["original_width"] = 2;
    item["original_height"] = 2;
    item["value"]["rle"] = rle;
    item["value"]["brushlabels"] = nlohmann::json::array({className});
    return item;
}

void test_masksToDrawForTask_separatesAnnotationAndPredictionMasks() {
    const cv::Mat mask(2, 2, CV_8UC1, cv::Scalar(255));
    const std::vector<int> rle = encodeMaskToLabelStudioRle(mask);

    nlohmann::json task;
    task["id"] = 1;
    task["data"]["image"] = "/a.jpg";
    task["annotations"] = nlohmann::json::array(
        {nlohmann::json{{"result", nlohmann::json::array({brushResultItem(rle, "mask", "Cat")})}}});
    task["predictions"] = nlohmann::json::array(
        {nlohmann::json{{"result", nlohmann::json::array({brushResultItem(rle, "mask", "Dog")})}}});

    const auto json = nlohmann::json::array({task});
    const auto masks = masksToDrawForTask(json, "mask", 1);
    CHECK(masks.annotationMasks.size() == 1);
    CHECK(masks.annotationMasks[0].className == "Cat");
    CHECK(!masks.annotationMasks[0].mask.empty());
    CHECK(masks.predictionMasks.size() == 1);
    CHECK(masks.predictionMasks[0].className == "Dog");
}

void test_masksToDrawForTask_emptyFromNameReturnsNothing() {
    const cv::Mat mask(2, 2, CV_8UC1, cv::Scalar(255));
    const std::vector<int> rle = encodeMaskToLabelStudioRle(mask);

    nlohmann::json task;
    task["id"] = 1;
    task["data"]["image"] = "/a.jpg";
    task["annotations"] = nlohmann::json::array(
        {nlohmann::json{{"result", nlohmann::json::array({brushResultItem(rle, "mask", "Cat")})}}});

    const auto json = nlohmann::json::array({task});
    const auto masks = masksToDrawForTask(json, "", 1);
    CHECK(masks.annotationMasks.empty());
    CHECK(masks.predictionMasks.empty());
}

void test_masksToDrawForTask_taskNotFoundReturnsEmpty() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a.jpg"}}])");
    const auto masks = masksToDrawForTask(json, "mask", 999);
    CHECK(masks.annotationMasks.empty());
    CHECK(masks.predictionMasks.empty());
}

void test_buildBoxesByTaskId_matchesPerTaskLookupForEveryTask() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a.jpg"},
         "annotations": [{"result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "rectanglelabels": ["Cat"]}}
         ]}]},
        {"id": 2, "data": {"image": "/b.jpg"},
         "predictions": [{"score": 0.5, "result": [
             {"type": "rectanglelabels", "from_name": "label", "original_width": 100, "original_height": 100,
              "value": {"x": 5.0, "y": 5.0, "width": 15.0, "height": 15.0, "rectanglelabels": ["Dog"]}}
         ]}]}
    ])");
    const auto byId = buildBoxesByTaskId(json, "label");
    CHECK(byId.size() == 2);
    CHECK(byId.at(1).annotationBoxes.size() == 1);
    CHECK(byId.at(1).annotationBoxes[0].className == "Cat");
    CHECK(byId.at(2).predictionBoxes.size() == 1);
    CHECK(byId.at(2).predictionBoxes[0].className == "Dog");
}

void test_buildBoxesByTaskId_emptyFromNameReturnsEmptyMap() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a.jpg"}}])");
    CHECK(buildBoxesByTaskId(json, "").empty());
}

void test_buildMasksByTaskId_matchesPerTaskLookupForEveryTask() {
    const cv::Mat mask(2, 2, CV_8UC1, cv::Scalar(255));
    const std::vector<int> rle = encodeMaskToLabelStudioRle(mask);

    nlohmann::json taskOne;
    taskOne["id"] = 1;
    taskOne["data"]["image"] = "/a.jpg";
    taskOne["annotations"] = nlohmann::json::array(
        {nlohmann::json{{"result", nlohmann::json::array({brushResultItem(rle, "mask", "Cat")})}}});

    const auto json = nlohmann::json::array({taskOne});
    const auto byId = buildMasksByTaskId(json, "mask");
    CHECK(byId.size() == 1);
    CHECK(byId.at(1).annotationMasks.size() == 1);
    CHECK(byId.at(1).annotationMasks[0].className == "Cat");
}

void test_buildMasksByTaskId_emptyFromNameReturnsEmptyMap() {
    const auto json = nlohmann::json::parse(R"([{"id": 1, "data": {"image": "/a.jpg"}}])");
    CHECK(buildMasksByTaskId(json, "").empty());
}

void test_indexSummariesByTaskId_mapsEachTaskIdToItsIndex() {
    const std::vector<DatasetTaskSummary> summaries = {
        DatasetTaskSummary{5, "/a.jpg", true, false, {}, std::nullopt, std::nullopt},
        DatasetTaskSummary{9, "/b.jpg", false, true, {}, std::nullopt, std::nullopt},
    };
    const auto index = indexSummariesByTaskId(summaries);
    CHECK(index.size() == 2);
    CHECK(summaries[index.at(5)].taskId == 5);
    CHECK(summaries[index.at(9)].taskId == 9);
}

void test_indexSummariesByTaskId_emptyInputReturnsEmptyMap() {
    CHECK(indexSummariesByTaskId({}).empty());
}

nlohmann::json brushItem(const std::string& fromName, const cv::Mat& mask, const std::string& className) {
    return {
        {"type", "brushlabels"},
        {"from_name", fromName},
        {"original_width", mask.cols},
        {"original_height", mask.rows},
        {"value", {{"format", "rle"}, {"rle", encodeMaskToLabelStudioRle(mask)}, {"brushlabels", {className}}}},
    };
}

cv::Mat squareMask(int size, int x0, int y0, int side) {
    cv::Mat mask = cv::Mat::zeros(size, size, CV_8UC1);
    mask(cv::Rect(x0, y0, side, side)).setTo(255);
    return mask;
}

void test_buildEncodedMasksByTaskId_decodesToSameMasksAsEagerPath() {
    const nlohmann::json tasks = nlohmann::json::array({{
        {"id", 3},
        {"annotations", {{{"result", {brushItem("tag", squareMask(16, 2, 2, 5), "scratch")}}}}},
        {"predictions", {{{"result", {brushItem("tag", squareMask(16, 8, 8, 4), "dent"),
                                      brushItem("other", squareMask(16, 0, 0, 3), "ignored")}}}}},
    }});

    const auto encoded = buildEncodedMasksByTaskId(tasks, "tag");
    const DatasetMasksToDraw eager = masksToDrawForTask(tasks, "tag", 3);

    CHECK(encoded.count(3) == 1);
    if (encoded.count(3) == 0) {
        return;
    }
    const auto annotation = decodeDatasetMasks(encoded.at(3).annotationMasks);
    const auto prediction = decodeDatasetMasks(encoded.at(3).predictionMasks);
    CHECK(annotation.size() == eager.annotationMasks.size());
    CHECK(prediction.size() == eager.predictionMasks.size());
    CHECK(prediction.size() == 1);
    if (annotation.size() != 1 || prediction.size() != 1 || eager.annotationMasks.size() != 1
        || eager.predictionMasks.size() != 1) {
        return;
    }
    CHECK(annotation[0].className == "scratch");
    CHECK(cv::countNonZero(annotation[0].mask != eager.annotationMasks[0].mask) == 0);
    CHECK(cv::countNonZero(prediction[0].mask != eager.predictionMasks[0].mask) == 0);
}

void test_buildEncodedMasksByTaskId_emptyFromNameReturnsEmptyMap() {
    const nlohmann::json tasks = nlohmann::json::array({{{"id", 1}}});
    CHECK(buildEncodedMasksByTaskId(tasks, "").empty());
}

void test_buildEncodedMasksByTaskId_skipsOutOfRangeRle() {
    nlohmann::json bad = brushItem("tag", squareMask(8, 1, 1, 2), "a");
    bad["value"]["rle"][0] = 300;
    const nlohmann::json tasks = nlohmann::json::array({{
        {"id", 1},
        {"annotations", {{{"result", {bad, brushItem("tag", squareMask(8, 4, 4, 2), "b")}}}}},
    }});
    const auto encoded = buildEncodedMasksByTaskId(tasks, "tag");
    CHECK(encoded.count(1) == 1);
    if (encoded.count(1) == 0) {
        return;
    }
    CHECK(encoded.at(1).annotationMasks.size() == 1);
    if (encoded.at(1).annotationMasks.size() == 1) {
        CHECK(encoded.at(1).annotationMasks[0].className == "b");
    }
}

void test_summarizeDatasetTasks_keepsCreatedAtString() {
    const nlohmann::json tasks = nlohmann::json::array({
        {{"id", 1}, {"data", {{"image", "/a.jpg"}}}, {"created_at", "2026-09-10T10:00:00.123Z"}},
        {{"id", 2}, {"data", {{"image", "/b.jpg"}}}},
    });
    const auto summaries = summarizeDatasetTasks(tasks, "image");
    CHECK(summaries.size() == 2);
    if (summaries.size() == 2) {
        CHECK(summaries[0].createdAt == "2026-09-10T10:00:00.123Z");
        CHECK(summaries[1].createdAt.empty());
    }
}

void test_writeMatchingTasksAsJsonArrayElements_acrossPagesPreservesOrder() {
    const nlohmann::json page1 = nlohmann::json::array({{{"id", 1}, {"x", "a"}}, {{"id", 2}}});
    const nlohmann::json page2 = nlohmann::json::array({{{"id", 3}}, {{"id", 4}, {"x", "d"}}});
    std::unordered_set<int> ids = {1, 4};

    std::ostringstream out;
    bool wroteAny = false;
    out << "[\n";
    const size_t n1 = writeMatchingTasksAsJsonArrayElements(page1, ids, out, wroteAny);
    const size_t n2 = writeMatchingTasksAsJsonArrayElements(page2, ids, out, wroteAny);
    out << "\n]\n";

    CHECK(n1 == 1);
    CHECK(n2 == 1);
    const nlohmann::json parsed = nlohmann::json::parse(out.str());
    CHECK(parsed == nlohmann::json::array({{{"id", 1}, {"x", "a"}}, {{"id", 4}, {"x", "d"}}}));
}

void test_writeMatchingTasksAsJsonArrayElements_noMatchesStillValidJson() {
    const nlohmann::json page = nlohmann::json::array({{{"id", 1}}});
    std::ostringstream out;
    bool wroteAny = false;
    out << "[\n";
    std::unordered_set<int> none;
    CHECK(writeMatchingTasksAsJsonArrayElements(page, none, out, wroteAny) == 0);
    out << "\n]\n";
    CHECK(nlohmann::json::parse(out.str()) == nlohmann::json::array());
    CHECK(!wroteAny);
}

void test_buildEncodedMasksByTaskId_skipsWronglyTypedFieldsWithoutThrowing() {
    nlohmann::json nullWidth = brushItem("tag", squareMask(8, 1, 1, 2), "a");
    nullWidth["original_width"] = nullptr;
    nlohmann::json stringHeight = brushItem("tag", squareMask(8, 1, 1, 2), "b");
    stringHeight["original_height"] = "8";
    nlohmann::json negativeWidth = brushItem("tag", squareMask(8, 1, 1, 2), "c");
    negativeWidth["original_width"] = -8;
    nlohmann::json numericLabel = brushItem("tag", squareMask(8, 1, 1, 2), "d");
    numericLabel["value"]["brushlabels"] = {5};
    const nlohmann::json tasks = nlohmann::json::array({{
        {"id", 1},
        {"annotations", {{{"result", {nullWidth, stringHeight, negativeWidth, numericLabel,
                                      brushItem("tag", squareMask(8, 4, 4, 2), "ok")}}}}},
    }});
    bool threw = false;
    std::unordered_map<int, DatasetEncodedMasks> encoded;
    try {
        encoded = buildEncodedMasksByTaskId(tasks, "tag");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(!threw);
    CHECK(encoded.count(1) == 1);
    if (encoded.count(1) == 1) {
        CHECK(encoded.at(1).annotationMasks.size() == 1);
        CHECK(!encoded.at(1).annotationMasks.empty() && encoded.at(1).annotationMasks[0].className == "ok");
    }
}

void test_writeMatchingTasksAsJsonArrayElements_writesEachTaskOnceAcrossPages() {
    // Offset shift during the export's re-fetch re-delivers task 2 on the next page.
    const nlohmann::json page1 = nlohmann::json::array({{{"id", 1}}, {{"id", 2}}});
    const nlohmann::json page2 = nlohmann::json::array({{{"id", 2}}, {{"id", 3}}});
    std::unordered_set<int> remaining = {2, 3};

    std::ostringstream out;
    bool wroteAny = false;
    out << "[\n";
    writeMatchingTasksAsJsonArrayElements(page1, remaining, out, wroteAny);
    writeMatchingTasksAsJsonArrayElements(page2, remaining, out, wroteAny);
    out << "\n]\n";

    CHECK(nlohmann::json::parse(out.str()) == nlohmann::json::array({{{"id", 2}}, {{"id", 3}}}));
    CHECK(remaining.empty());
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
    test_idsToEvict_noEvictionWhenUnderCap();
    test_idsToEvict_evictsLeastRecentlyTouchedFirst();
    test_idsToEvict_neverEvictsCurrentlyVisibleIds();
    test_idsToEvict_visibleIdsAloneOverCapEvictsOnlyNonVisible();
    test_idsToEvict_exactlyAtCapEvictsNothing();
    test_boxesToDrawForTask_separatesAnnotationAndPredictionBoxes();
    test_boxesToDrawForTask_ignoresResultsWithDifferentFromName();
    test_boxesToDrawForTask_emptyFromNameReturnsNothing();
    test_boxesToDrawForTask_taskNotFoundReturnsEmpty();
    test_boxesToDrawForTask_preservesRotation();
    test_masksToDrawForTask_separatesAnnotationAndPredictionMasks();
    test_masksToDrawForTask_emptyFromNameReturnsNothing();
    test_masksToDrawForTask_taskNotFoundReturnsEmpty();
    test_buildBoxesByTaskId_matchesPerTaskLookupForEveryTask();
    test_buildBoxesByTaskId_emptyFromNameReturnsEmptyMap();
    test_buildMasksByTaskId_matchesPerTaskLookupForEveryTask();
    test_buildMasksByTaskId_emptyFromNameReturnsEmptyMap();
    test_indexSummariesByTaskId_mapsEachTaskIdToItsIndex();
    test_indexSummariesByTaskId_emptyInputReturnsEmptyMap();

    test_buildEncodedMasksByTaskId_decodesToSameMasksAsEagerPath();
    test_buildEncodedMasksByTaskId_emptyFromNameReturnsEmptyMap();
    test_buildEncodedMasksByTaskId_skipsOutOfRangeRle();
    test_summarizeDatasetTasks_keepsCreatedAtString();
    test_writeMatchingTasksAsJsonArrayElements_acrossPagesPreservesOrder();
    test_writeMatchingTasksAsJsonArrayElements_noMatchesStillValidJson();
    test_buildEncodedMasksByTaskId_skipsWronglyTypedFieldsWithoutThrowing();
    test_writeMatchingTasksAsJsonArrayElements_writesEachTaskOnceAcrossPages();
    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

#include "manager/label_studio_client.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}

// parseTypedLocalTimestamp depends on the process's configured timezone
// (via mktime), so tests that need a deterministic result pin TZ for the
// duration of the call and restore it afterward -- otherwise the expected
// epoch would depend on whatever timezone happens to be set on the
// machine running the tests.
class ScopedTz {
public:
    explicit ScopedTz(const char* tz) {
        const char* existing = std::getenv("TZ");
        hadPrevious_ = existing != nullptr;
        if (hadPrevious_) {
            previous_ = existing;
        }
        setenv("TZ", tz, 1);
        tzset();
    }
    ~ScopedTz() {
        if (hadPrevious_) {
            setenv("TZ", previous_.c_str(), 1);
        } else {
            unsetenv("TZ");
        }
        tzset();
    }
    ScopedTz(const ScopedTz&) = delete;
    ScopedTz& operator=(const ScopedTz&) = delete;

private:
    bool hadPrevious_ = false;
    std::string previous_;
};
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

void test_parseIso8601Utc_standardFormat() {
    const auto result = parseIso8601Utc("2024-05-01T12:34:56Z");
    CHECK(result.has_value());
    CHECK(*result == 1714566896);
}

void test_parseIso8601Utc_withFractionalSecondsAndZ() {
    const auto result = parseIso8601Utc("2024-05-01T12:34:56.789012Z");
    CHECK(result.has_value());
    CHECK(*result == 1714566896);
}

void test_parseIso8601Utc_malformedReturnsNullopt() {
    CHECK(!parseIso8601Utc("not a timestamp").has_value());
    CHECK(!parseIso8601Utc("").has_value());
}

void test_parseTypedLocalTimestamp_interpretedAsUtcWhenTzIsUtc() {
    ScopedTz tz("UTC");
    const auto result = parseTypedLocalTimestamp("2026/09/10 12:00:00");
    CHECK(result.has_value());
    CHECK(*result == 1789041600);
}

void test_parseTypedLocalTimestamp_convertsFromLocalTimezone() {
    // Etc/GMT-2 is a fixed UTC+2 offset (POSIX's Etc/GMT sign convention
    // is inverted). 2026-06-22T14:15:38 local (UTC+2) is
    // 2026-06-22T12:15:38Z -- a real-world case: a factory timestamp
    // engraved in local time, matched against Label Studio's UTC
    // created_at ("2026-06-22T12:15:38.416491Z" for the same piece).
    ScopedTz tz("Etc/GMT-2");
    const auto result = parseTypedLocalTimestamp("2026/06/22 14:15:38");
    CHECK(result.has_value());
    CHECK(*result == 1782130538);
}

void test_parseTypedLocalTimestamp_malformedReturnsNullopt() {
    ScopedTz tz("UTC");
    CHECK(!parseTypedLocalTimestamp("2026-09-10 12:00:00").has_value()); // wrong separators
    CHECK(!parseTypedLocalTimestamp("garbage").has_value());
    CHECK(!parseTypedLocalTimestamp("").has_value());
}

void test_matchTasksToTimestamps_withinToleranceAndSorted() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}},
        {"id": 2, "created_at": "2026-09-10T12:01:30Z", "data": {"image": "/a/2.jpg"}},
        {"id": 3, "created_at": "2026-09-10T12:05:00Z", "data": {"image": "/a/3.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600; // 2026-09-10T12:00:00Z
    query.toleranceSeconds = 120; // +/- 2 min

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results.size() == 1);
    CHECK(results[0].size() == 2); // task 3 (5 min away) excluded
    CHECK(results[0][0].taskId == 1); // closest first
    CHECK(results[0][0].deltaSeconds == 0);
    CHECK(results[0][1].taskId == 2);
    CHECK(results[0][1].deltaSeconds == 90);
}

void test_matchTasksToTimestamps_toleranceBoundaryInclusive() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:02:00Z", "data": {"image": "/a/1.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600;
    query.toleranceSeconds = 120;

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results[0].size() == 1); // exactly at the boundary -- included
}

void test_matchTasksToTimestamps_skipsTaskMissingCreatedAtOrDataKey() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"}},
        {"id": 2, "created_at": "2026-09-10T12:00:00Z", "data": {}},
        {"id": 3, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/3.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600;
    query.toleranceSeconds = 60;

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results[0].size() == 1);
    CHECK(results[0][0].taskId == 3);
}

void test_matchTasksToTimestamps_taskMatchesMultipleQueriesIndependently() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}}
    ])");

    TimestampMatchQuery queryA;
    queryA.timestamp = 1789041600;
    queryA.toleranceSeconds = 60;
    TimestampMatchQuery queryB;
    queryB.timestamp = 1789041630; // 30s later, overlapping window
    queryB.toleranceSeconds = 60;

    const auto results = matchTasksToTimestamps(tasks, "image", {queryA, queryB});
    CHECK(results.size() == 2);
    CHECK(results[0].size() == 1);
    CHECK(results[1].size() == 1);
    CHECK(results[0][0].taskId == 1);
    CHECK(results[1][0].taskId == 1);
}

void test_matchTasksToTimestamps_emptyQueriesReturnsEmpty() {
    const auto tasks = nlohmann::json::parse(R"([{"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}}])");
    CHECK(matchTasksToTimestamps(tasks, "image", {}).empty());
}

void test_parseLabelStudioProjectConfigXml_singleRectangleLabelsTag() {
    const std::string xml =
        R"(<View><Image name="image" value="$image"/>)"
        R"(<RectangleLabels name="label" toName="image">)"
        R"(<Label value="Person"/><Label value="Car"/></RectangleLabels></View>)";

    const auto config = parseLabelStudioProjectConfigXml(xml);
    CHECK(config.error.empty());
    CHECK(config.dataImageKey == "image");
    CHECK(config.controlTags.size() == 1);
    CHECK(config.controlTags[0].type == LabelStudioControlTagType::RectangleLabels);
    CHECK(config.controlTags[0].name == "label");
    CHECK(config.controlTags[0].toName == "image");
    CHECK(config.controlTags[0].labels.size() == 2);
    CHECK(config.controlTags[0].labels[0] == "Person");
    CHECK(config.controlTags[0].labels[1] == "Car");
}

void test_parseLabelStudioProjectConfigXml_choicesTag() {
    const std::string xml =
        R"(<View><Image name="image" value="$photo"/>)"
        R"(<Choices name="class" toName="image">)"
        R"(<Choice value="Good"/><Choice value="Defect"/></Choices></View>)";

    const auto config = parseLabelStudioProjectConfigXml(xml);
    CHECK(config.error.empty());
    CHECK(config.dataImageKey == "photo"); // leading '$' stripped
    CHECK(config.controlTags.size() == 1);
    CHECK(config.controlTags[0].type == LabelStudioControlTagType::Choices);
    CHECK(config.controlTags[0].labels.size() == 2);
    CHECK(config.controlTags[0].labels[1] == "Defect");
}

void test_parseLabelStudioProjectConfigXml_bothTagsPresent() {
    const std::string xml =
        R"(<View><Image name="image" value="$image"/>)"
        R"(<RectangleLabels name="label" toName="image"><Label value="Person"/></RectangleLabels>)"
        R"(<Choices name="class" toName="image"><Choice value="Good"/></Choices></View>)";

    const auto config = parseLabelStudioProjectConfigXml(xml);
    CHECK(config.error.empty());
    CHECK(config.controlTags.size() == 2);
}

void test_parseLabelStudioProjectConfigXml_noImageTagIsError() {
    const auto config = parseLabelStudioProjectConfigXml(R"(<View><RectangleLabels name="label" toName="image"/></View>)");
    CHECK(!config.error.empty());
}

void test_parseLabelStudioProjectConfigXml_malformedXmlIsError() {
    const auto config = parseLabelStudioProjectConfigXml("<View><Unclosed>");
    CHECK(!config.error.empty());
}

void test_parseLabelStudioProjectConfigXml_brushLabelsTag() {
    const std::string xml =
        R"(<View><Image name="image" value="$image"/>)"
        R"(<BrushLabels name="brush" toName="image">)"
        R"(<Label value="Defect"/><Label value="Scratch"/></BrushLabels></View>)";

    const auto config = parseLabelStudioProjectConfigXml(xml);
    CHECK(config.error.empty());
    CHECK(config.controlTags.size() == 1);
    CHECK(config.controlTags[0].type == LabelStudioControlTagType::BrushLabels);
    CHECK(config.controlTags[0].labels.size() == 2);
}

void test_selectAllTaskSummaries_flagsAnnotationsAndPredictions() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"}, "total_annotations": 1, "total_predictions": 0},
        {"id": 2, "data": {"image": "/a/2.jpg"}, "total_annotations": 0, "total_predictions": 2},
        {"id": 3, "data": {"image": "/a/3.jpg"}, "total_annotations": 0, "total_predictions": 0}
    ])");

    const auto summaries = selectAllTaskSummaries(tasks, "image");
    CHECK(summaries.size() == 3);
    CHECK(summaries[0].taskId == 1);
    CHECK(summaries[0].hasAnnotation == true);
    CHECK(summaries[0].hasPrediction == false);
    CHECK(summaries[1].hasAnnotation == false);
    CHECK(summaries[1].hasPrediction == true);
    CHECK(summaries[2].hasAnnotation == false);
    CHECK(summaries[2].hasPrediction == false);
}

void test_selectAllTaskSummaries_fallsBackToArrayLengths() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"}, "annotations": [{}], "predictions": []}
    ])");
    const auto summaries = selectAllTaskSummaries(tasks, "image");
    CHECK(summaries.size() == 1);
    CHECK(summaries[0].hasAnnotation == true);
    CHECK(summaries[0].hasPrediction == false);
}

void test_selectAllTaskSummaries_skipsTaskMissingIdOrImageKey() {
    const auto tasks = nlohmann::json::parse(R"([
        {"data": {"image": "/a/1.jpg"}},
        {"id": 2, "data": {}}
    ])");
    CHECK(selectAllTaskSummaries(tasks, "image").empty());
}

void test_parseLabelStudioProjects_bareArray() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "title": "Detection Project"},
        {"id": 2, "title": "Classification Project"}
    ])");
    const auto projects = parseLabelStudioProjects(json);
    CHECK(projects.size() == 2);
    CHECK(projects[0].id == 1);
    CHECK(projects[0].title == "Detection Project");
    CHECK(projects[1].id == 2);
    CHECK(projects[1].title == "Classification Project");
}

void test_parseLabelStudioProjects_wrappedInResults() {
    const auto json = nlohmann::json::parse(R"({
        "count": 1,
        "next": null,
        "results": [{"id": 7, "title": "Wrapped Project"}]
    })");
    const auto projects = parseLabelStudioProjects(json);
    CHECK(projects.size() == 1);
    CHECK(projects[0].id == 7);
    CHECK(projects[0].title == "Wrapped Project");
}

void test_parseLabelStudioProjects_skipsEntriesMissingIdOrTitle() {
    const auto json = nlohmann::json::parse(R"([
        {"id": 1, "title": "Has Both"},
        {"title": "Missing Id"},
        {"id": 2},
        {"id": 3, "title": "Has Both Too"}
    ])");
    const auto projects = parseLabelStudioProjects(json);
    CHECK(projects.size() == 2);
    CHECK(projects[0].id == 1);
    CHECK(projects[1].id == 3);
}

void test_parseLabelStudioProjects_notAnArrayReturnsEmpty() {
    const auto json = nlohmann::json::parse(R"({"detail": "not found"})");
    CHECK(parseLabelStudioProjects(json).empty());
}

void test_parseLabelStudioTaskDetail_withAnnotationAndPrediction() {
    const auto task = nlohmann::json::parse(R"({
        "id": 5,
        "data": {"image": "/data/upload/1/x.png"},
        "annotations": [{"id": 42, "result": [{"type": "choices", "value": {"choices": ["Good"]}}]}],
        "predictions": [{"result": [{"type": "choices", "value": {"choices": ["Defect"]}}]}]
    })");

    const auto detail = parseLabelStudioTaskDetail(task, "image");
    CHECK(detail.error.empty());
    CHECK(detail.taskId == 5);
    CHECK(detail.imagePath == "/data/upload/1/x.png");
    CHECK(detail.annotationId.has_value());
    CHECK(*detail.annotationId == 42);
    CHECK(detail.annotationResult.size() == 1);
    CHECK(detail.predictionResult.size() == 1);
}

void test_parseLabelStudioTaskDetail_noAnnotationOrPrediction() {
    const auto task = nlohmann::json::parse(R"({"id": 6, "data": {"image": "/a/6.jpg"}, "annotations": [], "predictions": []})");
    const auto detail = parseLabelStudioTaskDetail(task, "image");
    CHECK(detail.error.empty());
    CHECK(!detail.annotationId.has_value());
    CHECK(detail.annotationResult.empty());
    CHECK(detail.predictionResult.empty());
}

void test_parseLabelStudioTaskDetail_missingImageKeyIsError() {
    const auto task = nlohmann::json::parse(R"({"id": 7, "data": {}})");
    const auto detail = parseLabelStudioTaskDetail(task, "image");
    CHECK(!detail.error.empty());
}

void test_parseDetectionResultBoxes_convertsPercentToPixels() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
         "original_width": 200, "original_height": 100,
         "value": {"x": 10.0, "y": 20.0, "width": 50.0, "height": 25.0, "rectanglelabels": ["Person"]}}
    ])");

    const auto boxes = parseDetectionResultBoxes(result, "label");
    CHECK(boxes.size() == 1);
    CHECK(boxes[0].box.x == 20);      // 10% of 200
    CHECK(boxes[0].box.y == 20);      // 20% of 100
    CHECK(boxes[0].box.width == 100); // 50% of 200
    CHECK(boxes[0].box.height == 25); // 25% of 100
    CHECK(boxes[0].className == "Person");
    CHECK(boxes[0].rotationDegrees == 0.0f); // absent in the JSON -- defaults to unrotated
}

void test_parseDetectionResultBoxes_filtersByFromNameAndType() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "rectanglelabels", "from_name": "otherLabel", "to_name": "image",
         "original_width": 100, "original_height": 100,
         "value": {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "rectanglelabels": ["Car"]}},
        {"type": "choices", "from_name": "label", "to_name": "image", "value": {"choices": ["Good"]}}
    ])");
    CHECK(parseDetectionResultBoxes(result, "label").empty());
}

void test_parseDetectionResultBoxes_preservesRotation() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
         "original_width": 100, "original_height": 100,
         "value": {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "rotation": 15.0, "rectanglelabels": ["Car"]}}
    ])");
    const auto boxes = parseDetectionResultBoxes(result, "label");
    CHECK(boxes.size() == 1);
    CHECK(boxes[0].rotationDegrees == 15.0f);
}

void test_parseLabelStudioExport_preservesRotation() {
    // Real schema, verified against Label Studio's own docs.
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image": "http://x/opensource/label-studio/1.jpg"},
        "annotations": [{"result": [{
            "type": "rectanglelabels",
            "value": {"x": 50.8, "y": 5.87, "width": 12.4, "height": 10.46, "rotation": 45.0, "rectanglelabels": ["Moonwalker"]},
            "original_width": 600, "original_height": 403
        }]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].boxes.size() == 1);
    CHECK(result.images[0].boxes[0].rotationDegrees == 45.0f);
}

void test_buildDetectionPredictionResult_writesRotation() {
    DraftDetectionLabel draft;
    draft.imageFilename = "1.jpg";
    draft.imageWidth = 200;
    draft.imageHeight = 100;
    DraftDetectionBox box;
    box.box = cv::Rect(20, 20, 100, 25);
    box.className = "Person";
    box.confidence = 0.9f;
    box.rotationDegrees = 30.0f;
    draft.boxes.push_back(box);

    const auto prediction = buildDetectionPredictionResult(draft, "label", "image");
    CHECK(prediction.result.size() == 1);
    CHECK(prediction.result[0]["value"]["rotation"].get<double>() == 30.0);
}

void test_buildDetectionPredictionResult_defaultsRotationToZero() {
    DraftDetectionLabel draft;
    draft.imageWidth = 200;
    draft.imageHeight = 100;
    DraftDetectionBox box;
    box.box = cv::Rect(0, 0, 10, 10);
    box.className = "Car";
    draft.boxes.push_back(box);

    const auto prediction = buildDetectionPredictionResult(draft, "label", "image");
    CHECK(prediction.result.size() == 1);
    CHECK(prediction.result[0]["value"]["rotation"].get<double>() == 0.0);
}

void test_parseChoiceResultLabel_findsMatchingChoice() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "choices", "from_name": "class", "to_name": "image", "value": {"choices": ["Defect"]}}
    ])");
    const auto label = parseChoiceResultLabel(result, "class");
    CHECK(label.has_value());
    CHECK(*label == "Defect");
}

void test_parseChoiceResultLabel_noMatchReturnsNullopt() {
    const auto result = nlohmann::json::parse(R"([{"type": "rectanglelabels", "from_name": "label"}])");
    CHECK(!parseChoiceResultLabel(result, "class").has_value());
}

void test_parseChoiceResultLabel_emptyResultReturnsNullopt() {
    CHECK(!parseChoiceResultLabel(nlohmann::json::array(), "class").has_value());
}

cv::Mat maskFromRows(const std::vector<std::vector<int>>& rows) {
    const int height = static_cast<int>(rows.size());
    const int width = static_cast<int>(rows[0].size());
    cv::Mat mask(height, width, CV_8UC1);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            mask.at<uint8_t>(r, c) = static_cast<uint8_t>(rows[r][c]);
        }
    }
    return mask;
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x2AllZero() {
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({{0, 0}, {0, 0}}));
    const std::vector<int> expected = {0, 0, 0, 16, 57, 27, 253, 240, 0};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x2All255() {
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({{255, 255}, {255, 255}}));
    const std::vector<int> expected = {0, 0, 0, 16, 57, 27, 253, 255, 240};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x3OnePixel() {
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({{0, 0, 0}, {0, 255, 0}}));
    const std::vector<int> expected = {0, 0, 0, 24, 57, 27, 253, 240, 8, 255, 227, 0, 0};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x20LongRun() {
    std::vector<int> row(20, 255);
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({row}));
    const std::vector<int> expected = {0, 0, 0, 80, 57, 27, 254, 79, 255, 0};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x9Run() {
    std::vector<int> row(9, 255);
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({row}));
    const std::vector<int> expected = {0, 0, 0, 36, 57, 27, 254, 35, 255, 0};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x17Run() {
    std::vector<int> row(17, 255);
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({row}));
    const std::vector<int> expected = {0, 0, 0, 68, 57, 27, 254, 67, 255, 0};
    CHECK(rle == expected);
}

void test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x6Alternating() {
    const auto rle = encodeMaskToLabelStudioRle(maskFromRows({{0, 255, 0, 255, 0, 255}}));
    const std::vector<int> expected = {0, 0, 0, 24, 57, 27, 252, 96, 17, 255, 198, 1, 31, 252, 96, 17, 255, 128};
    CHECK(rle == expected);
}

void test_rleRoundTrip_preserves2DShape() {
    // 3 rows x 4 cols, distinct per-row pattern -- catches a row/col transposition bug.
    const auto mask = maskFromRows({{0, 255, 0, 255}, {255, 255, 0, 0}, {0, 0, 0, 255}});
    const auto rle = encodeMaskToLabelStudioRle(mask);
    const auto decoded = decodeLabelStudioRleToMask(rle, 4, 3);
    CHECK(decoded.rows == 3);
    CHECK(decoded.cols == 4);
    bool allMatch = true;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (decoded.at<uint8_t>(r, c) != mask.at<uint8_t>(r, c)) {
                allMatch = false;
            }
        }
    }
    CHECK(allMatch);
}

void test_rleRoundTrip_runLengthBoundary_8vs9() {
    std::vector<int> row8(8, 255);
    std::vector<int> row9(9, 255);
    const auto mask8 = maskFromRows({row8});
    const auto mask9 = maskFromRows({row9});
    const auto decoded8 = decodeLabelStudioRleToMask(encodeMaskToLabelStudioRle(mask8), 8, 1);
    const auto decoded9 = decodeLabelStudioRleToMask(encodeMaskToLabelStudioRle(mask9), 9, 1);
    bool match8 = true;
    for (int c = 0; c < 8; ++c) {
        if (decoded8.at<uint8_t>(0, c) != 255) match8 = false;
    }
    bool match9 = true;
    for (int c = 0; c < 9; ++c) {
        if (decoded9.at<uint8_t>(0, c) != 255) match9 = false;
    }
    CHECK(match8);
    CHECK(match9);
}

void test_rleRoundTrip_runLengthBoundary_256vs257() {
    std::vector<int> row256(256, 255);
    std::vector<int> row257(257, 255);
    const auto decoded256 = decodeLabelStudioRleToMask(encodeMaskToLabelStudioRle(maskFromRows({row256})), 256, 1);
    const auto decoded257 = decodeLabelStudioRleToMask(encodeMaskToLabelStudioRle(maskFromRows({row257})), 257, 1);
    bool match256 = true;
    for (int c = 0; c < 256; ++c) {
        if (decoded256.at<uint8_t>(0, c) != 255) match256 = false;
    }
    bool match257 = true;
    for (int c = 0; c < 257; ++c) {
        if (decoded257.at<uint8_t>(0, c) != 255) match257 = false;
    }
    CHECK(match256);
    CHECK(match257);
}

void test_rleRoundTrip_runLongerThan65536Chunks() {
    // 70000 identical pixels in one row forces the encoder's chunking path
    // (runs >65536 are split into multiple 16-bit-length-field runs).
    std::vector<int> row(70000, 255);
    const auto mask = maskFromRows({row});
    const auto rle = encodeMaskToLabelStudioRle(mask);
    const auto decoded = decodeLabelStudioRleToMask(rle, 70000, 1);
    bool allMatch = true;
    for (int c = 0; c < 70000; ++c) {
        if (decoded.at<uint8_t>(0, c) != 255) allMatch = false;
    }
    CHECK(allMatch);
}

void test_buildBrushLabelResult_oneItemPerRegion() {
    std::vector<DraftBrushRegion> regions;
    DraftBrushRegion region1;
    region1.mask = maskFromRows({{0, 255}, {255, 0}});
    region1.className = "Defect";
    regions.push_back(region1);
    DraftBrushRegion region2;
    region2.mask = maskFromRows({{255, 255}, {0, 0}});
    region2.className = "Scratch";
    regions.push_back(region2);

    const auto result = buildBrushLabelResult(regions, "brush", "image", 2, 2);
    CHECK(result.size() == 2);
    CHECK(result[0]["type"] == "brushlabels");
    CHECK(result[0]["from_name"] == "brush");
    CHECK(result[0]["to_name"] == "image");
    CHECK(result[0]["original_width"] == 2);
    CHECK(result[0]["original_height"] == 2);
    CHECK(result[0]["value"]["format"] == "rle");
    CHECK(result[0]["value"]["brushlabels"][0] == "Defect");
    CHECK(result[1]["value"]["brushlabels"][0] == "Scratch");
}

void test_buildBrushLabelResult_skipsEmptyMask() {
    std::vector<DraftBrushRegion> regions;
    DraftBrushRegion region;
    region.className = "Defect"; // mask left default-constructed (empty)
    regions.push_back(region);
    CHECK(buildBrushLabelResult(regions, "brush", "image", 10, 10).empty());
}

void test_parseBrushResultRegions_roundTripsThroughBuild() {
    std::vector<DraftBrushRegion> original;
    DraftBrushRegion region;
    region.mask = maskFromRows({{0, 255, 0}, {255, 255, 0}});
    region.className = "Defect";
    original.push_back(region);

    const auto built = buildBrushLabelResult(original, "brush", "image", 3, 2);
    const auto parsed = parseBrushResultRegions(built, "brush");

    CHECK(parsed.size() == 1);
    CHECK(parsed[0].className == "Defect");
    CHECK(parsed[0].mask.rows == 2);
    CHECK(parsed[0].mask.cols == 3);
    bool allMatch = true;
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (parsed[0].mask.at<uint8_t>(r, c) != original[0].mask.at<uint8_t>(r, c)) {
                allMatch = false;
            }
        }
    }
    CHECK(allMatch);
}

void test_parseBrushResultRegions_filtersByFromNameAndType() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "brushlabels", "from_name": "otherBrush", "to_name": "image",
         "original_width": 2, "original_height": 2,
         "value": {"format": "rle", "rle": [0], "brushlabels": ["Defect"]}},
        {"type": "rectanglelabels", "from_name": "brush", "to_name": "image"}
    ])");
    CHECK(parseBrushResultRegions(result, "brush").empty());
}

} // namespace

int main() {
    test_parseIso8601Utc_standardFormat();
    test_parseIso8601Utc_withFractionalSecondsAndZ();
    test_parseIso8601Utc_malformedReturnsNullopt();
    test_parseTypedLocalTimestamp_interpretedAsUtcWhenTzIsUtc();
    test_parseTypedLocalTimestamp_convertsFromLocalTimezone();
    test_parseTypedLocalTimestamp_malformedReturnsNullopt();
    test_matchTasksToTimestamps_withinToleranceAndSorted();
    test_matchTasksToTimestamps_toleranceBoundaryInclusive();
    test_matchTasksToTimestamps_skipsTaskMissingCreatedAtOrDataKey();
    test_matchTasksToTimestamps_taskMatchesMultipleQueriesIndependently();
    test_matchTasksToTimestamps_emptyQueriesReturnsEmpty();
    test_parseLabelStudioProjectConfigXml_singleRectangleLabelsTag();
    test_parseLabelStudioProjectConfigXml_choicesTag();
    test_parseLabelStudioProjectConfigXml_bothTagsPresent();
    test_parseLabelStudioProjectConfigXml_noImageTagIsError();
    test_parseLabelStudioProjectConfigXml_malformedXmlIsError();
    test_parseLabelStudioProjectConfigXml_brushLabelsTag();
    test_selectAllTaskSummaries_flagsAnnotationsAndPredictions();
    test_selectAllTaskSummaries_fallsBackToArrayLengths();
    test_selectAllTaskSummaries_skipsTaskMissingIdOrImageKey();
    test_parseLabelStudioProjects_bareArray();
    test_parseLabelStudioProjects_wrappedInResults();
    test_parseLabelStudioProjects_skipsEntriesMissingIdOrTitle();
    test_parseLabelStudioProjects_notAnArrayReturnsEmpty();
    test_parseLabelStudioTaskDetail_withAnnotationAndPrediction();
    test_parseLabelStudioTaskDetail_noAnnotationOrPrediction();
    test_parseLabelStudioTaskDetail_missingImageKeyIsError();
    test_parseDetectionResultBoxes_convertsPercentToPixels();
    test_parseDetectionResultBoxes_filtersByFromNameAndType();
    test_parseDetectionResultBoxes_preservesRotation();
    test_parseLabelStudioExport_preservesRotation();
    test_buildDetectionPredictionResult_writesRotation();
    test_buildDetectionPredictionResult_defaultsRotationToZero();
    test_parseChoiceResultLabel_findsMatchingChoice();
    test_parseChoiceResultLabel_noMatchReturnsNullopt();
    test_parseChoiceResultLabel_emptyResultReturnsNullopt();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x2AllZero();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x2All255();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_2x3OnePixel();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x20LongRun();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x9Run();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x17Run();
    test_encodeMaskToLabelStudioRle_matchesGroundTruth_1x6Alternating();
    test_rleRoundTrip_preserves2DShape();
    test_rleRoundTrip_runLengthBoundary_8vs9();
    test_rleRoundTrip_runLengthBoundary_256vs257();
    test_rleRoundTrip_runLongerThan65536Chunks();
    test_buildBrushLabelResult_oneItemPerRegion();
    test_buildBrushLabelResult_skipsEmptyMask();
    test_parseBrushResultRegions_roundTripsThroughBuild();
    test_parseBrushResultRegions_filtersByFromNameAndType();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

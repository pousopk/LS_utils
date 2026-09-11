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

void test_parseDetectionResultBoxes_skipsRotatedBox() {
    const auto result = nlohmann::json::parse(R"([
        {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
         "original_width": 100, "original_height": 100,
         "value": {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "rotation": 15.0, "rectanglelabels": ["Car"]}}
    ])");
    CHECK(parseDetectionResultBoxes(result, "label").empty());
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
    test_selectAllTaskSummaries_flagsAnnotationsAndPredictions();
    test_selectAllTaskSummaries_fallsBackToArrayLengths();
    test_selectAllTaskSummaries_skipsTaskMissingIdOrImageKey();
    test_parseLabelStudioTaskDetail_withAnnotationAndPrediction();
    test_parseLabelStudioTaskDetail_noAnnotationOrPrediction();
    test_parseLabelStudioTaskDetail_missingImageKeyIsError();
    test_parseDetectionResultBoxes_convertsPercentToPixels();
    test_parseDetectionResultBoxes_filtersByFromNameAndType();
    test_parseDetectionResultBoxes_skipsRotatedBox();
    test_parseChoiceResultLabel_findsMatchingChoice();
    test_parseChoiceResultLabel_noMatchReturnsNullopt();
    test_parseChoiceResultLabel_emptyResultReturnsNullopt();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

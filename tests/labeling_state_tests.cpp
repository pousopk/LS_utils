#include "manager/labeling_state.hpp"

#include "manager/label_studio_dataset_browser.hpp"

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

void test_compositeMaskOverlay_blendsOnlyMaskedPixels() {
    cv::Mat base(2, 2, CV_8UC3, cv::Scalar(0, 0, 0)); // solid black, BGR
    std::vector<DraftBrushRegion> regions;
    DraftBrushRegion region;
    region.mask = maskFromRows({{255, 0}, {0, 0}}); // only pixel (0,0) painted
    region.className = "Defect";
    regions.push_back(region);

    const cv::Mat result = compositeMaskOverlay(base, regions);
    CHECK(result.rows == 2);
    CHECK(result.cols == 2);
    // The painted pixel must differ from the original black.
    const cv::Vec3b paintedPixel = result.at<cv::Vec3b>(0, 0);
    CHECK(paintedPixel != cv::Vec3b(0, 0, 0));
    // An unpainted pixel must stay exactly the original color.
    const cv::Vec3b untouchedPixel = result.at<cv::Vec3b>(1, 1);
    CHECK(untouchedPixel == cv::Vec3b(0, 0, 0));
}

void test_compositeMaskOverlay_skipsRegionWithMismatchedSize() {
    cv::Mat base(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<DraftBrushRegion> regions;
    DraftBrushRegion region;
    region.mask = maskFromRows({{255, 255}, {255, 255}}); // 2x2, doesn't match 4x4 base
    region.className = "Defect";
    regions.push_back(region);

    const cv::Mat result = compositeMaskOverlay(base, regions);
    // Mismatched-size region is skipped entirely -- output equals input.
    CHECK(result.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30));
}

void test_colorForClassName_isDeterministic() {
    const auto a = colorForClassName("Person");
    const auto b = colorForClassName("Person");
    CHECK(a.r == b.r);
    CHECK(a.g == b.g);
    CHECK(a.b == b.b);
}

void test_colorForClassName_differentNamesLikelyDiffer() {
    const auto person = colorForClassName("Person");
    const auto car = colorForClassName("Car");
    CHECK(person.r != car.r || person.g != car.g || person.b != car.b);
}

void test_colorForClassName_emptyStringDoesNotCrash() {
    const auto color = colorForClassName("");
    (void)color;
}

void test_resetLabelingEditorsFromConfig_setsDefaultPendingNewBoxLabel() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person", "Car"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(state.boxEditor->pendingNewBoxLabel == "Person");
}

void test_resetLabelingEditorsFromConfig_boxOnly() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person", "Car"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(state.boxEditor.has_value());
    CHECK(!state.choiceEditor.has_value());
    CHECK(state.boxEditor->fromName == "label");
    CHECK(state.boxEditor->toName == "image");
    CHECK(state.boxEditor->availableLabels.size() == 2);
}

void test_resetLabelingEditorsFromConfig_choiceOnly() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "class", "image", {"Good", "Defect"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(!state.boxEditor.has_value());
    CHECK(state.choiceEditor.has_value());
    CHECK(state.choiceEditor->availableLabels.size() == 2);
}

void test_resetLabelingEditorsFromConfig_bothTags() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person"}},
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "class", "image", {"Good"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(state.boxEditor.has_value());
    CHECK(state.choiceEditor.has_value());
}

void test_resetLabelingEditorsFromConfig_brushOnly() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::BrushLabels, "brush", "image", {"Defect", "Scratch"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(state.maskEditor.has_value());
    CHECK(state.maskEditor->fromName == "brush");
    CHECK(state.maskEditor->toName == "image");
    CHECK(state.maskEditor->availableLabels.size() == 2);
    CHECK(state.maskEditor->pendingNewMaskLabel == "Defect");
}

void test_resetLabelingEditorsFromConfig_allThreeTags() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person"}},
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "class", "image", {"Good"}},
        LabelStudioControlTag{LabelStudioControlTagType::BrushLabels, "brush", "image", {"Defect"}}};

    resetLabelingEditorsFromConfig(state);

    CHECK(state.boxEditor.has_value());
    CHECK(state.choiceEditor.has_value());
    CHECK(state.maskEditor.has_value());
}

void test_applyTaskDetailToEditors_seedsMaskRegionsFromExistingAnnotation() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::BrushLabels, "brush", "image", {"Defect"}}};
    resetLabelingEditorsFromConfig(state);

    std::vector<DraftBrushRegion> seedRegions;
    DraftBrushRegion region;
    region.mask = maskFromRows({{0, 255}, {255, 0}});
    region.className = "Defect";
    seedRegions.push_back(region);

    LabelStudioTaskDetail detail;
    detail.annotationId = 7;
    detail.annotationResult = buildBrushLabelResult(seedRegions, "brush", "image", 2, 2);

    applyTaskDetailToEditors(state, detail);

    CHECK(state.maskEditor->regions.size() == 1);
    CHECK(state.maskEditor->regions[0].className == "Defect");
    CHECK(!state.maskEditor->dirty);
}

void test_anyEditorDirty_trueWhenMaskEditorDirty() {
    LabelingState state;
    state.maskEditor = BrushLabelEditorState{};
    state.maskEditor->dirty = true;
    CHECK(anyEditorDirty(state));
}

void test_buildCombinedAnnotationResult_includesMaskRegions() {
    LabelingState state;
    state.maskEditor = BrushLabelEditorState{};
    state.maskEditor->fromName = "brush";
    state.maskEditor->toName = "image";
    DraftBrushRegion region;
    region.mask = maskFromRows({{0, 255}, {255, 0}});
    region.className = "Defect";
    state.maskEditor->regions.push_back(region);

    const auto result = buildCombinedAnnotationResult(state, 2, 2);
    CHECK(result.size() == 1);
    CHECK(result[0]["type"] == "brushlabels");
}

void test_confirmDiscardAndSwitchTask_clearsMaskEditorDirty() {
    // Regression test: adding maskEditor's dirty-clearing to
    // confirmDiscardAndSwitchTask (and to updateLabelingState's
    // SubmitAnnotation success handling) was missed when the mask editor
    // was introduced, so the unsaved-changes prompt kept reappearing even
    // after a successful Submit/Discard.
    LabelingState state;
    state.maskEditor = BrushLabelEditorState{};
    state.maskEditor->dirty = true;
    state.unsavedPromptAction = LabelingUnsavedPromptAction::CloseWindow;

    const LabelStudioSessionState session;
    confirmDiscardAndSwitchTask(state, session);

    CHECK(!state.maskEditor->dirty);
    CHECK(!anyEditorDirty(state));
}

void test_nextLabelingTaskId_returnsNextTask() {
    LabelingState state;
    state.taskList = {
        LabelStudioTaskSummary{10, "/a/10.jpg", false, false},
        LabelStudioTaskSummary{20, "/a/20.jpg", false, false},
        LabelStudioTaskSummary{30, "/a/30.jpg", false, false}};
    state.selectedTaskId = 20;

    const auto next = nextLabelingTaskId(state, 1);
    CHECK(next.has_value());
    CHECK(*next == 30);
}

void test_nextLabelingTaskId_returnsPreviousTask() {
    LabelingState state;
    state.taskList = {
        LabelStudioTaskSummary{10, "/a/10.jpg", false, false},
        LabelStudioTaskSummary{20, "/a/20.jpg", false, false},
        LabelStudioTaskSummary{30, "/a/30.jpg", false, false}};
    state.selectedTaskId = 20;

    const auto prev = nextLabelingTaskId(state, -1);
    CHECK(prev.has_value());
    CHECK(*prev == 10);
}

void test_nextLabelingTaskId_nulloptPastEitherEnd() {
    LabelingState state;
    state.taskList = {
        LabelStudioTaskSummary{10, "/a/10.jpg", false, false},
        LabelStudioTaskSummary{20, "/a/20.jpg", false, false}};

    state.selectedTaskId = 20;
    CHECK(!nextLabelingTaskId(state, 1).has_value());

    state.selectedTaskId = 10;
    CHECK(!nextLabelingTaskId(state, -1).has_value());
}

void test_nextLabelingTaskId_nulloptWhenCurrentTaskNotFoundOrListEmpty() {
    LabelingState state;
    CHECK(!nextLabelingTaskId(state, 1).has_value()); // empty list

    state.taskList = {LabelStudioTaskSummary{10, "/a/10.jpg", false, false}};
    state.selectedTaskId = 999; // not in the list
    CHECK(!nextLabelingTaskId(state, 1).has_value());
}

void test_applyTaskDetailToEditors_seedsFromExistingAnnotation() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person"}}};
    resetLabelingEditorsFromConfig(state);

    LabelStudioTaskDetail detail;
    detail.annotationId = 42;
    detail.annotationResult = nlohmann::json::parse(R"([
        {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
         "original_width": 100, "original_height": 100,
         "value": {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "rectanglelabels": ["Person"]}}
    ])");
    detail.predictionResult = nlohmann::json::array(); // present but must be ignored since an annotation exists

    applyTaskDetailToEditors(state, detail);

    CHECK(state.currentAnnotationId.has_value());
    CHECK(*state.currentAnnotationId == 42);
    CHECK(state.boxEditor->boxes.size() == 1);
    CHECK(!state.boxEditor->dirty); // freshly loaded, not user-edited yet
}

void test_applyTaskDetailToEditors_fallsBackToPredictionWhenNoAnnotation() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "class", "image", {"Good", "Defect"}}};
    resetLabelingEditorsFromConfig(state);

    LabelStudioTaskDetail detail;
    detail.predictionResult = nlohmann::json::parse(
        R"([{"type": "choices", "from_name": "class", "to_name": "image", "value": {"choices": ["Defect"]}}])");

    applyTaskDetailToEditors(state, detail);

    CHECK(!state.currentAnnotationId.has_value());
    CHECK(state.choiceEditor->selectedLabel.has_value());
    CHECK(*state.choiceEditor->selectedLabel == "Defect");
}

void test_applyTaskDetailToEditors_blankWhenNeitherPresent() {
    LabelingState state;
    state.projectConfig.controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "label", "image", {"Person"}}};
    resetLabelingEditorsFromConfig(state);

    applyTaskDetailToEditors(state, LabelStudioTaskDetail{});

    CHECK(!state.currentAnnotationId.has_value());
    CHECK(state.boxEditor->boxes.empty());
}

void test_anyEditorDirty_falseWhenNeitherActiveOrDirty() {
    LabelingState state;
    CHECK(!anyEditorDirty(state));
}

void test_anyEditorDirty_trueWhenBoxEditorDirty() {
    LabelingState state;
    state.boxEditor = BoxLabelEditorState{};
    state.boxEditor->dirty = true;
    CHECK(anyEditorDirty(state));
}

void test_anyEditorDirty_trueWhenChoiceEditorDirty() {
    LabelingState state;
    state.choiceEditor = ChoiceLabelEditorState{};
    state.choiceEditor->dirty = true;
    CHECK(anyEditorDirty(state));
}

void test_buildCombinedAnnotationResult_mergesBothEditors() {
    LabelingState state;
    state.boxEditor = BoxLabelEditorState{};
    state.boxEditor->fromName = "label";
    state.boxEditor->toName = "image";
    state.boxEditor->boxes.push_back(DraftDetectionBox{cv::Rect(0, 0, 10, 10), "Person", 0.0f});
    state.choiceEditor = ChoiceLabelEditorState{};
    state.choiceEditor->fromName = "class";
    state.choiceEditor->toName = "image";
    state.choiceEditor->selectedLabel = "Defect";

    const auto result = buildCombinedAnnotationResult(state, 100, 100);
    CHECK(result.size() == 2);
}

void test_buildCombinedAnnotationResult_choiceOnlyWhenNoSelection() {
    LabelingState state;
    state.choiceEditor = ChoiceLabelEditorState{};
    state.choiceEditor->fromName = "class";
    state.choiceEditor->toName = "image";
    // selectedLabel left unset

    const auto result = buildCombinedAnnotationResult(state, 100, 100);
    CHECK(result.empty());
}

void test_rotatePointClockwise_ninetyDegreesAroundOrigin() {
    const cv::Point2f result = rotatePointClockwise(cv::Point2f(10.0f, 0.0f), cv::Point2f(0.0f, 0.0f), 90.0f);
    CHECK(std::abs(result.x - 0.0f) < 0.01f);
    CHECK(std::abs(result.y - 10.0f) < 0.01f);
}

void test_rotatePointClockwise_oneEightyDegreesAroundOffsetPivot() {
    const cv::Point2f result = rotatePointClockwise(cv::Point2f(10.0f, 0.0f), cv::Point2f(5.0f, 5.0f), 180.0f);
    CHECK(std::abs(result.x - 0.0f) < 0.01f);
    CHECK(std::abs(result.y - 10.0f) < 0.01f);
}

void test_rotatePointClockwise_zeroDegreesIsIdentity() {
    const cv::Point2f result = rotatePointClockwise(cv::Point2f(7.0f, 3.0f), cv::Point2f(1.0f, 1.0f), 0.0f);
    CHECK(std::abs(result.x - 7.0f) < 0.01f);
    CHECK(std::abs(result.y - 3.0f) < 0.01f);
}

void test_rotatedBoxCorners_topLeftStaysAtPivotRegardlessOfAngle() {
    const cv::Rect box(10, 20, 100, 50);
    const auto corners = rotatedBoxCorners(box, 37.0f);
    CHECK(std::abs(corners[0].x - 10.0f) < 0.01f);
    CHECK(std::abs(corners[0].y - 20.0f) < 0.01f);
}

void test_rotatedBoxContainsPoint_unrotatedMatchesPlainContains() {
    const cv::Rect box(10, 10, 20, 20);
    CHECK(rotatedBoxContainsPoint(box, 0.0f, cv::Point2f(15.0f, 15.0f)));
    CHECK(!rotatedBoxContainsPoint(box, 0.0f, cv::Point2f(5.0f, 5.0f)));
}

void test_rotatedBoxContainsPoint_rotatedBoxAcceptsPointInRotatedFootprint() {
    const cv::Rect box(0, 0, 10, 20); // rotated 90 deg around (0,0) covers x in [-20,0], y in [0,10]
    CHECK(rotatedBoxContainsPoint(box, 90.0f, cv::Point2f(-10.0f, 5.0f)));
    // The box's ORIGINAL (unrotated) footprint no longer contains points after rotating away from it.
    CHECK(!rotatedBoxContainsPoint(box, 90.0f, cv::Point2f(5.0f, 5.0f)));
}

void test_rotatedHandleAnchorPoint_zeroRotationMatchesUnrotatedCorner() {
    const cv::Rect box(10, 20, 30, 40);
    const cv::Point2f anchor = rotatedHandleAnchorPoint(0, box, 0.0f); // handle 0 = TL, anchor = BR
    CHECK(std::abs(anchor.x - 40.0f) < 0.01f); // 10 + 30
    CHECK(std::abs(anchor.y - 60.0f) < 0.01f); // 20 + 40
}

void test_rotatedHandleAnchorPoint_matchesRotatedBoxCorners() {
    const cv::Rect box(0, 0, 10, 20);
    const cv::Point2f anchor = rotatedHandleAnchorPoint(0, box, 90.0f); // handle 0 = TL, anchor = BR
    const auto corners = rotatedBoxCorners(box, 90.0f);
    CHECK(std::abs(anchor.x - corners[2].x) < 0.01f); // corners[2] = rotated BR
    CHECK(std::abs(anchor.y - corners[2].y) < 0.01f);
}

void test_resizeRotatedBox_zeroRotationMatchesPlainMinAbs() {
    // Anchor is the min corner: current mouse is further bottom-right.
    const cv::Rect result1 = resizeRotatedBox(0.0f, cv::Point2f(50.0f, 50.0f), cv::Point2f(80.0f, 70.0f));
    CHECK(result1.x == 50);
    CHECK(result1.y == 50);
    CHECK(result1.width == 31);  // abs(80-50)+1
    CHECK(result1.height == 21); // abs(70-50)+1

    // Anchor is the max corner: current mouse is further top-left (dragging
    // the opposite corner past the anchor in both axes).
    const cv::Rect result2 = resizeRotatedBox(0.0f, cv::Point2f(80.0f, 70.0f), cv::Point2f(50.0f, 50.0f));
    CHECK(result2.x == 50);
    CHECK(result2.y == 50);
    CHECK(result2.width == 31);
    CHECK(result2.height == 21);
}

void test_resizeRotatedBox_keepsAnchorFixedOnScreenWhenRotated() {
    const float rotation = 90.0f;
    const cv::Point2f anchor(20.0f, 30.0f);
    const cv::Rect result = resizeRotatedBox(rotation, anchor, cv::Point2f(60.0f, 55.0f));

    // The anchor corner must still be (up to 1px) one of the resulting
    // box's 4 rotated corners -- that's the whole point of anchoring a
    // resize. Tolerance is 1.5px, not the usual 0.01f: the pre-existing
    // (unrotated) resize convention this generalizes already sizes boxes
    // as abs(delta)+1 ("inclusive" sizing), which puts the anchor exactly
    // on the boundary only when it's the min-side corner in a given axis
    // -- on the max side it's exactly 1px shy of the boundary (confirmed
    // against the *original* axis-aligned formula too, e.g. anchor
    // (80,70)/current (50,50) -> box (50,50,31,21), whose far corner is
    // (81,71), not (80,70)). This is inherited behavior, not a bug.
    const auto corners = rotatedBoxCorners(result, rotation);
    bool foundAnchor = false;
    for (const auto& corner : corners) {
        if (std::abs(corner.x - anchor.x) < 1.5f && std::abs(corner.y - anchor.y) < 1.5f) {
            foundAnchor = true;
            break;
        }
    }
    CHECK(foundAnchor);
}

void test_rotateBoxAroundCenter_ninetyDegreeDelta() {
    // width=10,height=20 -> local center (5,10); starting unrotated, pivot (0,0)
    // so the on-screen center starts at (5,10).
    const cv::Rect originalBox(0, 0, 10, 20);
    const RotatedBoxAngleDrag result = rotateBoxAroundCenter(originalBox, 0.0f, 0.0f, static_cast<float>(CV_PI) / 2.0f);

    CHECK(std::abs(result.rotationDegrees - 90.0f) < 0.01f);
    CHECK(result.box.width == 10);
    CHECK(result.box.height == 20);
    CHECK(std::abs(static_cast<float>(result.box.x) - 15.0f) < 0.5f);
    CHECK(std::abs(static_cast<float>(result.box.y) - 5.0f) < 0.5f);
}

void test_rotateBoxAroundCenter_keepsCenterFixed() {
    const cv::Rect originalBox(3, 4, 12, 8);
    const float startRotation = 20.0f;
    const float startAngle = 0.3f;
    const float currentAngle = 1.1f;

    const cv::Point2f localCenter(originalBox.width / 2.0f, originalBox.height / 2.0f);
    const cv::Point2f pivotBefore(static_cast<float>(originalBox.x), static_cast<float>(originalBox.y));
    const cv::Point2f centerBefore = rotatePointClockwise(pivotBefore + localCenter, pivotBefore, startRotation);

    const RotatedBoxAngleDrag result = rotateBoxAroundCenter(originalBox, startRotation, startAngle, currentAngle);

    const cv::Point2f pivotAfter(static_cast<float>(result.box.x), static_cast<float>(result.box.y));
    const cv::Point2f centerAfter = rotatePointClockwise(pivotAfter + localCenter, pivotAfter, result.rotationDegrees);

    // Tolerance is 1.0px, not the usual 0.01f: result.box.x/y are ints
    // (cv::Rect), so the pivot this function computes exactly (verified
    // separately to agree with centerBefore to within 1e-15 before
    // rounding) picks up up to ~0.5px of rounding error per axis when
    // stored -- an expected consequence of integer box storage, not a
    // bug in the rotation math itself.
    CHECK(std::abs(centerBefore.x - centerAfter.x) < 1.0f);
    CHECK(std::abs(centerBefore.y - centerAfter.y) < 1.0f);
}

void test_rotatedBoxCorners_ninetyDegrees() {
    const cv::Rect box(0, 0, 10, 20); // TL(0,0) TR(10,0) BR(10,20) BL(0,20)
    const auto corners = rotatedBoxCorners(box, 90.0f);
    // TL rotates onto itself.
    CHECK(std::abs(corners[0].x - 0.0f) < 0.01f);
    CHECK(std::abs(corners[0].y - 0.0f) < 0.01f);
    // TR(10,0) -> (0,10)
    CHECK(std::abs(corners[1].x - 0.0f) < 0.01f);
    CHECK(std::abs(corners[1].y - 10.0f) < 0.01f);
    // BR(10,20) -> (-20,10)
    CHECK(std::abs(corners[2].x - (-20.0f)) < 0.01f);
    CHECK(std::abs(corners[2].y - 10.0f) < 0.01f);
    // BL(0,20) -> (-20,0)
    CHECK(std::abs(corners[3].x - (-20.0f)) < 0.01f);
    CHECK(std::abs(corners[3].y - 0.0f) < 0.01f);
}

void test_deriveLabelingTaskList_copiesTaskIdImagePathAndPresenceFlagsOnly() {
    const std::vector<DatasetTaskSummary> sharedSummaries = {
        DatasetTaskSummary{1, "/a.jpg", true, false, {"Cat"}, std::nullopt, std::nullopt},
        DatasetTaskSummary{2, "/b.jpg", false, true, {"Dog"}, 0.9f, 0.9f},
    };
    const auto taskList = deriveLabelingTaskList(sharedSummaries);
    CHECK(taskList.size() == 2);
    CHECK(taskList[0].taskId == 1);
    CHECK(taskList[0].imagePath == "/a.jpg");
    CHECK(taskList[0].hasAnnotation);
    CHECK(!taskList[0].hasPrediction);
    CHECK(taskList[1].taskId == 2);
    CHECK(!taskList[1].hasAnnotation);
    CHECK(taskList[1].hasPrediction);
}

} // namespace

int main() {
    test_compositeMaskOverlay_blendsOnlyMaskedPixels();
    test_compositeMaskOverlay_skipsRegionWithMismatchedSize();
    test_colorForClassName_isDeterministic();
    test_colorForClassName_differentNamesLikelyDiffer();
    test_colorForClassName_emptyStringDoesNotCrash();
    test_resetLabelingEditorsFromConfig_setsDefaultPendingNewBoxLabel();
    test_resetLabelingEditorsFromConfig_boxOnly();
    test_resetLabelingEditorsFromConfig_choiceOnly();
    test_resetLabelingEditorsFromConfig_bothTags();
    test_resetLabelingEditorsFromConfig_brushOnly();
    test_resetLabelingEditorsFromConfig_allThreeTags();
    test_applyTaskDetailToEditors_seedsMaskRegionsFromExistingAnnotation();
    test_anyEditorDirty_trueWhenMaskEditorDirty();
    test_buildCombinedAnnotationResult_includesMaskRegions();
    test_confirmDiscardAndSwitchTask_clearsMaskEditorDirty();
    test_nextLabelingTaskId_returnsNextTask();
    test_nextLabelingTaskId_returnsPreviousTask();
    test_nextLabelingTaskId_nulloptPastEitherEnd();
    test_nextLabelingTaskId_nulloptWhenCurrentTaskNotFoundOrListEmpty();
    test_applyTaskDetailToEditors_seedsFromExistingAnnotation();
    test_applyTaskDetailToEditors_fallsBackToPredictionWhenNoAnnotation();
    test_applyTaskDetailToEditors_blankWhenNeitherPresent();
    test_anyEditorDirty_falseWhenNeitherActiveOrDirty();
    test_anyEditorDirty_trueWhenBoxEditorDirty();
    test_anyEditorDirty_trueWhenChoiceEditorDirty();
    test_buildCombinedAnnotationResult_mergesBothEditors();
    test_buildCombinedAnnotationResult_choiceOnlyWhenNoSelection();
    test_rotatePointClockwise_ninetyDegreesAroundOrigin();
    test_rotatePointClockwise_oneEightyDegreesAroundOffsetPivot();
    test_rotatePointClockwise_zeroDegreesIsIdentity();
    test_rotatedBoxCorners_topLeftStaysAtPivotRegardlessOfAngle();
    test_rotatedBoxCorners_ninetyDegrees();
    test_rotatedBoxContainsPoint_unrotatedMatchesPlainContains();
    test_rotatedBoxContainsPoint_rotatedBoxAcceptsPointInRotatedFootprint();
    test_rotatedHandleAnchorPoint_zeroRotationMatchesUnrotatedCorner();
    test_rotatedHandleAnchorPoint_matchesRotatedBoxCorners();
    test_resizeRotatedBox_zeroRotationMatchesPlainMinAbs();
    test_resizeRotatedBox_keepsAnchorFixedOnScreenWhenRotated();
    test_rotateBoxAroundCenter_ninetyDegreeDelta();
    test_rotateBoxAroundCenter_keepsCenterFixed();
    test_deriveLabelingTaskList_copiesTaskIdImagePathAndPresenceFlagsOnly();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

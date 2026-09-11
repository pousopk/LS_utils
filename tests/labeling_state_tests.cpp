#include "manager/labeling_state.hpp"

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

} // namespace

int main() {
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
    test_applyTaskDetailToEditors_seedsFromExistingAnnotation();
    test_applyTaskDetailToEditors_fallsBackToPredictionWhenNoAnnotation();
    test_applyTaskDetailToEditors_blankWhenNeitherPresent();
    test_anyEditorDirty_falseWhenNeitherActiveOrDirty();
    test_anyEditorDirty_trueWhenBoxEditorDirty();
    test_anyEditorDirty_trueWhenChoiceEditorDirty();
    test_buildCombinedAnnotationResult_mergesBothEditors();
    test_buildCombinedAnnotationResult_choiceOnlyWhenNoSelection();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

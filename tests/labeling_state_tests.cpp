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
    test_resetLabelingEditorsFromConfig_boxOnly();
    test_resetLabelingEditorsFromConfig_choiceOnly();
    test_resetLabelingEditorsFromConfig_bothTags();
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

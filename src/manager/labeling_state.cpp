#include "manager/labeling_state.hpp"

void resetLabelingEditorsFromConfig(LabelingState& state) {
    state.boxEditor.reset();
    state.choiceEditor.reset();

    for (const auto& tag : state.projectConfig.controlTags) {
        if (tag.type == LabelStudioControlTagType::RectangleLabels) {
            BoxLabelEditorState editor;
            editor.fromName = tag.name;
            editor.toName = tag.toName;
            editor.availableLabels = tag.labels;
            state.boxEditor = std::move(editor);
        } else if (tag.type == LabelStudioControlTagType::Choices) {
            ChoiceLabelEditorState editor;
            editor.fromName = tag.name;
            editor.toName = tag.toName;
            editor.availableLabels = tag.labels;
            state.choiceEditor = std::move(editor);
        }
        // BrushLabels: not handled until phase 3.
    }
}

void applyTaskDetailToEditors(LabelingState& state, const LabelStudioTaskDetail& detail) {
    state.currentAnnotationId = detail.annotationId;
    const bool hasAnnotation = detail.annotationId.has_value();
    const nlohmann::json& sourceResult = hasAnnotation ? detail.annotationResult : detail.predictionResult;

    if (state.boxEditor) {
        state.boxEditor->boxes = parseDetectionResultBoxes(sourceResult, state.boxEditor->fromName);
        state.boxEditor->selectedBoxIndex = -1;
        state.boxEditor->dirty = false;
    }
    if (state.choiceEditor) {
        state.choiceEditor->selectedLabel = parseChoiceResultLabel(sourceResult, state.choiceEditor->fromName);
        state.choiceEditor->dirty = false;
    }
}

bool anyEditorDirty(const LabelingState& state) {
    if (state.boxEditor && state.boxEditor->dirty) {
        return true;
    }
    if (state.choiceEditor && state.choiceEditor->dirty) {
        return true;
    }
    return false;
}

nlohmann::json buildCombinedAnnotationResult(const LabelingState& state, int imageWidth, int imageHeight) {
    nlohmann::json combined = nlohmann::json::array();

    if (state.boxEditor && !state.boxEditor->boxes.empty()) {
        DraftDetectionLabel draft;
        draft.imageWidth = imageWidth;
        draft.imageHeight = imageHeight;
        draft.boxes = state.boxEditor->boxes;
        const auto boxResult = buildDetectionPredictionResult(draft, state.boxEditor->fromName, state.boxEditor->toName);
        for (const auto& item : boxResult.result) {
            combined.push_back(item);
        }
    }

    if (state.choiceEditor && state.choiceEditor->selectedLabel.has_value()) {
        DraftClassificationLabel draft;
        draft.predictedLabel = *state.choiceEditor->selectedLabel;
        const auto choiceResult =
            buildClassificationPredictionResult(draft, state.choiceEditor->fromName, state.choiceEditor->toName);
        for (const auto& item : choiceResult.result) {
            combined.push_back(item);
        }
    }

    return combined;
}

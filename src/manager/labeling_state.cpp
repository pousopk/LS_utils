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

void updateLabelingState(LabelingState& state) {
    const std::string key = state.labelStudioBaseUrl + "|" + std::to_string(state.labelStudioProjectId) + "|"
        + state.labelStudioApiToken;
    if (!state.labelStudioBaseUrl.empty() && state.labelStudioProjectId > 0 && !state.labelStudioApiToken.empty()
        && key != state.lastAutoFetchKey) {
        state.lastAutoFetchKey = key;
        state.projectConfig =
            fetchLabelStudioProjectConfigDetailed(state.labelStudioBaseUrl, state.labelStudioProjectId, state.labelStudioApiToken);
        if (!state.projectConfig.error.empty()) {
            state.configStatus = "Config error: " + state.projectConfig.error;
        } else {
            state.configStatus = "Loaded " + std::to_string(state.projectConfig.controlTags.size()) + " control tag(s)";
            resetLabelingEditorsFromConfig(state);
        }
    }

    LabelingJobResult result;
    if (!state.worker.tryTakeResult(result)) {
        return;
    }

    switch (result.kind) {
        case LabelingJobKind::FetchTaskList:
            state.taskListLoading = false;
            state.taskList = result.taskList;
            state.taskListError = result.taskListError;
            break;
        case LabelingJobKind::FetchTaskDetail:
            if (!result.taskDetailError.empty()) {
                state.taskLoadState = LabelingTaskLoadState::Failed;
                state.taskLoadError = result.taskDetailError;
                break;
            }
            applyTaskDetailToEditors(state, result.taskDetail);
            state.taskLoadState = LabelingTaskLoadState::Loaded;
            state.taskLoadError.clear();
            // Image texture upload (uploadFrameToTexture) happens in the
            // window layer (Task 8), which owns GL state -- this function
            // stays free of GL/ImGui includes, so it only records the
            // local path for the window to pick up.
            state.pendingLocalImagePath = result.localImagePath;
            break;
        case LabelingJobKind::SubmitAnnotation:
            state.submitInProgress = false;
            if (result.submitSuccess) {
                state.submitStatus = "Saved.";
                if (state.boxEditor) {
                    state.boxEditor->dirty = false;
                }
                if (state.choiceEditor) {
                    state.choiceEditor->dirty = false;
                }
            } else {
                state.submitStatus = "Save failed: " + result.submitError;
            }
            break;
    }
}

void requestSelectLabelingTask(LabelingState& state, int taskId) {
    if (anyEditorDirty(state)) {
        state.unsavedPromptOpen = true;
        state.unsavedPromptAction = LabelingUnsavedPromptAction::SwitchTask;
        state.unsavedPromptPendingTaskId = taskId;
        return;
    }

    state.selectedTaskId = taskId;
    state.taskLoadState = LabelingTaskLoadState::Loading;
    state.currentAnnotationId.reset();

    LabelingJobRequest request;
    request.kind = LabelingJobKind::FetchTaskDetail;
    request.taskDetailJob.baseUrl = state.labelStudioBaseUrl;
    request.taskDetailJob.apiToken = state.labelStudioApiToken;
    request.taskDetailJob.taskId = taskId;
    request.taskDetailJob.dataImageKey = state.projectConfig.dataImageKey;
    request.taskDetailJob.scratchFolderPath = state.scratchFolderPath;
    state.worker.start(std::move(request));
}

void beginSubmitLabelingAnnotation(LabelingState& state) {
    state.submitInProgress = true;
    state.submitStatus.clear();

    LabelingJobRequest request;
    request.kind = LabelingJobKind::SubmitAnnotation;
    request.submitJob.baseUrl = state.labelStudioBaseUrl;
    request.submitJob.apiToken = state.labelStudioApiToken;
    request.submitJob.taskId = state.selectedTaskId;
    request.submitJob.existingAnnotationId = state.currentAnnotationId;
    request.submitJob.resultArray = buildCombinedAnnotationResult(state, state.imageWidth, state.imageHeight);
    state.worker.start(std::move(request));
}

void confirmDiscardAndSwitchTask(LabelingState& state) {
    const int pendingTaskId = state.unsavedPromptPendingTaskId;
    const LabelingUnsavedPromptAction pendingAction = state.unsavedPromptAction;
    state.unsavedPromptOpen = false;
    state.unsavedPromptAction = LabelingUnsavedPromptAction::None;
    state.unsavedPromptPendingTaskId = -1;

    if (state.boxEditor) {
        state.boxEditor->dirty = false;
    }
    if (state.choiceEditor) {
        state.choiceEditor->dirty = false;
    }

    if (pendingAction == LabelingUnsavedPromptAction::SwitchTask) {
        requestSelectLabelingTask(state, pendingTaskId);
    }
    // CloseWindow: the window itself (Task 11) closes on seeing the prompt cleared with no pending switch.
}

void confirmSaveAndSwitchTask(LabelingState& state) {
    // Closing the prompt here and leaving unsavedPromptPendingTaskId/
    // unsavedPromptAction set lets the window (Task 11) re-check
    // submitInProgress each frame and re-issue the pending switch/close
    // once the save completes, without LabelingState scheduling anything
    // across frames itself.
    state.unsavedPromptOpen = false;
    beginSubmitLabelingAnnotation(state);
}

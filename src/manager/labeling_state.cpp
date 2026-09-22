#include "manager/labeling_state.hpp"

#include <cmath>
#include <cstdint>

bool rotatedBoxContainsPoint(const cv::Rect& box, float rotationDegrees, cv::Point2f point) {
    const cv::Point2f pivot(static_cast<float>(box.x), static_cast<float>(box.y));
    const cv::Point2f local = rotatePointClockwise(point, pivot, -rotationDegrees);
    return local.x >= static_cast<float>(box.x) && local.x <= static_cast<float>(box.x + box.width)
        && local.y >= static_cast<float>(box.y) && local.y <= static_cast<float>(box.y + box.height);
}

cv::Point2f rotatedHandleAnchorPoint(int handleIndex, const cv::Rect& box, float rotationDegrees) {
    cv::Point2f localAnchor;
    switch (handleIndex) {
        case 0: localAnchor = cv::Point2f(static_cast<float>(box.x + box.width), static_cast<float>(box.y + box.height)); break;
        case 1: localAnchor = cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y + box.height)); break;
        case 2: localAnchor = cv::Point2f(static_cast<float>(box.x + box.width), static_cast<float>(box.y)); break;
        default: localAnchor = cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y)); break;
    }
    const cv::Point2f pivot(static_cast<float>(box.x), static_cast<float>(box.y));
    return rotatePointClockwise(localAnchor, pivot, rotationDegrees);
}

cv::Rect resizeRotatedBox(float rotationDegrees, cv::Point2f anchorImage, cv::Point2f currentMouseImage) {
    const cv::Point2f currentLocal =
        rotatePointClockwise(currentMouseImage, anchorImage, -rotationDegrees) - anchorImage;
    const float newLocalX = std::min(0.0f, currentLocal.x);
    const float newLocalY = std::min(0.0f, currentLocal.y);
    const float newWidth = std::abs(currentLocal.x) + 1.0f;
    const float newHeight = std::abs(currentLocal.y) + 1.0f;

    const cv::Point2f newPivot =
        rotatePointClockwise(anchorImage + cv::Point2f(newLocalX, newLocalY), anchorImage, rotationDegrees);

    cv::Rect result;
    result.x = static_cast<int>(std::round(newPivot.x));
    result.y = static_cast<int>(std::round(newPivot.y));
    result.width = static_cast<int>(std::round(newWidth));
    result.height = static_cast<int>(std::round(newHeight));
    return result;
}

RotatedBoxAngleDrag rotateBoxAroundCenter(
    const cv::Rect& originalBox, float startRotationDegrees, float startAngleRadians, float currentAngleRadians) {
    const cv::Point2f pivot(static_cast<float>(originalBox.x), static_cast<float>(originalBox.y));
    const cv::Point2f localCenter(originalBox.width / 2.0f, originalBox.height / 2.0f);
    const cv::Point2f center = rotatePointClockwise(pivot + localCenter, pivot, startRotationDegrees);

    const float deltaDegrees = (currentAngleRadians - startAngleRadians) * 180.0f / static_cast<float>(CV_PI);
    const float newRotationDegrees = startRotationDegrees + deltaDegrees;

    // Pure rotation of the center->pivot offset vector (rotating around the
    // origin, not around a point) -- gives where the pivot must sit,
    // relative to the fixed center, at the new angle.
    const cv::Point2f rotatedCenterOffset =
        rotatePointClockwise(localCenter, cv::Point2f(0.0f, 0.0f), newRotationDegrees);
    const cv::Point2f newPivot = center - rotatedCenterOffset;

    RotatedBoxAngleDrag out;
    out.box = originalBox;
    out.box.x = static_cast<int>(std::round(newPivot.x));
    out.box.y = static_cast<int>(std::round(newPivot.y));
    out.rotationDegrees = newRotationDegrees;
    return out;
}

void resetLabelingEditorsFromConfig(LabelingState& state) {
    state.boxEditor.reset();
    state.choiceEditor.reset();
    state.maskEditor.reset();

    for (const auto& tag : state.projectConfig.controlTags) {
        if (tag.type == LabelStudioControlTagType::RectangleLabels) {
            BoxLabelEditorState editor;
            editor.fromName = tag.name;
            editor.toName = tag.toName;
            editor.availableLabels = tag.labels;
            if (!tag.labels.empty()) {
                editor.pendingNewBoxLabel = tag.labels.front();
            }
            state.boxEditor = std::move(editor);
        } else if (tag.type == LabelStudioControlTagType::Choices) {
            ChoiceLabelEditorState editor;
            editor.fromName = tag.name;
            editor.toName = tag.toName;
            editor.availableLabels = tag.labels;
            state.choiceEditor = std::move(editor);
        } else if (tag.type == LabelStudioControlTagType::BrushLabels) {
            BrushLabelEditorState editor;
            editor.fromName = tag.name;
            editor.toName = tag.toName;
            editor.availableLabels = tag.labels;
            if (!tag.labels.empty()) {
                editor.pendingNewMaskLabel = tag.labels.front();
            }
            state.maskEditor = std::move(editor);
        }
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
    if (state.maskEditor) {
        state.maskEditor->regions = parseBrushResultRegions(sourceResult, state.maskEditor->fromName);
        state.maskEditor->selectedRegionIndex = -1;
        state.maskEditor->dirty = false;
    }
}

bool anyEditorDirty(const LabelingState& state) {
    if (state.boxEditor && state.boxEditor->dirty) {
        return true;
    }
    if (state.choiceEditor && state.choiceEditor->dirty) {
        return true;
    }
    if (state.maskEditor && state.maskEditor->dirty) {
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

    if (state.maskEditor && !state.maskEditor->regions.empty()) {
        const auto maskResult = buildBrushLabelResult(
            state.maskEditor->regions, state.maskEditor->fromName, state.maskEditor->toName, imageWidth, imageHeight);
        for (const auto& item : maskResult) {
            combined.push_back(item);
        }
    }

    return combined;
}

void updateLabelingState(LabelingState& state, const LabelStudioSessionState& session) {
    const std::string key =
        session.baseUrl + "|" + std::to_string(session.activeProjectId) + "|" + session.apiToken;
    if (!session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty()
        && key != state.lastAutoFetchKey) {
        state.lastAutoFetchKey = key;
        state.projectConfig =
            fetchLabelStudioProjectConfigDetailed(session.baseUrl, session.activeProjectId, session.apiToken);
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
                if (state.maskEditor) {
                    state.maskEditor->dirty = false;
                }
            } else {
                state.submitStatus = "Save failed: " + result.submitError;
            }
            break;
    }
}

void requestSelectLabelingTask(LabelingState& state, const LabelStudioSessionState& session, int taskId) {
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
    request.taskDetailJob.baseUrl = session.baseUrl;
    request.taskDetailJob.apiToken = session.apiToken;
    request.taskDetailJob.taskId = taskId;
    request.taskDetailJob.dataImageKey = state.projectConfig.dataImageKey;
    request.taskDetailJob.scratchFolderPath = state.scratchFolderPath;
    state.worker.start(std::move(request));
}

std::optional<int> nextLabelingTaskId(const LabelingState& state, int direction) {
    int currentIndex = -1;
    for (int i = 0; i < static_cast<int>(state.taskList.size()); ++i) {
        if (state.taskList[i].taskId == state.selectedTaskId) {
            currentIndex = i;
            break;
        }
    }
    if (currentIndex < 0) {
        return std::nullopt;
    }

    const int targetIndex = currentIndex + direction;
    if (targetIndex < 0 || targetIndex >= static_cast<int>(state.taskList.size())) {
        return std::nullopt;
    }
    return state.taskList[targetIndex].taskId;
}

void beginSubmitLabelingAnnotation(LabelingState& state, const LabelStudioSessionState& session) {
    state.submitInProgress = true;
    state.submitStatus.clear();

    LabelingJobRequest request;
    request.kind = LabelingJobKind::SubmitAnnotation;
    request.submitJob.baseUrl = session.baseUrl;
    request.submitJob.apiToken = session.apiToken;
    request.submitJob.taskId = state.selectedTaskId;
    request.submitJob.existingAnnotationId = state.currentAnnotationId;
    request.submitJob.resultArray = buildCombinedAnnotationResult(state, state.imageWidth, state.imageHeight);
    state.worker.start(std::move(request));
}

void confirmDiscardAndSwitchTask(LabelingState& state, const LabelStudioSessionState& session) {
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
    if (state.maskEditor) {
        state.maskEditor->dirty = false;
    }

    if (pendingAction == LabelingUnsavedPromptAction::SwitchTask) {
        requestSelectLabelingTask(state, session, pendingTaskId);
    }
    // CloseWindow: the window itself (Task 11) closes on seeing the prompt cleared with no pending switch.
}

void confirmSaveAndSwitchTask(LabelingState& state, const LabelStudioSessionState& session) {
    // Closing the prompt here and leaving unsavedPromptPendingTaskId/
    // unsavedPromptAction set lets the window (Task 11) re-check
    // submitInProgress each frame and re-issue the pending switch/close
    // once the save completes, without LabelingState scheduling anything
    // across frames itself.
    state.unsavedPromptOpen = false;
    beginSubmitLabelingAnnotation(state, session);
}


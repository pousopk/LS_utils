#include "manager/labeling_state.hpp"

#include <cmath>
#include <cstdint>

LabelColor colorForClassName(const std::string& className) {
    // FNV-1a: simple, deterministic, well-distributed for short strings.
    uint32_t hash = 2166136261u;
    for (const char c : className) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 16777619u;
    }

    const float hue = static_cast<float>(hash % 360u);
    const float saturation = 0.55f;
    const float value = 0.85f;

    // Standard HSV -> RGB conversion.
    const float c = value * saturation;
    const float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    const float m = value - c;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (hue < 60.0f) {
        r = c; g = x; b = 0.0f;
    } else if (hue < 120.0f) {
        r = x; g = c; b = 0.0f;
    } else if (hue < 180.0f) {
        r = 0.0f; g = c; b = x;
    } else if (hue < 240.0f) {
        r = 0.0f; g = x; b = c;
    } else if (hue < 300.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }

    LabelColor color;
    color.r = static_cast<unsigned char>((r + m) * 255.0f);
    color.g = static_cast<unsigned char>((g + m) * 255.0f);
    color.b = static_cast<unsigned char>((b + m) * 255.0f);
    return color;
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
                if (state.maskEditor) {
                    state.maskEditor->dirty = false;
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
    if (state.maskEditor) {
        state.maskEditor->dirty = false;
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

cv::Mat compositeMaskOverlay(const cv::Mat& baseImage, const std::vector<DraftBrushRegion>& regions) {
    cv::Mat result = baseImage.clone();
    constexpr double kMaskAlpha = 0.45;

    for (const auto& region : regions) {
        if (region.mask.empty() || region.mask.size() != baseImage.size()) {
            continue;
        }
        const LabelColor color = colorForClassName(region.className);
        cv::Mat colorLayer(baseImage.size(), baseImage.type(), cv::Scalar(color.b, color.g, color.r));
        cv::Mat blended;
        cv::addWeighted(result, 1.0 - kMaskAlpha, colorLayer, kMaskAlpha, 0.0, blended);
        blended.copyTo(result, region.mask);
    }

    return result;
}

#include "manager/label_assistant_state.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>

void loadLabelAssistantModel(LabelAssistantState& state) {
    loadModelSlot(state.modelConfig, state.taskMode);
}

void applyLabelAssistantAutoDetect(LabelAssistantState& state) {
    applyAutoDetectToModelSlot(state.modelConfig, state.taskMode);
    if (state.taskMode == ComparisonTaskMode::Anomaly) {
        state.taskMode = ComparisonTaskMode::Classification;
        state.modelConfig.autoDetectStatus += " (Anomaly not supported here; defaulted to Classification)";
    }
}

namespace {

// Fixed, hidden scratch location for LabelStudioProject source mode's
// downloaded images -- cleared and recreated at the start of every run,
// so no leftover files accumulate across runs and nothing needs to be
// cleaned up on window close.
std::string labelAssistantScratchFolder() {
    return (std::filesystem::temp_directory_path() / "vision_app_label_assistant_download").string();
}

} // namespace

void startLabelAssistantRun(LabelAssistantState& state) {
    LabelAssistantRunConfig config;
    config.mode = state.taskMode;
    config.detectionModel = state.modelConfig.detectionModel;
    config.classificationModel = state.modelConfig.classificationModel;
    config.confThreshold = state.modelConfig.confThreshold;
    config.nmsThreshold = state.modelConfig.nmsThreshold;
    config.source = state.sourceMode;

    if (state.sourceMode == LabelAssistantSourceMode::LabelStudioProject) {
        const std::string scratchFolder = labelAssistantScratchFolder();
        std::error_code ec;
        std::filesystem::remove_all(scratchFolder, ec);
        std::filesystem::create_directories(scratchFolder, ec);

        config.labelStudioBaseUrl = state.labelStudioBaseUrl;
        config.labelStudioProjectId = state.labelStudioProjectId;
        config.labelStudioApiToken = state.labelStudioApiToken;
        config.labelStudioDataImageKey = state.labelStudioDataImageKey;
        config.scratchFolderPath = scratchFolder;
        state.imageFolderPath = scratchFolder;
    } else {
        config.imageFolderPath = state.imageFolderPath;
    }

    state.result = LabelAssistantResult{};
    state.selectedImageFilename.reset();

    state.worker.start(std::move(config));
    state.runState = LabelAssistantRunState::Running;
}

namespace {

const DraftDetectionLabel* findDetectionDraft(const LabelAssistantResult& result, const std::string& filename) {
    for (const auto& draft : result.detectionDrafts) {
        if (draft.imageFilename == filename) {
            return &draft;
        }
    }
    return nullptr;
}

std::vector<Detection> toDetections(const DraftDetectionLabel& draft) {
    std::vector<Detection> detections;
    detections.reserve(draft.boxes.size());
    for (const auto& box : draft.boxes) {
        Detection detection;
        detection.box = box.box;
        detection.className = box.className;
        detection.confidence = box.confidence;
        detections.push_back(std::move(detection));
    }
    return detections;
}

void syncLabelAssistantSelectedPreview(LabelAssistantState& state) {
    if (state.selectedImageFilename == state.renderedPreviewFilename) {
        return;
    }
    state.renderedPreviewFilename = state.selectedImageFilename;

    if (!state.selectedImageFilename) {
        return;
    }
    const cv::Mat frame = cv::imread(state.imageFolderPath + "/" + *state.selectedImageFilename);
    if (frame.empty()) {
        return;
    }

    cv::Mat toUpload = frame;
    if (state.taskMode == ComparisonTaskMode::Detection) {
        const DraftDetectionLabel* draft = findDetectionDraft(state.result, *state.selectedImageFilename);
        if (draft != nullptr) {
            toUpload = annotateDetections(frame, toDetections(*draft));
        }
    }

    if (state.previewTexture == 0) {
        glGenTextures(1, &state.previewTexture);
    }
    uploadFrameToTexture(state.previewTexture, toUpload, state.previewTextureWidth, state.previewTextureHeight);
}

std::string buildAutoFetchKey(const LabelAssistantState& state) {
    return state.labelStudioBaseUrl + "|" + std::to_string(state.labelStudioProjectId) + "|"
        + state.labelStudioApiToken + "|" + (state.taskMode == ComparisonTaskMode::Detection ? "D" : "C");
}

void syncLabelAssistantAutoFetch(LabelAssistantState& state) {
    if (state.labelStudioBaseUrl.empty() || state.labelStudioProjectId <= 0 || state.labelStudioApiToken.empty()) {
        return;
    }
    const std::string key = buildAutoFetchKey(state);
    if (key == state.lastAutoFetchKey) {
        return;
    }
    state.lastAutoFetchKey = key;

    const bool isDetection = state.taskMode == ComparisonTaskMode::Detection;
    const LabelStudioLabelingConfig config = fetchLabelStudioLabelingConfig(
        state.labelStudioBaseUrl, state.labelStudioProjectId, state.labelStudioApiToken, isDetection);

    if (config.error.empty()) {
        state.labelFromName = config.fromName;
        state.imageToName = config.toName;
        state.labelStudioDataImageKey = config.dataImageKey;
        state.labelStudioAutoFetchStatus = "Auto-filled from Label Studio project settings";
    } else {
        state.labelStudioAutoFetchStatus = "Labeling config: " + config.error;
    }
}

} // namespace

void updateLabelAssistantState(LabelAssistantState& state) {
    if (state.runState == LabelAssistantRunState::Running) {
        state.lastProgress = state.worker.progress();

        LabelAssistantRunResult runResult;
        if (state.worker.tryTakeResult(runResult)) {
            if (runResult.cancelled) {
                state.runState = LabelAssistantRunState::Cancelled;
            } else {
                state.result = std::move(runResult.result);
                state.runState = LabelAssistantRunState::Complete;
            }
        }
    }

    syncLabelAssistantAutoFetch(state);

    syncLabelAssistantSelectedPreview(state);
}

namespace {

struct DraftPrediction {
    std::string filename;
    PredictionResultAndScore resultAndScore;
};

std::vector<DraftPrediction> buildDraftPredictions(const LabelAssistantState& state) {
    std::vector<DraftPrediction> draftPredictions;

    if (state.taskMode == ComparisonTaskMode::Classification) {
        draftPredictions.reserve(state.result.classificationDrafts.size());
        for (const auto& draft : state.result.classificationDrafts) {
            draftPredictions.push_back(DraftPrediction{
                draft.imageFilename,
                buildClassificationPredictionResult(draft, state.labelFromName, state.imageToName)});
        }
    } else {
        draftPredictions.reserve(state.result.detectionDrafts.size());
        for (const auto& draft : state.result.detectionDrafts) {
            if (draft.boxes.empty() && !state.includeZeroDetectionImages) {
                continue;
            }
            draftPredictions.push_back(DraftPrediction{
                draft.imageFilename, buildDetectionPredictionResult(draft, state.labelFromName, state.imageToName)});
        }
    }

    return draftPredictions;
}

} // namespace

void pushLabelAssistantDraftsToLabelStudio(LabelAssistantState& state) {
    const std::vector<DraftPrediction> draftPredictions = buildDraftPredictions(state);

    if (state.sourceMode == LabelAssistantSourceMode::LabelStudioProject) {
        std::vector<LabelStudioKnownTaskPrediction> predictions;
        predictions.reserve(draftPredictions.size());
        int unresolved = 0;
        for (const auto& draft : draftPredictions) {
            const std::optional<int> taskId = parseTaskIdFromFilename(draft.filename);
            if (!taskId) {
                unresolved++;
                continue;
            }
            predictions.push_back(LabelStudioKnownTaskPrediction{*taskId, draft.resultAndScore});
        }

        const LabelStudioAttachSummary summary =
            attachPredictionsToKnownTasks(state.labelStudioBaseUrl, state.labelStudioApiToken, predictions);

        state.exportStatus = "Created " + std::to_string(summary.created) + ", failed "
            + std::to_string(summary.failed) + (unresolved > 0 ? ", unresolved " + std::to_string(unresolved) : "");
        return;
    }

    std::vector<LabelStudioPredictionInput> predictions;
    predictions.reserve(draftPredictions.size());
    for (const auto& draft : draftPredictions) {
        predictions.push_back(LabelStudioPredictionInput{
            draft.filename, state.imageFolderPath + "/" + draft.filename, draft.resultAndScore});
    }

    const LabelStudioPushSummary summary = pushDraftsAsNewLabelStudioTasks(
        state.labelStudioBaseUrl, state.labelStudioProjectId, state.labelStudioApiToken, predictions);

    if (!summary.error.empty()) {
        state.exportStatus = summary.error;
        return;
    }

    state.exportStatus = "Created " + std::to_string(summary.created) + ", upload failed "
        + std::to_string(summary.uploadFailed) + ", task not resolved " + std::to_string(summary.taskNotResolved)
        + ", prediction failed " + std::to_string(summary.predictionFailed);
}

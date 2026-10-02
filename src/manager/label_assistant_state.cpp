#include "manager/label_assistant_state.hpp"

#include "ui_common/gl_texture.hpp"
#include "manager/label_assistant_control_tag.hpp"
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

void startLabelAssistantRun(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
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

        config.labelStudioBaseUrl = session.baseUrl;
        config.labelStudioApiToken = session.apiToken;
        config.unlabeledTasks = selectUnlabeledFromSummaries(sharedData.summaries);
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

std::string buildControlTagKey(const LabelAssistantState& state, const SharedLabelStudioProjectData& sharedData) {
    return sharedData.lastFetchKey + "|" + (state.taskMode == ComparisonTaskMode::Detection ? "D" : "C");
}

void syncLabelAssistantControlTag(LabelAssistantState& state, const SharedLabelStudioProjectData& sharedData) {
    if (sharedData.lastFetchKey.empty()) {
        return;
    }
    const std::string key = buildControlTagKey(state, sharedData);
    if (key == state.lastAutoFetchKey) {
        return;
    }
    state.lastAutoFetchKey = key;

    const LabelAssistantControlTagResolution resolution =
        resolveLabelAssistantControlTag(sharedData.projectConfig.controlTags, state.taskMode);
    state.labelFromName = resolution.fromName;
    state.imageToName = resolution.toName;
    state.labelStudioAutoFetchStatus =
        resolution.error.empty() ? "Auto-filled from Label Studio project settings" : resolution.error;
}

} // namespace

namespace {

std::string describePushResult(const LabelAssistantState& state, const LabelStudioPushRunResult& pushResult) {
    if (pushResult.mode == LabelStudioPushMode::AttachToKnownTasks) {
        const LabelStudioAttachSummary& summary = pushResult.attachSummary;
        std::string status = "Created " + std::to_string(summary.created) + ", failed "
            + std::to_string(summary.failed);
        if (state.lastPushUnresolvedCount > 0) {
            status += ", unresolved " + std::to_string(state.lastPushUnresolvedCount);
        }
        return status;
    }

    const LabelStudioPushSummary& summary = pushResult.uploadSummary;
    if (!summary.error.empty()) {
        return summary.error;
    }
    return "Created " + std::to_string(summary.created) + ", upload failed "
        + std::to_string(summary.uploadFailed) + ", task not resolved " + std::to_string(summary.taskNotResolved)
        + ", prediction failed " + std::to_string(summary.predictionFailed);
}

} // namespace

void updateLabelAssistantState(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
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

    if (state.pushState == LabelAssistantPushState::Running) {
        state.lastPushProgress = state.pushWorker.progress();

        LabelStudioPushRunResult pushResult;
        if (state.pushWorker.tryTakeResult(pushResult)) {
            state.pushState = pushResult.cancelled ? LabelAssistantPushState::Cancelled
                                                    : LabelAssistantPushState::Complete;
            state.exportStatus = describePushResult(state, pushResult);
        }
    }

    syncLabelAssistantControlTag(state, sharedData);

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

void pushLabelAssistantDraftsToLabelStudio(LabelAssistantState& state, const LabelStudioSessionState& session) {
    const std::vector<DraftPrediction> draftPredictions = buildDraftPredictions(state);

    LabelStudioPushConfig config;
    config.baseUrl = session.baseUrl;
    config.apiToken = session.apiToken;

    if (state.sourceMode == LabelAssistantSourceMode::LabelStudioProject) {
        config.mode = LabelStudioPushMode::AttachToKnownTasks;
        config.knownTaskPredictions.reserve(draftPredictions.size());
        state.lastPushUnresolvedCount = 0;
        for (const auto& draft : draftPredictions) {
            const std::optional<int> taskId = parseTaskIdFromFilename(draft.filename);
            if (!taskId) {
                state.lastPushUnresolvedCount++;
                continue;
            }
            config.knownTaskPredictions.push_back(LabelStudioKnownTaskPrediction{*taskId, draft.resultAndScore});
        }
    } else {
        config.mode = LabelStudioPushMode::UploadNewTasks;
        config.projectId = session.activeProjectId;
        config.newTaskPredictions.reserve(draftPredictions.size());
        for (const auto& draft : draftPredictions) {
            config.newTaskPredictions.push_back(LabelStudioPredictionInput{
                draft.filename, state.imageFolderPath + "/" + draft.filename, draft.resultAndScore});
        }
    }

    state.exportStatus.clear();
    state.pushWorker.start(std::move(config));
    state.pushState = LabelAssistantPushState::Running;
}

std::vector<LabelAssistantImageEntry> buildLabelAssistantImageEntries(const LabelAssistantState& state) {
    std::vector<LabelAssistantImageEntry> entries;
    if (state.taskMode == ComparisonTaskMode::Classification) {
        entries.reserve(state.result.classificationDrafts.size());
        for (const auto& draft : state.result.classificationDrafts) {
            entries.push_back(LabelAssistantImageEntry{draft.imageFilename, draft.confidence, draft.predictedLabel});
        }
    } else {
        entries.reserve(state.result.detectionDrafts.size());
        for (const auto& draft : state.result.detectionDrafts) {
            float sum = 0.0f;
            for (const auto& box : draft.boxes) {
                sum += box.confidence;
            }
            const float meanConfidence = draft.boxes.empty() ? 0.0f : sum / static_cast<float>(draft.boxes.size());
            entries.push_back(LabelAssistantImageEntry{
                draft.imageFilename, meanConfidence, std::to_string(draft.boxes.size()) + " box(es)"});
        }
    }
    return entries;
}

std::vector<const LabelAssistantImageEntry*> filterLabelAssistantEntries(
    const std::vector<LabelAssistantImageEntry>& entries, const TextSearch& search, const ConfidenceFilter& confidence,
    ConfidenceSort sort) {
    std::vector<const LabelAssistantImageEntry*> filtered;
    for (const auto& entry : entries) {
        if (search.passes(entry.filename) && confidence.passes(entry.confidence)) {
            filtered.push_back(&entry);
        }
    }
    sortByConfidence(filtered, sort, [](const LabelAssistantImageEntry* entry) {
        return std::optional<float>(entry->confidence);
    });
    return filtered;
}

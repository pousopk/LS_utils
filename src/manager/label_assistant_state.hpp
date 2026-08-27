#pragma once

#include "manager/label_assistant_worker.hpp"
#include "manager/model_evaluation_state.hpp"
#include "manager/onnx_metadata.hpp"

#include <GLFW/glfw3.h>

#include <optional>
#include <string>

enum class LabelAssistantRunState {
    NotStarted,
    Running,
    Complete,
    Cancelled,
};

enum class LabelAssistantSortMode {
    Filename,
    ConfidenceAscending,
    ConfidenceDescending,
};

enum class LabelAssistantConfidenceFilterMode {
    None,
    LessThan,
    GreaterThan,
};

enum class LabelAssistantFilePickerTarget {
    OnnxModel,
    ClassNamesFile,
};

struct LabelAssistantState {
    // This window only offers Detection/Classification in its UI, never
    // Anomaly, even though ComparisonTaskMode (reused from Model
    // Evaluation) has a third value -- see applyLabelAssistantAutoDetect.
    ComparisonTaskMode taskMode = ComparisonTaskMode::Classification;
    ModelSlotConfig modelConfig;   // reused from model_evaluation_state.hpp

    // LocalFolder: imageFolderPath is user-picked (see folderPickerOpen
    // below). LabelStudioProject: startLabelAssistantRun sets
    // imageFolderPath to a hidden scratch folder it downloads unlabeled
    // task images into first -- from that point on, review/preview/export
    // treat it exactly like any local folder.
    LabelAssistantSourceMode sourceMode = LabelAssistantSourceMode::LocalFolder;
    std::string imageFolderPath;

    LabelAssistantWorker worker;
    LabelAssistantRunState runState = LabelAssistantRunState::NotStarted;
    LabelAssistantProgress lastProgress;
    LabelAssistantResult result;

    std::string imageListFilter;
    LabelAssistantSortMode sortMode = LabelAssistantSortMode::Filename;
    LabelAssistantConfidenceFilterMode confidenceFilterMode = LabelAssistantConfidenceFilterMode::None;
    float confidenceFilterThreshold = 0.5f;
    std::optional<std::string> selectedImageFilename;
    std::optional<std::string> renderedPreviewFilename;
    GLuint previewTexture = 0;
    int previewTextureWidth = 0;
    int previewTextureHeight = 0;

    // Export/Label Studio connection. LocalFolder mode: uploads each
    // drafted image and creates a brand-new task with the prediction
    // attached (never reuses an existing task, so re-running duplicates
    // tasks). LabelStudioProject mode: attaches predictions directly to
    // the already-known task ids from the download phase -- no upload,
    // no duplication risk.
    std::string labelFromName;   // choicesFromName (Classification mode) or rectangleLabelsFromName (Detection mode)
    std::string imageToName;
    std::string labelStudioBaseUrl;
    int labelStudioProjectId = 0;
    std::string labelStudioApiToken;
    // The key under a task's `data` holding its image path (e.g.
    // "image"), auto-fetched alongside labelFromName/imageToName --
    // needed in LabelStudioProject mode to know what to download.
    std::string labelStudioDataImageKey;
    // Detection mode only: whether images with zero detected boxes are
    // included in the push/attach (given an empty prediction) or left out
    // entirely. Classification always includes every drafted image, so
    // this has no effect there.
    bool includeZeroDetectionImages = false;
    std::string exportStatus;

    // Tracks which (baseUrl, projectId, apiToken, taskMode) combination
    // labelFromName/imageToName/labelStudioDataImageKey were last
    // auto-fetched for, so updateLabelAssistantState only re-fetches when
    // something relevant actually changes.
    std::string lastAutoFetchKey;
    std::string labelStudioAutoFetchStatus;

    // Folder-picker popup state -- image folder only.
    bool folderPickerOpen = false;
    std::string folderPickerExplorerDir;
    std::string folderPickerFilter;

    // Shared file-picker popup state, reused for onnx model / class names
    // file selection.
    bool filePickerOpen = false;
    LabelAssistantFilePickerTarget filePickerTarget = LabelAssistantFilePickerTarget::OnnxModel;
    std::string filePickerDir;
    std::string filePickerSelectedFile;
    std::string filePickerFilter;
};

// (Re)loads state.modelConfig's model per state.taskMode. Thin wrapper
// around the shared loadModelSlot(ModelSlotConfig&, ComparisonTaskMode).
void loadLabelAssistantModel(LabelAssistantState& state);

// Runs auto-detect on state.modelConfig.onnxPath via the shared
// applyAutoDetectToModelSlot. Since this window's UI only supports
// Detection/Classification, a suggested Anomaly mode is not applied as-is:
// state.taskMode falls back to Classification instead, with a note
// appended to modelConfig.autoDetectStatus explaining why.
void applyLabelAssistantAutoDetect(LabelAssistantState& state);

// Builds a LabelAssistantRunConfig from state.modelConfig/taskMode/
// sourceMode, clears any previous result, and calls state.worker.start(...).
// Caller must have already verified the model matching state.taskMode is
// loaded (and, in LabelStudioProject mode, that the connection fields are
// filled in). In LabelStudioProject mode, also clears/recreates the
// hidden scratch download folder and points state.imageFolderPath at it
// before starting, so every downstream consumer (review list, preview,
// export) treats it exactly like a user-picked local folder. Sets
// state.runState = Running.
void startLabelAssistantRun(LabelAssistantState& state);

// Called once per main-loop iteration while the window is open: polls
// worker.progress()/tryTakeResult(), and on completion stores the result.
// Always also lazily loads/uploads the currently selected image's preview
// texture -- in Detection mode, draws that image's predicted boxes onto
// the texture via annotateDetections first. A no-op if the selection
// hasn't changed since the last call. Also lazily auto-fetches
// labelFromName/imageToName/labelStudioDataImageKey from the Label Studio
// project once labelStudioBaseUrl/labelStudioProjectId/labelStudioApiToken
// are all set, via fetchLabelStudioLabelingConfig -- re-fetches only when
// that combination (plus taskMode) actually changes (see
// lastAutoFetchKey), never blocking manual edits to the resulting fields.
void updateLabelAssistantState(LabelAssistantState& state);

// Builds the mode-appropriate per-draft predictions (via
// buildClassificationPredictionResult/buildDetectionPredictionResult) and
// sends them to Label Studio -- the mechanism depends on state.sourceMode:
// LocalFolder calls pushDraftsAsNewLabelStudioTasks (uploads each image,
// creating a brand-new task per prediction); LabelStudioProject calls
// attachPredictionsToKnownTasks, recovering each draft's already-known
// task id from its filename via parseTaskIdFromFilename (the id this
// window's own download phase encoded into it) -- no upload, no
// duplication risk. In Detection mode, drafts with zero boxes are
// excluded from either path unless state.includeZeroDetectionImages is
// set (Classification always includes every drafted image). Sets
// exportStatus to a summary of what happened.
void pushLabelAssistantDraftsToLabelStudio(LabelAssistantState& state);

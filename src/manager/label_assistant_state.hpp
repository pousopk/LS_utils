#pragma once

#include "manager/item_filters.hpp"
#include "manager/label_assistant_worker.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_push_worker.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/model_evaluation_state.hpp"
#include "manager/onnx_metadata.hpp"

#include <GLFW/glfw3.h>

#include <optional>
#include <string>
#include <vector>

enum class LabelAssistantRunState {
    NotStarted,
    Running,
    Complete,
    Cancelled,
};

// Separate from LabelAssistantRunState: pushing drafts to Label Studio is
// an independent operation from the model-inference run above (its own
// worker, its own progress), even though both reuse the
// NotStarted/Running/Complete/Cancelled shape.
enum class LabelAssistantPushState {
    NotStarted,
    Running,
    Complete,
    Cancelled,
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

    TextSearch imageSearch;
    ConfidenceSort sortMode = ConfidenceSort::None;   // None = filename order
    ConfidenceFilter confidenceFilter;
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
    // Detection mode only: whether images with zero detected boxes are
    // included in the push/attach (given an empty prediction) or left out
    // entirely. Classification always includes every drafted image, so
    // this has no effect there.
    bool includeZeroDetectionImages = false;
    LabelStudioPushWorker pushWorker;
    LabelAssistantPushState pushState = LabelAssistantPushState::NotStarted;
    LabelStudioPushProgress lastPushProgress;
    // Drafts whose filename didn't decode to a task id (LabelStudioProject
    // mode only, via parseTaskIdFromFilename) -- computed synchronously
    // before starting pushWorker, since it doesn't need a network call;
    // folded into exportStatus once the push completes.
    int lastPushUnresolvedCount = 0;
    std::string exportStatus;

    // Tracks which (sharedData.lastFetchKey, taskMode) combination
    // labelFromName/imageToName were last resolved for (see
    // resolveLabelAssistantControlTag), so updateLabelAssistantState only
    // overwrites them when something relevant actually changes -- a user
    // who manually edits labelFromName/imageToName in the UI must not
    // have that edit clobbered every frame.
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

// One row per drafted image, mode-agnostic so the list/sort/filter code
// below doesn't need to branch on taskMode. Detection's confidence is the
// mean across that image's boxes, matching batchEvalImageConfidence's
// existing convention (model_evaluation_state.hpp) for the same class of
// "one representative confidence per image" need.
struct LabelAssistantImageEntry {
    std::string filename;
    float confidence = 0.0f;
    std::string labelSummary;
};

// One entry per drafted image in state.result, for the current taskMode.
std::vector<LabelAssistantImageEntry> buildLabelAssistantImageEntries(const LabelAssistantState& state);

// Pure function: the entries passing `search` (filename) and
// `confidence`, then ordered by `sort` (stable; None keeps `entries`
// order). Pointers point into `entries`.
std::vector<const LabelAssistantImageEntry*> filterLabelAssistantEntries(
    const std::vector<LabelAssistantImageEntry>& entries, const TextSearch& search, const ConfidenceFilter& confidence,
    ConfidenceSort sort);

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
// sourceMode plus session's connection info, clears any previous result,
// and calls state.worker.start(...). Caller must have already verified
// the model matching state.taskMode is loaded (and, in LabelStudioProject
// mode, that session is connected with an active project). In
// LabelStudioProject mode, also clears/recreates the hidden scratch
// download folder and points state.imageFolderPath at it before
// starting, so every downstream consumer (review list, preview, export)
// treats it exactly like a user-picked local folder. Sets
// state.runState = Running.
void startLabelAssistantRun(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

// Called once per main-loop iteration while the window is open: polls
// worker.progress()/tryTakeResult(), and on completion stores the result.
// Always also lazily loads/uploads the currently selected image's preview
// texture -- in Detection mode, draws that image's predicted boxes onto
// the texture via annotateDetections first. A no-op if the selection
// hasn't changed since the last call. Also resolves labelFromName/
// imageToName from sharedData.projectConfig.controlTags via
// resolveLabelAssistantControlTag -- a plain local lookup, no network
// call -- re-resolving only when (sharedData.lastFetchKey, taskMode)
// actually changes (see lastAutoFetchKey), never blocking manual edits to
// the resulting fields.
void updateLabelAssistantState(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

// Builds the mode-appropriate per-draft predictions (via
// buildClassificationPredictionResult/buildDetectionPredictionResult),
// then starts state.pushWorker to send them to Label Studio via
// session's connection info and returns immediately (does not block on
// the network calls) -- the mechanism depends on state.sourceMode:
// LocalFolder uses LabelStudioPushMode::UploadNewTasks
// (pushDraftsAsNewLabelStudioTasks: uploads each image, creating a
// brand-new task per prediction); LabelStudioProject uses
// LabelStudioPushMode::AttachToKnownTasks (attachPredictionsToKnownTasks),
// recovering each draft's already-known task id from its filename via
// parseTaskIdFromFilename (the id this window's own download phase
// encoded into it; drafts that don't resolve are counted into
// state.lastPushUnresolvedCount and left out of the push) -- no upload,
// no duplication risk. In Detection mode, drafts with zero boxes are
// excluded from either path unless state.includeZeroDetectionImages is
// set (Classification always includes every drafted image). Sets
// state.pushState = Running; updateLabelAssistantState polls the worker
// and sets state.exportStatus to a summary once it completes.
void pushLabelAssistantDraftsToLabelStudio(LabelAssistantState& state, const LabelStudioSessionState& session);

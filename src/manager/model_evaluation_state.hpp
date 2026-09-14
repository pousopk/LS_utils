#pragma once

#include "manager/anomaly_inference.hpp"
#include "manager/anomaly_worker.hpp"
#include "manager/app_runtime.hpp"
#include "manager/batch_evaluation_worker.hpp"
#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/classification_worker.hpp"
#include "manager/comparison_task_mode.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/inference_worker.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/model_metadata_detection.hpp"
#include "manager/yolo_inference.hpp"
#include "objects/frame_source.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ModelEvaluationState;

// Shared model configuration for one "slot" (Model A or Model B), used
// regardless of whether the evaluation source is Live or Batch. Loading a
// model here makes it available to both sources without reloading.
struct ModelSlotConfig {
    std::string onnxPath;
    std::string classNamesPath;
    std::vector<std::string> autoDetectedClassNames;
    OnnxPreprocessingHints hints;
    std::string autoDetectStatus;
    int inputWidth = 0;
    int inputHeight = 0;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    // Manual toggle -- auto-detecting OBB vs. axis-aligned from the ONNX
    // file alone is ambiguous (a 1-class OBB model and a 2-class
    // axis-aligned model produce the same output channel count).
    bool isObbDetectionModel = false;

    std::shared_ptr<YoloModel> detectionModel;
    std::shared_ptr<ClassificationModel> classificationModel;
    std::shared_ptr<AnomalyModel> anomalyModel;
    float anomalyScoreMin = 0.0f;
    float anomalyScoreMax = 1.0f;
    float anomalyThreshold = 0.5f;
    std::string loadError;
    std::string engineStatus;  // "Engine: GPU" / "Engine: CPU", set on successful load
};

// (Re)loads slot's model from its onnxPath/classNamesPath as either a YOLO
// detector or a classifier depending on mode. Clears any previous
// model/error first. No-op if onnxPath is empty.
void loadModelSlot(ModelSlotConfig& slot, ComparisonTaskMode mode);

// Runs autoDetectModel() on slot.onnxPath and applies the result: on
// success, pre-fills inputWidth/inputHeight, autoDetectedClassNames, hints,
// a one-line autoDetectStatus summary, and switches taskMode if a mode was
// confidently suggested. On failure, leaves existing field values
// untouched and sets autoDetectStatus to the error -- never blocks manual
// entry. Takes `taskMode` directly (not the whole ModelEvaluationState) so
// other windows with their own task-mode field can reuse this function.
void applyAutoDetectToModelSlot(ModelSlotConfig& slot, ComparisonTaskMode& taskMode);

// Draws boxes + labels for `detections` onto a clone of `frame`. Box/label
// size scales up with the frame's own resolution relative to an 800px-wide
// baseline (never down), and every label is drawn on a solid background so
// it stays readable regardless of the photo's own colors.
cv::Mat annotateDetections(const cv::Mat& frame, const std::vector<Detection>& detections);

enum class EvaluationSourceMode {
    Live,
    Batch,
};

enum class LiveSourceMode {
    None,
    ExistingSession,
    LoadedFile,
};

// Per-slot live-streaming runtime: the continuously-running inference
// worker, its output texture, and its latest results. Kept separate from
// ModelSlotConfig because it is Live-source-specific runtime state, not
// model configuration. workerModel/workerClassificationModel track which
// model the current worker was built from, so updateLiveRuntime can detect
// a (re)load -- including one that happened while Batch was the active
// source -- and rebuild the worker lazily.
struct LiveRuntimeSlot {
    GLuint texture = 0;
    int textureWidth = 0;
    int textureHeight = 0;

    std::unique_ptr<InferenceWorker> worker;
    std::shared_ptr<YoloModel> workerModel;
    std::vector<Detection> latestDetections;
    double latestInferenceMs = 0.0;

    std::unique_ptr<ClassificationInferenceWorker> classificationWorker;
    std::shared_ptr<ClassificationModel> workerClassificationModel;
    std::vector<ClassPrediction> latestPredictions;

    std::unique_ptr<AnomalyInferenceWorker> anomalyWorker;
    std::shared_ptr<AnomalyModel> workerAnomalyModel;
    AnomalyResult latestAnomalyResult;

    std::string runtimeError;
};

struct LiveRuntime {
    LiveSourceMode sourceMode = LiveSourceMode::None;
    std::string sessionId;
    std::unique_ptr<FrameSource> loadedSource;
    std::string loadedSourcePath;

    std::array<LiveRuntimeSlot, 2> liveSlots;

    float agreementIoUThreshold = 0.5f;
    BoxAgreement latestAgreement;
};

// Pulls the current frame from whichever live source is selected, submits
// it to each active (0, or 0 and 1 if compareTwoModels) slot's idle
// worker -- rebuilding that slot's worker first if its model was
// (re)loaded since the last call -- drains finished results into that
// slot's texture/stats, and recomputes the agreement stat (Detection mode,
// compareTwoModels only).
void updateLiveRuntime(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, LiveRuntime& live,
    std::vector<CameraSession>& sessions);

enum class BatchEvalRunState {
    NotStarted,
    Running,
    Complete,
    Cancelled,
};

enum class BatchEvalImageSortMode {
    Filename,
    ConfidenceAscending,
    ConfidenceDescending,
};

enum class BatchEvalConfidenceFilterMode {
    None,
    LessThan,
    GreaterThan,
};

// Detection mode only -- whether an image produced at least one detected
// box. Meaningless for classification (a run always produces a top-1
// prediction unless inference itself failed), so it's ignored there.
enum class BatchEvalDetectionPresenceFilter {
    Any,
    HasDetections,
    NoDetections,
};

struct BatchPreviewTexture {
    GLuint previewTexture = 0;
    int previewTextureWidth = 0;
    int previewTextureHeight = 0;
};

struct BatchRuntime {
    // LocalFolder mode: imageFolderPath is user-picked, groundTruth comes
    // from a manually-exported Label Studio JSON file (groundTruthJsonPath).
    // LabelStudioProject mode: startBatchEvaluationRun points
    // imageFolderPath at a hidden scratch folder instead, and the worker
    // fills in groundTruth/hasGroundTruth itself from the download phase
    // -- from that point on, results/metrics/preview treat it exactly
    // like the LocalFolder case.
    BatchEvalSourceMode sourceMode = BatchEvalSourceMode::LocalFolder;
    std::string imageFolderPath;
    std::string groundTruthJsonPath;
    LabelStudioImportResult groundTruth;
    std::string groundTruthStatus;
    bool hasGroundTruth = false;

    // LabelStudioProject mode connection details. labelStudioDataImageKey
    // is auto-fetched (see syncBatchEvalLabelStudioAutoFetch) -- the key
    // under a task's `data` holding its image path, needed to know what
    // to download; unlike Label Assistant's push, no from_name/to_name
    // are needed here since this only ever downloads, never pushes.
    std::string labelStudioDataImageKey;
    std::string labelStudioAutoFetchStatus;
    std::string lastAutoFetchKey;

    BatchEvaluationWorker worker;
    BatchEvalRunState runState = BatchEvalRunState::NotStarted;
    BatchEvalProgress lastProgress;

    BatchEvaluationResult resultA;
    BatchEvaluationResult resultB;
    DetectionMetrics detectionMetricsA, detectionMetricsB;
    ClassificationMetrics classificationMetricsA, classificationMetricsB;

    bool mismatchesOnly = false;
    std::string imageListFilter;
    BatchEvalImageSortMode imageSortMode = BatchEvalImageSortMode::Filename;
    BatchEvalConfidenceFilterMode confidenceFilterMode = BatchEvalConfidenceFilterMode::None;
    float confidenceFilterThreshold = 0.5f;
    BatchEvalDetectionPresenceFilter detectionPresenceFilter = BatchEvalDetectionPresenceFilter::Any;
    std::optional<std::string> selectedImageFilename;
    std::optional<std::string> renderedPreviewFilename;

    // When enabled, only a random sampleSize-sized subset of the folder's
    // images is evaluated -- useful for a quick look at a huge dataset
    // with a slow CPU model instead of waiting for the whole thing.
    bool sampleEnabled = false;
    int sampleSize = 100;

    // Per-slot preview texture for the currently selected image (Batch's
    // per-image detail pane); Live's per-slot streaming texture lives on
    // LiveRuntimeSlot instead.
    std::array<BatchPreviewTexture, 2> previewTextures;

    bool folderPickerOpen = false;
    std::string folderPickerExplorerDir;
    std::string folderPickerFilter;
};

// Loads and parses batch.groundTruthJsonPath into batch.groundTruth,
// setting hasGroundTruth/groundTruthStatus. Clears ground truth (sets
// hasGroundTruth = false) if the path is empty or parsing fails.
// LocalFolder mode only.
void loadBatchEvalGroundTruth(BatchRuntime& batch);

// Runs auto-detect on session's (baseUrl, activeProjectId, apiToken) via
// the shared fetchLabelStudioLabelingConfig, storing only dataImageKey
// (the from_name/to_name it also returns are unused here). Re-fetches
// only when that connection combination actually changes (see
// batch.lastAutoFetchKey). LabelStudioProject mode only.
void syncBatchEvalLabelStudioAutoFetch(BatchRuntime& batch, const LabelStudioSessionState& session);

// Builds a BatchEvalRunConfig from slots + batch config (LocalFolder:
// imageFolderPath + any loaded groundTruth as-is; LabelStudioProject:
// clears/recreates a hidden scratch folder, points batch.imageFolderPath
// at it, and lets the worker fill in groundTruth itself during its
// download phase), clears any previous results, and calls
// batch.worker.start(...). Caller must have already verified the
// required slot(s) are loaded (and, in LabelStudioProject mode, that
// session is connected with an active project). Sets batch.runState =
// Running.
void startBatchEvaluationRun(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, BatchRuntime& batch,
    const LabelStudioSessionState& session);

// Called once per main-loop iteration while the window is open: while a
// run is in progress, polls worker.progress()/tryTakeResult() and, on
// completion, computes detection/classification metrics for both slots.
// Always also lazily loads/annotates/uploads the currently selected
// image's preview textures (a no-op if the selection hasn't changed), and
// lazily runs syncBatchEvalLabelStudioAutoFetch (a no-op unless
// LabelStudioProject mode's session is connected with an active project
// and it changed).
void updateBatchRuntime(
    ComparisonTaskMode mode, const std::array<ModelSlotConfig, 2>& slots, const std::string& imageFolderPath,
    BatchRuntime& batch, const LabelStudioSessionState& session);

// True if the given filename is a "mismatch" for either slot: predicted
// top-1 != true label (classification), or the image has an unmatched
// prediction/ground-truth box at IoU >= 0.5 (detection). False if
// `filename` has no ground truth in either slot's result.
bool isBatchEvalImageMismatch(ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename);

// Representative confidence for `filename` in one slot's result: the
// classification top-1 probability, or the mean confidence across all
// detected boxes. std::nullopt if that slot has no result or no
// prediction/detection for this image.
std::optional<float> batchEvalImageConfidence(
    ComparisonTaskMode mode, const BatchEvaluationResult& result, const std::string& filename);

// True if `filename` passes the current confidence filter for either slot
// (OR, matching isBatchEvalImageMismatch's convention). Always true when
// batch.confidenceFilterMode == None.
bool batchEvalImagePassesConfidenceFilter(
    ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename);

// Sort key for BatchEvalImageSortMode::Confidence{Ascending,Descending}: the
// lower of the two slots' confidences for `filename` (a slot with no
// prediction contributes 0.0f, so images missing a prediction sort first
// ascending / last descending).
float batchEvalImageSortConfidence(ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename);

// True if `filename` passes the current detection-presence filter for
// either slot (OR, matching isBatchEvalImageMismatch's convention):
// HasDetections passes if slot A or slot B found at least one box,
// NoDetections passes if slot A or slot B found none. Always true when
// mode != Detection or batch.detectionPresenceFilter == Any.
bool batchEvalImagePassesDetectionPresenceFilter(
    ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename);

enum class FilePickerTarget {
    SlotAModel,
    SlotAClassNames,
    SlotBModel,
    SlotBClassNames,
    GroundTruthJson,
    LiveSourceFile,
};

struct ModelEvaluationState {
    ComparisonTaskMode taskMode = ComparisonTaskMode::Detection;
    bool compareTwoModels = false;  // default: single model
    EvaluationSourceMode source = EvaluationSourceMode::Live;

    std::array<ModelSlotConfig, 2> slots;

    LiveRuntime live;
    BatchRuntime batch;

    // Shared file-picker popup state, reused for onnx/class-names/ground-
    // truth/live-source-file selection across both slots and both sources.
    bool filePickerOpen = false;
    FilePickerTarget filePickerTarget = FilePickerTarget::SlotAModel;
    std::string filePickerDir;
    std::string filePickerSelectedFile;
    std::string filePickerFilter;
};

// Resets both runtimes' results (live workers/textures/detections/
// predictions, batch resultA/B/metrics/run state) without touching slot
// config. Called on taskMode/compareTwoModels changes so neither source
// ever shows stale content for a mode/count that no longer matches. Live
// workers are torn down here but rebuild automatically on the next
// updateLiveRuntime call (workerModel becomes null, which no longer
// matches the still-loaded ModelSlotConfig, triggering a rebuild).
void resetModelEvaluationResults(ModelEvaluationState& state);

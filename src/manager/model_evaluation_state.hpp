#pragma once

#include "manager/app_runtime.hpp"
#include "manager/batch_evaluation_worker.hpp"
#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/classification_worker.hpp"
#include "manager/comparison_task_mode.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/inference_worker.hpp"
#include "manager/label_studio_import.hpp"
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

    std::shared_ptr<YoloModel> detectionModel;
    std::shared_ptr<ClassificationModel> classificationModel;
    std::string loadError;
    std::string engineStatus;  // "Engine: GPU" / "Engine: CPU", set on successful load
};

// (Re)loads slot's model from its onnxPath/classNamesPath as either a YOLO
// detector or a classifier depending on mode. Clears any previous
// model/error first. No-op if onnxPath is empty.
void loadModelSlot(ModelSlotConfig& slot, ComparisonTaskMode mode);

// Runs autoDetectModel() on slot.onnxPath and applies the result: on
// success, pre-fills inputWidth/inputHeight, autoDetectedClassNames, hints,
// a one-line autoDetectStatus summary, and switches state.taskMode if a
// mode was confidently suggested. On failure, leaves existing field values
// untouched and sets autoDetectStatus to the error -- never blocks manual
// entry.
void applyAutoDetectToModelSlot(ModelSlotConfig& slot, ModelEvaluationState& state);

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

struct BatchPreviewTexture {
    GLuint previewTexture = 0;
    int previewTextureWidth = 0;
    int previewTextureHeight = 0;
};

struct BatchRuntime {
    std::string imageFolderPath;
    std::string groundTruthJsonPath;
    LabelStudioImportResult groundTruth;
    std::string groundTruthStatus;
    bool hasGroundTruth = false;

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
    std::optional<std::string> selectedImageFilename;
    std::optional<std::string> renderedPreviewFilename;

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
void loadBatchEvalGroundTruth(BatchRuntime& batch);

// Builds a BatchEvalRunConfig from slots + batch config, clears any
// previous results, and calls batch.worker.start(...). Caller must have
// already verified the required slot(s) are loaded. Sets
// batch.runState = Running.
void startBatchEvaluationRun(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, BatchRuntime& batch);

// Called once per main-loop iteration while the window is open: while a
// run is in progress, polls worker.progress()/tryTakeResult() and, on
// completion, computes detection/classification metrics for both slots.
// Always also lazily loads/annotates/uploads the currently selected
// image's preview textures (a no-op if the selection hasn't changed).
void updateBatchRuntime(ComparisonTaskMode mode, const std::string& imageFolderPath, BatchRuntime& batch);

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

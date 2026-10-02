#pragma once

#include "manager/anomaly_inference.hpp"
#include "ui_common/gl_texture.hpp"
#include "ui_common/path_picker.hpp"
#include "manager/benchmark_filters.hpp"
#include "manager/benchmark_worker.hpp"
#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/model_task.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/model_slot.hpp"
#include "manager/yolo_inference.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class BenchmarkRunState {
    NotStarted,
    Running,
    Complete,
    Cancelled,
};

struct BenchmarkPreviewTexture {
    GLuint previewTexture = 0;
    int previewTextureWidth = 0;
    int previewTextureHeight = 0;
};

enum class BenchmarkPickerTarget {
    ImageFolder,
    SlotAModel,
    SlotBModel,
    GroundTruthJson,
};

struct BenchmarkState {
    // Follows Model A's task (see loadBenchmarkSlot) -- not user-picked.
    ModelTask taskMode = ModelTask::Detection;
    bool compareTwoModels = false;  // default: single model

    std::array<ModelSlotConfig, 2> slots;

    // LocalFolder mode: imageFolderPath is user-picked, groundTruth comes
    // from a manually-exported Label Studio JSON file (groundTruthJsonPath).
    // LabelStudioProject mode: tasks and ground truth come from the shared
    // task list, and startBenchmarkRun points imageFolderPath at the
    // project's session image cache -- from that point on,
    // results/metrics/preview treat it exactly like the LocalFolder case.
    BenchmarkSourceMode sourceMode = BenchmarkSourceMode::LocalFolder;
    std::string imageFolderPath;
    std::string groundTruthJsonPath;
    LabelStudioImportResult groundTruth;
    std::string groundTruthStatus;
    bool hasGroundTruth = false;

    // The LabelStudioProject image cache is wiped once per app session,
    // on the first Label Studio run (see benchmarkImageCacheRoot).
    bool imageCacheCleared = false;

    BenchmarkWorker worker;
    BenchmarkRunState runState = BenchmarkRunState::NotStarted;
    BenchmarkProgress lastProgress;

    BenchmarkResult resultA;
    BenchmarkResult resultB;
    DetectionMetrics detectionMetricsA, detectionMetricsB;
    ClassificationMetrics classificationMetricsA, classificationMetricsB;

    TextSearch imageSearch;                          // filename search
    ConfidenceSort imageSort = ConfidenceSort::None; // None = filename order
    BenchmarkImageFilters filters;
    std::optional<std::string> selectedImageFilename;
    std::optional<std::string> renderedPreviewFilename;

    // When enabled, only a random sampleSize-sized subset of the folder's
    // images is evaluated -- useful for a quick look at a huge dataset
    // with a slow CPU model instead of waiting for the whole thing.
    bool sampleEnabled = false;
    int sampleSize = 100;

    // Per-slot preview texture for the currently selected image (the
    // per-image detail pane).
    std::array<BenchmarkPreviewTexture, 2> previewTextures;

    // Shared picker popup, reused for the image folder, both slots' models
    // and the ground-truth file; filePickerTarget says which.
    PathPickerState picker;
    BenchmarkPickerTarget filePickerTarget = BenchmarkPickerTarget::SlotAModel;
};

// Loads and parses state.groundTruthJsonPath into state.groundTruth,
// setting hasGroundTruth/groundTruthStatus. Clears ground truth (sets
// hasGroundTruth = false) if the path is empty or parsing fails.
// LocalFolder mode only.
void loadBenchmarkGroundTruth(BenchmarkState& state);

struct BenchmarkTaskSelection {
    std::vector<LabelStudioUnlabeledTask> tasks;  // sorted by task id
    // Ground truth for `tasks`, each image's filename set to
    // taskImageLocalFilename -- the name it's cached under.
    LabelStudioImportResult groundTruth;
    int labeledTaskCount = 0;  // labeled tasks before sampling
};

// Pure function: the labeled tasks in `data` (an annotation plus parsed
// ground truth), randomly sampled down to `sampleSize` with `seed`
// (0 = keep all). Sampling happens here, before anything is downloaded.
BenchmarkTaskSelection selectBenchmarkTasks(const SharedLabelStudioProjectData& data, int sampleSize, unsigned seed);

// Where LabelStudioProject runs cache downloaded images:
// <temp>/vision_app_benchmark_cache, one subfolder per project id.
std::filesystem::path benchmarkImageCacheRoot();

// Builds a BenchmarkRunConfig from state's slots + source config
// (LocalFolder: imageFolderPath + any loaded groundTruth as-is;
// LabelStudioProject: selectBenchmarkTasks over sharedData, with
// state.imageFolderPath pointed at the project's image cache -- wiping
// the cache root first if this is the session's first Label Studio run),
// clears any previous results, and calls state.worker.start(...). Caller
// must have already verified the required slot(s) are loaded (and, in
// LabelStudioProject mode, that sharedData has finished loading). Sets
// state.runState = Running.
void startBenchmarkRun(
    BenchmarkState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

// Called once per main-loop iteration while the window is open: while a
// run is in progress, polls worker.progress()/tryTakeResult() and, on
// completion, computes detection/classification metrics for both slots.
// Always also lazily loads/annotates/uploads the currently selected
// image's preview textures (a no-op if the selection hasn't changed).
void updateBenchmarkState(BenchmarkState& state);

// Resets results (resultA/B, metrics, run state, selection, confusion-cell
// filter) and drops filter values that no longer apply, without touching
// slot or source config. Called before every run and whenever the task,
// model count, source or image folder changes, so the tab never shows
// results that no longer match.
void resetBenchmarkResults(BenchmarkState& state);

// Loads state.slots[slotIndex] via loadModelSlot. Loading Model A sets
// state.taskMode to its task (resetting results if that changes it).
void loadBenchmarkSlot(BenchmarkState& state, int slotIndex);

// True when both models are loaded but Model B's task differs from Model
// A's -- a two-model run is blocked until they match.
bool modelBTaskMismatch(const BenchmarkState& state);

// True when the slot(s) needed for a run are loaded with matching tasks.
bool benchmarkSlotsReady(const BenchmarkState& state);

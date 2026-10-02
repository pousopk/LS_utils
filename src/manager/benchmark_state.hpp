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
#include "manager/label_studio_session.hpp"
#include "manager/model_slot.hpp"
#include "manager/yolo_inference.hpp"

#include <GLFW/glfw3.h>

#include <array>
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
    // LabelStudioProject mode: startBenchmarkRun points
    // imageFolderPath at a hidden scratch folder instead, and the worker
    // fills in groundTruth/hasGroundTruth itself from the download phase
    // -- from that point on, results/metrics/preview treat it exactly
    // like the LocalFolder case.
    BenchmarkSourceMode sourceMode = BenchmarkSourceMode::LocalFolder;
    std::string imageFolderPath;
    std::string groundTruthJsonPath;
    LabelStudioImportResult groundTruth;
    std::string groundTruthStatus;
    bool hasGroundTruth = false;

    // LabelStudioProject mode connection details. labelStudioDataImageKey
    // is auto-fetched (see syncBenchmarkLabelStudioAutoFetch) -- the key
    // under a task's `data` holding its image path, needed to know what
    // to download; unlike Label Assistant's push, no from_name/to_name
    // are needed here since this only ever downloads, never pushes.
    std::string labelStudioDataImageKey;
    std::string labelStudioAutoFetchStatus;
    std::string lastAutoFetchKey;

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

// Runs auto-detect on session's (baseUrl, activeProjectId, apiToken) via
// the shared fetchLabelStudioLabelingConfig, storing only dataImageKey
// (the from_name/to_name it also returns are unused here). Re-fetches
// only when that connection combination actually changes (see
// state.lastAutoFetchKey). LabelStudioProject mode only.
void syncBenchmarkLabelStudioAutoFetch(BenchmarkState& state, const LabelStudioSessionState& session);

// Builds a BenchmarkRunConfig from state's slots + source config (LocalFolder:
// imageFolderPath + any loaded groundTruth as-is; LabelStudioProject:
// clears/recreates a hidden scratch folder, points state.imageFolderPath
// at it, and lets the worker fill in groundTruth itself during its
// download phase), clears any previous results, and calls
// state.worker.start(...). Caller must have already verified the
// required slot(s) are loaded (and, in LabelStudioProject mode, that
// session is connected with an active project). Sets state.runState =
// Running.
void startBenchmarkRun(BenchmarkState& state, const LabelStudioSessionState& session);

// Called once per main-loop iteration while the window is open: while a
// run is in progress, polls worker.progress()/tryTakeResult() and, on
// completion, computes detection/classification metrics for both slots.
// Always also lazily loads/annotates/uploads the currently selected
// image's preview textures (a no-op if the selection hasn't changed), and
// lazily runs syncBenchmarkLabelStudioAutoFetch (a no-op unless
// LabelStudioProject mode's session is connected with an active project
// and it changed).
void updateBenchmarkState(BenchmarkState& state, const LabelStudioSessionState& session);

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

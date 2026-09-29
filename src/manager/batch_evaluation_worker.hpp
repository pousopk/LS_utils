#pragma once

#include "manager/anomaly_inference.hpp"
#include "manager/batch_evaluation.hpp"
#include "manager/classification_inference.hpp"
#include "manager/comparison_task_mode.hpp"
#include "manager/yolo_inference.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Where a batch run's images (and, in LabelStudioProject mode, ground
// truth) come from. LabelStudioProject adds a download+parse phase ahead
// of the same per-slot evaluation phase LocalFolder already used.
enum class BatchEvalSourceMode {
    LocalFolder,
    LabelStudioProject,
};

struct BatchEvalRunConfig {
    ComparisonTaskMode mode = ComparisonTaskMode::Detection;

    BatchEvalSourceMode source = BatchEvalSourceMode::LocalFolder;

    // LocalFolder: the folder to scan directly, and (if hasGroundTruth) a
    // manually-exported ground truth already parsed by the caller.
    std::string imageFolderPath;
    LabelStudioImportResult groundTruth;  // ignored unless hasGroundTruth
    bool hasGroundTruth = false;

    // LabelStudioProject: connection details for the download+ground-truth
    // phase, plus the (already-created, ideally-cleared) scratch folder
    // downloaded images are written into -- this becomes imageFolderPath,
    // and the fetched ground truth becomes `groundTruth`/`hasGroundTruth`,
    // for the rest of this run exactly as if the caller had supplied them.
    std::string labelStudioBaseUrl;
    int labelStudioProjectId = 0;
    std::string labelStudioApiToken;
    std::string labelStudioDataImageKey;
    std::string scratchFolderPath;
    bool runSlotB = true;  // false = single-model run, slot B's models/thresholds are ignored
    int sampleSize = 0;    // 0 = evaluate every image; otherwise a random subset of this size,
                            // seeded once per run so both slots see the exact same subset
    std::shared_ptr<YoloModel> detectionModelA;
    std::shared_ptr<YoloModel> detectionModelB;
    std::shared_ptr<ClassificationModel> classificationModelA;
    std::shared_ptr<ClassificationModel> classificationModelB;
    std::shared_ptr<AnomalyModel> anomalyModelA;
    std::shared_ptr<AnomalyModel> anomalyModelB;
    // Per-slot, matching ModelSlotConfig's existing precedent (two
    // models may legitimately want different confidence/NMS thresholds).
    float confThresholdA = 0.25f;
    float nmsThresholdA = 0.45f;
    float confThresholdB = 0.25f;
    float nmsThresholdB = 0.45f;
    // Anomaly mode only -- see AnomalyModel::infer (threshold is a
    // call-time parameter, not baked into the model).
    float anomalyThresholdA = 0.5f;
    float anomalyThresholdB = 0.5f;
};

struct BatchEvalProgress {
    int currentSlot = 0;  // 0 = not started, 1 = running Model A, 2 = running Model B
    int completed = 0;
    int total = 0;
    std::string phaseLabel;   // "Downloading" during LabelStudioProject's pre-phase; empty otherwise
};

struct BatchEvalRunResult {
    BatchEvaluationResult slotA;
    BatchEvaluationResult slotB;
    bool cancelled = false;
};

// Runs Model A then Model B sequentially over the same image folder on a
// single background thread, reporting live progress and honoring
// cancellation. Not copyable. Reuse one instance across runs -- start()
// joins any previous thread first. In LabelStudioProject source mode,
// run() does an extra phase first -- download the project's labeled
// tasks and their ground truth (via fetchAndDownloadLabeledDataset) --
// before the same per-slot evaluation phase LocalFolder mode uses,
// distinguished in BatchEvalProgress via `phaseLabel`.
class BatchEvaluationWorker {
public:
    BatchEvaluationWorker() = default;
    ~BatchEvaluationWorker();

    BatchEvaluationWorker(const BatchEvaluationWorker&) = delete;
    BatchEvaluationWorker& operator=(const BatchEvaluationWorker&) = delete;

    // Joins any prior run, then starts a new background run with `config`.
    void start(BatchEvalRunConfig config);

    bool isRunning() const { return running_.load(); }

    // Sets the cancel flag the running thread checks between images and
    // between slots. No-op if not running.
    void requestCancel();

    // Lock-free snapshot of current progress -- safe to call every frame.
    BatchEvalProgress progress() const;

    // Non-blocking poll:
    // returns true and moves the result out exactly once, on the frame
    // after the run finishes (normally or via cancel).
    bool tryTakeResult(BatchEvalRunResult& out);

private:
    void run(BatchEvalRunConfig config);
    void finish(BatchEvalRunResult result);

    std::thread thread_;
    std::atomic<int> currentSlot_{0};
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex phaseMutex_;
    std::string phaseLabel_;

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    BatchEvalRunResult latestResult_;
};

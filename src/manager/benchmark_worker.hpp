#pragma once

#include "manager/anomaly_inference.hpp"
#include "manager/benchmark_evaluation.hpp"
#include "manager/classification_inference.hpp"
#include "manager/label_studio_client.hpp"
#include "manager/model_task.hpp"
#include "manager/yolo_inference.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Where a run's images and ground truth come from. LabelStudioProject adds
// a download phase (only images not already cached) ahead of the same
// per-slot evaluation phase LocalFolder uses.
enum class BenchmarkSourceMode {
    LocalFolder,
    LabelStudioProject,
};

struct BenchmarkRunConfig {
    ModelTask mode = ModelTask::Detection;

    BenchmarkSourceMode source = BenchmarkSourceMode::LocalFolder;

    // Ground truth for the images evaluated (ignored unless hasGroundTruth).
    // LocalFolder: a manually-exported file, matched by basename.
    // LabelStudioProject: built by the caller from the shared task list,
    // with each image's filename set to taskImageLocalFilename.
    LabelStudioImportResult groundTruth;
    bool hasGroundTruth = false;

    // LocalFolder: the folder to scan; the worker samples it down to
    // sampleSize (0 = every image), once per run so both slots see the
    // exact same subset.
    std::string imageFolderPath;
    int sampleSize = 0;

    // LabelStudioProject: the (already sampled) tasks to evaluate. Each
    // image lives in cacheFolderPath as taskImageLocalFilename(...); only
    // the ones not already there are downloaded. labeledTaskCount is how
    // many labeled tasks there were before sampling.
    std::string labelStudioBaseUrl;
    std::string labelStudioApiToken;
    std::string cacheFolderPath;
    std::vector<LabelStudioUnlabeledTask> tasks;
    int labeledTaskCount = 0;

    bool runSlotB = true;  // false = single-model run, slot B's models/thresholds are ignored
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

struct BenchmarkProgress {
    int currentSlot = 0;  // 0 = not started, 1 = running Model A, 2 = running Model B
    int completed = 0;
    int total = 0;
    std::string phaseLabel;   // e.g. "Downloading (63 cached)" during LabelStudioProject's pre-phase; empty otherwise
};

struct BenchmarkRunResult {
    BenchmarkResult slotA;
    BenchmarkResult slotB;
    bool cancelled = false;
};

// Runs Model A then Model B sequentially over the same images on a single
// background thread, reporting live progress and honoring cancellation.
// Not copyable. Reuse one instance across runs -- start() joins any
// previous thread first. In LabelStudioProject source mode, run() first
// downloads whichever of config.tasks' images aren't cached yet,
// distinguished in BenchmarkProgress via `phaseLabel`.
class BenchmarkWorker {
public:
    BenchmarkWorker() = default;
    ~BenchmarkWorker();

    BenchmarkWorker(const BenchmarkWorker&) = delete;
    BenchmarkWorker& operator=(const BenchmarkWorker&) = delete;

    // Joins any prior run, then starts a new background run with `config`.
    void start(BenchmarkRunConfig config);

    bool isRunning() const { return running_.load(); }

    // Sets the cancel flag the running thread checks between images and
    // between slots. No-op if not running.
    void requestCancel();

    // Lock-free snapshot of current progress -- safe to call every frame.
    BenchmarkProgress progress() const;

    // Non-blocking poll:
    // returns true and moves the result out exactly once, on the frame
    // after the run finishes (normally or via cancel).
    bool tryTakeResult(BenchmarkRunResult& out);

private:
    void run(BenchmarkRunConfig config);
    void finish(BenchmarkRunResult result);

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
    BenchmarkRunResult latestResult_;
};

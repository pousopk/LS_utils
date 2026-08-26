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

struct BatchEvalRunConfig {
    ComparisonTaskMode mode = ComparisonTaskMode::Detection;
    std::string imageFolderPath;
    LabelStudioImportResult groundTruth;  // ignored unless hasGroundTruth
    bool hasGroundTruth = false;
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
};

struct BatchEvalRunResult {
    BatchEvaluationResult slotA;
    BatchEvaluationResult slotB;
    bool cancelled = false;
};

// Runs Model A then Model B sequentially over the same image folder on a
// single background thread, reporting live progress and honoring
// cancellation. Not copyable. Reuse one instance across runs -- start()
// joins any previous thread first.
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

    // Non-blocking poll, same pattern as InferenceWorker::tryTakeResult:
    // returns true and moves the result out exactly once, on the frame
    // after the run finishes (normally or via cancel).
    bool tryTakeResult(BatchEvalRunResult& out);

private:
    void run(BatchEvalRunConfig config);

    std::thread thread_;
    std::atomic<int> currentSlot_{0};
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    BatchEvalRunResult latestResult_;
};

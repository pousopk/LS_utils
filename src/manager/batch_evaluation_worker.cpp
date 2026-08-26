#include "manager/batch_evaluation_worker.hpp"

BatchEvaluationWorker::~BatchEvaluationWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void BatchEvaluationWorker::start(BatchEvalRunConfig config) {
    if (thread_.joinable()) {
        thread_.join();
    }
    cancelRequested_.store(false);
    completed_.store(0);
    total_.store(0);
    currentSlot_.store(0);
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hasResult_ = false;
    }
    running_.store(true);
    thread_ = std::thread(&BatchEvaluationWorker::run, this, std::move(config));
}

void BatchEvaluationWorker::requestCancel() {
    cancelRequested_.store(true);
}

BatchEvalProgress BatchEvaluationWorker::progress() const {
    return BatchEvalProgress{currentSlot_.load(), completed_.load(), total_.load()};
}

bool BatchEvaluationWorker::tryTakeResult(BatchEvalRunResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void BatchEvaluationWorker::run(BatchEvalRunConfig config) {
    const LabelStudioImportResult* groundTruth = config.hasGroundTruth ? &config.groundTruth : nullptr;
    BatchEvalRunResult result;

    currentSlot_.store(1);
    completed_.store(0);
    total_.store(0);
    const auto onProgressA = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };
    if (config.mode == ComparisonTaskMode::Detection) {
        result.slotA = runDetectionBatchEvaluation(
            config.imageFolderPath, groundTruth,
            [&](const cv::Mat& frame) {
                return config.detectionModelA->infer(frame, config.confThresholdA, config.nmsThresholdA);
            },
            onProgressA, &cancelRequested_);
    } else {
        result.slotA = runClassificationBatchEvaluation(
            config.imageFolderPath, groundTruth,
            [&](const cv::Mat& frame) { return config.classificationModelA->infer(frame); },
            onProgressA, &cancelRequested_);
    }

    if (!cancelRequested_.load() && config.runSlotB) {
        currentSlot_.store(2);
        completed_.store(0);
        total_.store(0);
        const auto onProgressB = [this](int completed, int total) {
            completed_.store(completed);
            total_.store(total);
        };
        if (config.mode == ComparisonTaskMode::Detection) {
            result.slotB = runDetectionBatchEvaluation(
                config.imageFolderPath, groundTruth,
                [&](const cv::Mat& frame) {
                    return config.detectionModelB->infer(frame, config.confThresholdB, config.nmsThresholdB);
                },
                onProgressB, &cancelRequested_);
        } else {
            result.slotB = runClassificationBatchEvaluation(
                config.imageFolderPath, groundTruth,
                [&](const cv::Mat& frame) { return config.classificationModelB->infer(frame); },
                onProgressB, &cancelRequested_);
        }
    }

    result.cancelled = cancelRequested_.load();

    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        latestResult_ = std::move(result);
        hasResult_ = true;
    }
    running_.store(false);
}

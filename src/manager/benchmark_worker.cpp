#include "manager/benchmark_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <random>

BenchmarkWorker::~BenchmarkWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void BenchmarkWorker::start(BenchmarkRunConfig config) {
    if (thread_.joinable()) {
        thread_.join();
    }
    cancelRequested_.store(false);
    completed_.store(0);
    total_.store(0);
    currentSlot_.store(0);
    {
        std::lock_guard<std::mutex> lock(phaseMutex_);
        phaseLabel_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hasResult_ = false;
    }
    running_.store(true);
    thread_ = std::thread(&BenchmarkWorker::run, this, std::move(config));
}

void BenchmarkWorker::requestCancel() {
    cancelRequested_.store(true);
}

BenchmarkProgress BenchmarkWorker::progress() const {
    std::lock_guard<std::mutex> lock(phaseMutex_);
    return BenchmarkProgress{currentSlot_.load(), completed_.load(), total_.load(), phaseLabel_};
}

bool BenchmarkWorker::tryTakeResult(BenchmarkRunResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void BenchmarkWorker::finish(BenchmarkRunResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void BenchmarkWorker::run(BenchmarkRunConfig config) {
    if (config.source == BenchmarkSourceMode::LabelStudioProject) {
        {
            std::lock_guard<std::mutex> lock(phaseMutex_);
            phaseLabel_ = "Downloading";
        }
        completed_.store(0);
        total_.store(0);

        const auto onDownloadProgress = [this](int completed, int total) {
            completed_.store(completed);
            total_.store(total);
        };

        const LabelStudioGroundTruthDataset dataset = fetchAndDownloadLabeledDataset(
            config.labelStudioBaseUrl, config.labelStudioProjectId, config.labelStudioApiToken,
            config.labelStudioDataImageKey, config.scratchFolderPath, onDownloadProgress, &cancelRequested_);

        if (cancelRequested_.load()) {
            BenchmarkRunResult result;
            result.cancelled = true;
            finish(std::move(result));
            return;
        }
        if (!dataset.error.empty()) {
            BenchmarkRunResult result;
            result.slotA.error = dataset.error;
            finish(std::move(result));
            return;
        }

        config.imageFolderPath = config.scratchFolderPath;
        config.groundTruth = dataset.groundTruth;
        config.hasGroundTruth = true;

        {
            std::lock_guard<std::mutex> lock(phaseMutex_);
            phaseLabel_.clear();
        }
    }

    const LabelStudioImportResult* groundTruth = config.hasGroundTruth ? &config.groundTruth : nullptr;
    // Drawn once per run and reused for both slots -- if sampling is on, A
    // and B must be evaluated against the identical random subset or their
    // results aren't comparable.
    const unsigned sampleSeed = std::random_device{}();
    BenchmarkRunResult result;

    currentSlot_.store(1);
    completed_.store(0);
    total_.store(0);
    const auto onProgressA = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };
    if (config.mode == ModelTask::Detection) {
        result.slotA = runDetectionBenchmark(
            config.imageFolderPath, groundTruth,
            [&](const cv::Mat& frame) {
                return config.detectionModelA->infer(frame, config.confThresholdA, config.nmsThresholdA);
            },
            onProgressA, &cancelRequested_, config.sampleSize, sampleSeed);
    } else if (config.mode == ModelTask::Classification) {
        result.slotA = runClassificationBenchmark(
            config.imageFolderPath, groundTruth,
            [&](const cv::Mat& frame) { return config.classificationModelA->infer(frame); },
            onProgressA, &cancelRequested_, config.sampleSize, sampleSeed);
    } else {
        result.slotA = runAnomalyBenchmark(
            config.imageFolderPath, groundTruth,
            [&](const cv::Mat& frame) { return config.anomalyModelA->infer(frame, config.anomalyThresholdA); },
            onProgressA, &cancelRequested_, config.sampleSize, sampleSeed);
    }

    if (!cancelRequested_.load() && config.runSlotB) {
        currentSlot_.store(2);
        completed_.store(0);
        total_.store(0);
        const auto onProgressB = [this](int completed, int total) {
            completed_.store(completed);
            total_.store(total);
        };
        if (config.mode == ModelTask::Detection) {
            result.slotB = runDetectionBenchmark(
                config.imageFolderPath, groundTruth,
                [&](const cv::Mat& frame) {
                    return config.detectionModelB->infer(frame, config.confThresholdB, config.nmsThresholdB);
                },
                onProgressB, &cancelRequested_, config.sampleSize, sampleSeed);
        } else if (config.mode == ModelTask::Classification) {
            result.slotB = runClassificationBenchmark(
                config.imageFolderPath, groundTruth,
                [&](const cv::Mat& frame) { return config.classificationModelB->infer(frame); },
                onProgressB, &cancelRequested_, config.sampleSize, sampleSeed);
        } else {
            result.slotB = runAnomalyBenchmark(
                config.imageFolderPath, groundTruth,
                [&](const cv::Mat& frame) { return config.anomalyModelB->infer(frame, config.anomalyThresholdB); },
                onProgressB, &cancelRequested_, config.sampleSize, sampleSeed);
        }
    }

    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

#include "manager/benchmark_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <filesystem>
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

namespace {

BenchmarkRunResult failedRun(const std::string& error) {
    BenchmarkRunResult result;
    result.slotA.error = error;
    return result;
}

} // namespace

void BenchmarkWorker::run(BenchmarkRunConfig config) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    int totalAvailable = 0;

    if (config.source == BenchmarkSourceMode::LabelStudioProject) {
        if (config.tasks.empty()) {
            finish(failedRun("No labeled tasks to evaluate in the loaded task list"));
            return;
        }
        std::vector<LabelStudioUnlabeledTask> missing;
        for (const auto& task : config.tasks) {
            const fs::path path = fs::path(config.cacheFolderPath) / taskImageLocalFilename(task.taskId, task.imagePath);
            std::error_code ec;
            if (!fs::exists(path, ec)) {
                missing.push_back(task);
            }
            files.push_back(path);
        }
        totalAvailable = config.labeledTaskCount;

        if (!missing.empty()) {
            const int cached = static_cast<int>(config.tasks.size() - missing.size());
            {
                std::lock_guard<std::mutex> lock(phaseMutex_);
                phaseLabel_ = cached > 0 ? "Downloading (" + std::to_string(cached) + " cached)" : "Downloading";
            }
            completed_.store(0);
            total_.store(static_cast<int>(missing.size()));
            const auto onDownloadProgress = [this](int completed, int total) {
                completed_.store(completed);
                total_.store(total);
            };
            downloadUnlabeledTaskImages(
                config.labelStudioBaseUrl, config.labelStudioApiToken, missing, config.cacheFolderPath,
                onDownloadProgress, &cancelRequested_);
            {
                std::lock_guard<std::mutex> lock(phaseMutex_);
                phaseLabel_.clear();
            }
            if (cancelRequested_.load()) {
                BenchmarkRunResult result;
                result.cancelled = true;
                finish(std::move(result));
                return;
            }
        }
    } else {
        files = listImageFiles(config.imageFolderPath);
        if (files.empty()) {
            finish(failedRun("No recognized image files found in: " + config.imageFolderPath));
            return;
        }
        totalAvailable = static_cast<int>(files.size());
        files = sampleImageFiles(std::move(files), config.sampleSize, std::random_device{}());
    }

    const LabelStudioImportResult* groundTruth = config.hasGroundTruth ? &config.groundTruth : nullptr;
    const auto evaluate = [&](int slot, const std::shared_ptr<YoloModel>& detectionModel,
                              const std::shared_ptr<ClassificationModel>& classificationModel,
                              const std::shared_ptr<AnomalyModel>& anomalyModel, float confThreshold,
                              float nmsThreshold, float anomalyThreshold) {
        currentSlot_.store(slot);
        completed_.store(0);
        total_.store(0);
        const auto onProgress = [this](int completed, int total) {
            completed_.store(completed);
            total_.store(total);
        };
        BenchmarkResult result;
        if (config.mode == ModelTask::Detection) {
            result = runDetectionBenchmark(
                files, groundTruth,
                [&](const cv::Mat& frame) { return detectionModel->infer(frame, confThreshold, nmsThreshold); },
                onProgress, &cancelRequested_);
        } else if (config.mode == ModelTask::Classification) {
            result = runClassificationBenchmark(
                files, groundTruth, [&](const cv::Mat& frame) { return classificationModel->infer(frame); },
                onProgress, &cancelRequested_);
        } else {
            result = runAnomalyBenchmark(
                files, [&](const cv::Mat& frame) { return anomalyModel->infer(frame, anomalyThreshold); },
                onProgress, &cancelRequested_);
        }
        result.totalAvailableImages = totalAvailable;
        return result;
    };

    BenchmarkRunResult result;
    result.slotA = evaluate(
        1, config.detectionModelA, config.classificationModelA, config.anomalyModelA, config.confThresholdA,
        config.nmsThresholdA, config.anomalyThresholdA);
    if (!cancelRequested_.load() && config.runSlotB) {
        result.slotB = evaluate(
            2, config.detectionModelB, config.classificationModelB, config.anomalyModelB, config.confThresholdB,
            config.nmsThresholdB, config.anomalyThresholdB);
    }

    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

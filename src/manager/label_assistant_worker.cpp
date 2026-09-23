#include "manager/label_assistant_worker.hpp"

#include "manager/label_studio_client.hpp"

LabelAssistantWorker::~LabelAssistantWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void LabelAssistantWorker::start(LabelAssistantRunConfig config) {
    if (thread_.joinable()) {
        thread_.join();
    }
    cancelRequested_.store(false);
    completed_.store(0);
    total_.store(0);
    {
        std::lock_guard<std::mutex> lock(phaseMutex_);
        phaseLabel_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hasResult_ = false;
    }
    running_.store(true);
    thread_ = std::thread(&LabelAssistantWorker::run, this, std::move(config));
}

void LabelAssistantWorker::requestCancel() {
    cancelRequested_.store(true);
}

LabelAssistantProgress LabelAssistantWorker::progress() const {
    std::lock_guard<std::mutex> lock(phaseMutex_);
    return LabelAssistantProgress{completed_.load(), total_.load(), phaseLabel_};
}

bool LabelAssistantWorker::tryTakeResult(LabelAssistantRunResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void LabelAssistantWorker::finish(LabelAssistantRunResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void LabelAssistantWorker::beginPhase(const std::string& label) {
    {
        std::lock_guard<std::mutex> lock(phaseMutex_);
        phaseLabel_ = label;
    }
    completed_.store(0);
    total_.store(0);
}

void LabelAssistantWorker::run(LabelAssistantRunConfig config) {
    std::string folderToScan = config.imageFolderPath;

    if (config.source == LabelAssistantSourceMode::LabelStudioProject) {
        beginPhase("Downloading");

        const auto onDownloadProgress = [this](int completed, int total) {
            completed_.store(completed);
            total_.store(total);
        };

        downloadUnlabeledTaskImages(
            config.labelStudioBaseUrl, config.labelStudioApiToken, config.unlabeledTasks, config.scratchFolderPath,
            onDownloadProgress, &cancelRequested_);

        if (cancelRequested_.load()) {
            LabelAssistantRunResult result;
            result.cancelled = true;
            finish(std::move(result));
            return;
        }

        folderToScan = config.scratchFolderPath;
        beginPhase("Running inference");
    }

    const auto onProgress = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };

    const std::function<std::vector<ClassPrediction>(const cv::Mat&)> classify = config.classificationModel
        ? std::function<std::vector<ClassPrediction>(const cv::Mat&)>(
              [model = config.classificationModel](const cv::Mat& frame) { return model->infer(frame); })
        : nullptr;

    const std::function<std::vector<Detection>(const cv::Mat&)> detect = config.detectionModel
        ? std::function<std::vector<Detection>(const cv::Mat&)>(
              [model = config.detectionModel, conf = config.confThreshold, nms = config.nmsThreshold](
                  const cv::Mat& frame) { return model->infer(frame, conf, nms); })
        : nullptr;

    LabelAssistantRunResult result;
    result.result = runAutoLabel(folderToScan, config.mode, classify, detect, onProgress, &cancelRequested_);
    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

#include "manager/label_studio_push_worker.hpp"

LabelStudioPushWorker::~LabelStudioPushWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void LabelStudioPushWorker::start(LabelStudioPushConfig config) {
    if (thread_.joinable()) {
        thread_.join();
    }
    cancelRequested_.store(false);
    completed_.store(0);
    total_.store(0);
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hasResult_ = false;
    }
    running_.store(true);
    thread_ = std::thread(&LabelStudioPushWorker::run, this, std::move(config));
}

void LabelStudioPushWorker::requestCancel() {
    cancelRequested_.store(true);
}

LabelStudioPushProgress LabelStudioPushWorker::progress() const {
    return LabelStudioPushProgress{completed_.load(), total_.load()};
}

bool LabelStudioPushWorker::tryTakeResult(LabelStudioPushRunResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void LabelStudioPushWorker::finish(LabelStudioPushRunResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void LabelStudioPushWorker::run(LabelStudioPushConfig config) {
    const auto onProgress = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };

    LabelStudioPushRunResult result;
    result.mode = config.mode;

    if (config.mode == LabelStudioPushMode::AttachToKnownTasks) {
        result.attachSummary = attachPredictionsToKnownTasks(
            config.baseUrl, config.apiToken, config.knownTaskPredictions, onProgress, &cancelRequested_);
    } else {
        result.uploadSummary = pushDraftsAsNewLabelStudioTasks(
            config.baseUrl, config.projectId, config.apiToken, config.newTaskPredictions, onProgress,
            &cancelRequested_);
    }

    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

#include "manager/timestamp_search_worker.hpp"

#include <algorithm>

TimestampSearchWorker::~TimestampSearchWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void TimestampSearchWorker::start(TimestampSearchRunConfig config) {
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
    thread_ = std::thread(&TimestampSearchWorker::run, this, std::move(config));
}

void TimestampSearchWorker::requestCancel() {
    cancelRequested_.store(true);
}

TimestampSearchProgress TimestampSearchWorker::progress() const {
    std::lock_guard<std::mutex> lock(phaseMutex_);
    return TimestampSearchProgress{completed_.load(), total_.load(), phaseLabel_};
}

bool TimestampSearchWorker::tryTakeResult(TimestampSearchRunResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void TimestampSearchWorker::finish(TimestampSearchRunResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void TimestampSearchWorker::beginPhase(const std::string& label) {
    {
        std::lock_guard<std::mutex> lock(phaseMutex_);
        phaseLabel_ = label;
    }
    completed_.store(0);
    total_.store(0);
}

void TimestampSearchWorker::run(TimestampSearchRunConfig config) {
    beginPhase("Downloading thumbnails");
    const auto onProgress = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };
    downloadLabelStudioTaskImages(
        config.labelStudioBaseUrl, config.labelStudioApiToken, config.candidates, config.scratchFolderPath,
        onProgress, &cancelRequested_);

    TimestampSearchRunResult result;
    result.scratchFolderPath = config.scratchFolderPath;
    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

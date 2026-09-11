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
    beginPhase("Searching");

    const FindTasksNearTimestampsResult searchResult = findTasksNearTimestamps(
        config.labelStudioBaseUrl, config.labelStudioProjectId, config.labelStudioApiToken,
        config.labelStudioDataImageKey, config.queries);

    if (!searchResult.error.empty()) {
        TimestampSearchRunResult result;
        result.error = searchResult.error;
        finish(std::move(result));
        return;
    }

    // Dedup by task id across queries before downloading -- a task
    // matching more than one typed entry should only be fetched once.
    std::vector<TimestampMatchCandidate> allCandidates;
    for (const auto& candidates : searchResult.perQuery) {
        for (const auto& candidate : candidates) {
            const bool alreadyIncluded = std::any_of(
                allCandidates.begin(), allCandidates.end(),
                [&](const TimestampMatchCandidate& c) { return c.taskId == candidate.taskId; });
            if (!alreadyIncluded) {
                allCandidates.push_back(candidate);
            }
        }
    }

    beginPhase("Downloading thumbnails");
    const auto onProgress = [this](int completed, int total) {
        completed_.store(completed);
        total_.store(total);
    };
    downloadLabelStudioTaskImages(
        config.labelStudioBaseUrl, config.labelStudioApiToken, allCandidates, config.scratchFolderPath, onProgress,
        &cancelRequested_);

    TimestampSearchRunResult result;
    result.perQuery = searchResult.perQuery;
    result.scratchFolderPath = config.scratchFolderPath;
    result.cancelled = cancelRequested_.load();
    finish(std::move(result));
}

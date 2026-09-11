#pragma once

#include "manager/label_studio_client.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct TimestampSearchRunConfig {
    std::string labelStudioBaseUrl;
    int labelStudioProjectId = 0;
    std::string labelStudioApiToken;
    std::string labelStudioDataImageKey;
    std::vector<TimestampMatchQuery> queries;   // one per still-valid typed entry, same order
    std::string scratchFolderPath;              // already-created, ideally-cleared download destination
};

struct TimestampSearchProgress {
    int completed = 0;
    int total = 0;
    std::string phaseLabel;   // "Searching" or "Downloading thumbnails"
};

struct TimestampSearchRunResult {
    std::vector<std::vector<TimestampMatchCandidate>> perQuery;   // matches TimestampSearchRunConfig.queries order/length
    std::string scratchFolderPath;   // where thumbnails were downloaded, for loading textures afterward
    std::string error;               // set only on a hard failure to fetch the task list
    bool cancelled = false;
};

// Runs a timestamp-search job on a single background thread: first fetches
// and matches tasks (findTasksNearTimestamps), then downloads every
// distinct matched task's image (deduplicated by task id across queries)
// into scratchFolderPath, reporting live progress and honoring
// cancellation during the download phase. Mirrors LabelAssistantWorker's
// exact shape. Not copyable. Reuse one instance across runs -- start()
// joins any previous thread first.
class TimestampSearchWorker {
public:
    TimestampSearchWorker() = default;
    ~TimestampSearchWorker();
    TimestampSearchWorker(const TimestampSearchWorker&) = delete;
    TimestampSearchWorker& operator=(const TimestampSearchWorker&) = delete;

    void start(TimestampSearchRunConfig config);
    bool isRunning() const { return running_.load(); }
    void requestCancel();
    TimestampSearchProgress progress() const;
    bool tryTakeResult(TimestampSearchRunResult& out);

private:
    void run(TimestampSearchRunConfig config);
    void finish(TimestampSearchRunResult result);
    void beginPhase(const std::string& label);

    std::thread thread_;
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex phaseMutex_;
    std::string phaseLabel_;

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    TimestampSearchRunResult latestResult_;
};

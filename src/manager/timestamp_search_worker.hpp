#pragma once

#include "manager/label_studio_client.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct TimestampSearchRunConfig {
    std::string labelStudioBaseUrl;
    std::string labelStudioApiToken;
    // Already matched and deduplicated (see dedupTimestampMatchCandidates)
    // on the main thread before this worker starts -- matching against
    // the shared project data doesn't need a network call, so it no
    // longer happens on this worker's thread.
    std::vector<TimestampMatchCandidate> candidates;
    std::string scratchFolderPath;   // already-created, ideally-cleared download destination
};

struct TimestampSearchProgress {
    int completed = 0;
    int total = 0;
    std::string phaseLabel;   // "Searching" or "Downloading thumbnails"
};

struct TimestampSearchRunResult {
    std::string scratchFolderPath;   // where thumbnails were downloaded, for loading textures afterward
    bool cancelled = false;
};

// Runs the thumbnail-download phase of a timestamp search on a single
// background thread: downloads every already-matched, already-deduplicated
// candidate's image (see dedupTimestampMatchCandidates) into
// scratchFolderPath, reporting live progress and honoring cancellation.
// Matching itself now happens on the main thread before this worker ever
// starts (see startTimestampSearch) -- it doesn't need a network call, so
// it doesn't need to be off the UI thread. Not copyable. Reuse one
// instance across runs -- start() joins any previous thread first.
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

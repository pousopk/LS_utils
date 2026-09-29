#pragma once

#include "manager/label_studio_client.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Which of Label Studio's two prediction-push mechanisms to use --
// mirrors LabelAssistantSourceMode's LocalFolder/LabelStudioProject
// split, since that's what decides which one applies (see
// pushLabelAssistantDraftsToLabelStudio).
enum class LabelStudioPushMode {
    UploadNewTasks,     // pushDraftsAsNewLabelStudioTasks
    AttachToKnownTasks, // attachPredictionsToKnownTasks
};

struct LabelStudioPushConfig {
    LabelStudioPushMode mode = LabelStudioPushMode::UploadNewTasks;
    std::string baseUrl;
    std::string apiToken;

    // UploadNewTasks only.
    int projectId = 0;
    std::vector<LabelStudioPredictionInput> newTaskPredictions;

    // AttachToKnownTasks only.
    std::vector<LabelStudioKnownTaskPrediction> knownTaskPredictions;
};

struct LabelStudioPushProgress {
    int completed = 0;
    int total = 0;
};

struct LabelStudioPushRunResult {
    LabelStudioPushMode mode = LabelStudioPushMode::UploadNewTasks;
    LabelStudioPushSummary uploadSummary;     // set when mode == UploadNewTasks
    LabelStudioAttachSummary attachSummary;   // set when mode == AttachToKnownTasks
    bool cancelled = false;
};

// Runs pushDraftsAsNewLabelStudioTasks or attachPredictionsToKnownTasks
// (picked by config.mode) on a single background thread, reporting live
// progress and honoring cancellation -- mirrors LabelAssistantWorker's
// shape (start/isRunning/requestCancel/progress/tryTakeResult). Moves
// what pushLabelAssistantDraftsToLabelStudio used to do as a direct,
// blocking call on the UI thread onto a worker thread instead, the same
// way every other multi-request Label Studio operation in this app
// already works. Not copyable; reuse one instance -- start() joins any
// previous thread first, so only one push is ever in flight.
class LabelStudioPushWorker {
public:
    LabelStudioPushWorker() = default;
    ~LabelStudioPushWorker();
    LabelStudioPushWorker(const LabelStudioPushWorker&) = delete;
    LabelStudioPushWorker& operator=(const LabelStudioPushWorker&) = delete;

    // Joins any prior run, then starts a new background push with `config`.
    void start(LabelStudioPushConfig config);

    bool isRunning() const { return running_.load(); }

    // Sets the cancel flag the running thread checks between items.
    // No-op if not running.
    void requestCancel();

    // Lock-free snapshot of current progress -- safe to call every frame.
    LabelStudioPushProgress progress() const;

    // Non-blocking poll: returns true and moves the result out exactly
    // once, on the frame after the push finishes (normally or via cancel).
    bool tryTakeResult(LabelStudioPushRunResult& out);

private:
    void run(LabelStudioPushConfig config);
    void finish(LabelStudioPushRunResult result);

    std::thread thread_;
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    LabelStudioPushRunResult latestResult_;
};

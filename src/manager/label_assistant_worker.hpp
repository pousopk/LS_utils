#pragma once

#include "manager/classification_inference.hpp"
#include "manager/comparison_task_mode.hpp"
#include "manager/label_assistant.hpp"
#include "manager/label_studio_client.hpp"
#include "manager/yolo_inference.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Where the images for a run come from. LabelStudioProject adds a
// download phase (see LabelAssistantWorker::run) ahead of the same
// inference phase LocalFolder already used.
enum class LabelAssistantSourceMode {
    LocalFolder,
    LabelStudioProject,
};

struct LabelAssistantRunConfig {
    ComparisonTaskMode mode;
    std::shared_ptr<YoloModel> detectionModel;               // set when mode == Detection
    std::shared_ptr<ClassificationModel> classificationModel; // set when mode == Classification
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;

    LabelAssistantSourceMode source = LabelAssistantSourceMode::LocalFolder;

    // LocalFolder: the folder to scan directly.
    std::string imageFolderPath;

    // LabelStudioProject: connection details for the download phase, the
    // already-selected unlabeled tasks to download (selected on the main
    // thread from the shared project data before this worker starts --
    // see startLabelAssistantRun), and the (already-created,
    // ideally-cleared) scratch folder downloaded images are written into
    // -- this becomes the folder scanned for the inference phase, exactly
    // as if it were imageFolderPath.
    std::string labelStudioBaseUrl;
    std::string labelStudioApiToken;
    std::vector<LabelStudioUnlabeledTask> unlabeledTasks;
    std::string scratchFolderPath;
};

struct LabelAssistantProgress {
    int completed = 0;
    int total = 0;
    std::string phaseLabel;   // e.g. "Downloading" or "Running inference"; empty in LocalFolder mode
};

struct LabelAssistantRunResult {
    LabelAssistantResult result;
    bool cancelled = false;
};

// Runs a label-assistant job on a single background thread, reporting
// live progress and honoring cancellation. Not copyable. Reuse one
// instance across runs -- start() joins any previous thread first.
// Mirrors BatchEvaluationWorker's exact shape, minus the two-slot
// machinery (one model, one run). In LabelStudioProject source mode,
// run() does two sequential phases -- download (via
// downloadUnlabeledTaskImages, given the already-selected task list in
// config.unlabeledTasks) then the same inference phase LocalFolder mode
// uses (via runAutoLabel) -- each restarting completed/total from zero,
// distinguished by LabelAssistantProgress::phaseLabel. Unlike before,
// this worker no longer fetches or selects which tasks are unlabeled
// itself -- that happens on the main thread, against the shared project
// data, before this worker ever starts.
class LabelAssistantWorker {
public:
    LabelAssistantWorker() = default;
    ~LabelAssistantWorker();
    LabelAssistantWorker(const LabelAssistantWorker&) = delete;
    LabelAssistantWorker& operator=(const LabelAssistantWorker&) = delete;

    // Joins any prior run, then starts a new background run with `config`.
    void start(LabelAssistantRunConfig config);

    bool isRunning() const { return running_.load(); }

    // Sets the cancel flag the running thread checks between images.
    // No-op if not running.
    void requestCancel();

    // Lock-free snapshot of current progress -- safe to call every frame.
    LabelAssistantProgress progress() const;

    // Non-blocking poll: returns true and moves the result out exactly
    // once, on the frame after the run finishes (normally or via cancel).
    bool tryTakeResult(LabelAssistantRunResult& out);

private:
    void run(LabelAssistantRunConfig config);
    void finish(LabelAssistantRunResult result);
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
    LabelAssistantRunResult latestResult_;
};

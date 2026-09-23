#pragma once

#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct DatasetTaskListConfig {
    std::string baseUrl;
    int projectId = 0;
    std::string apiToken;
    std::string dataImageKey;
    std::string rectangleLabelsFromName;
    std::string brushLabelsFromName;
};

struct DatasetTaskListResult {
    bool success = false;
    std::string error;   // set only if !success
    nlohmann::json rawTasksJson;
    std::vector<DatasetTaskSummary> summaries;
    std::unordered_map<int, size_t> summaryIndexByTaskId;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetMasksToDraw> masksByTaskId;
};

// Runs fetchAllLabelStudioTasksRaw, then the same O(N) precompute
// refreshDatasetBrowserTaskList used to do inline on the main thread
// (summarizeDatasetTasks, indexSummariesByTaskId, buildBoxesByTaskId,
// buildMasksByTaskId -- all pure functions, safe to run on any thread),
// on a single background thread. Mirrors LabelStudioPushWorker's shape
// (start/isRunning/tryTakeResult); the main thread just moves the
// finished result into DatasetBrowserState. Unlike
// DatasetThumbnailWorker, this reports no numeric progress and offers no
// mid-fetch cancellation: fetchAllLabelStudioTasksRaw has no per-page
// progress/cancel hook (unlike the per-item download loops elsewhere in
// this app), so the only meaningful state to show while this runs is the
// same indeterminate "still working" text LabelStudioSessionStatus::
// Connecting already uses elsewhere in this app for a single in-flight
// network call. Not copyable; reuse one instance -- start() joins any
// previous thread first.
class DatasetTaskListWorker {
public:
    DatasetTaskListWorker() = default;
    ~DatasetTaskListWorker();
    DatasetTaskListWorker(const DatasetTaskListWorker&) = delete;
    DatasetTaskListWorker& operator=(const DatasetTaskListWorker&) = delete;

    void start(DatasetTaskListConfig config);
    bool isRunning() const { return running_.load(); }

    // Non-blocking poll: returns true and moves the result out exactly
    // once, on the frame after the fetch+precompute finishes.
    bool tryTakeResult(DatasetTaskListResult& out);

private:
    void run(DatasetTaskListConfig config);
    void finish(DatasetTaskListResult result);

    std::thread thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    DatasetTaskListResult latestResult_;
};

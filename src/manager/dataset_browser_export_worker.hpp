#pragma once

#include "manager/label_studio_client.hpp"
#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct DatasetExportConfig {
    std::string baseUrl;
    std::string apiToken;
    int projectId = 0;
    std::string destinationFolder;               // already exists by the time start() is called
    std::vector<DatasetTaskSummary> tasksToExport;
    std::vector<int> matchingTaskIds;             // the tasks whose raw JSON goes into export.json
    TaskCreatedAtBounds createdAtBounds;          // the loaded list's import-date window, so the re-fetch is just as narrow
};

enum class DatasetExportPhase { FetchingTasks, DownloadingImages };

struct DatasetExportProgress {
    DatasetExportPhase phase = DatasetExportPhase::FetchingTasks;
    int completed = 0;   // FetchingTasks: tasks fetched so far; DownloadingImages: images attempted
    int total = 0;       // FetchingTasks: the project's task count, 0 if unknown; DownloadingImages: images to fetch
};

struct DatasetExportResult {
    int downloaded = 0;
    int downloadFailed = 0;
    std::vector<std::string> sampleErrors;   // up to 5 distinct download error messages, so failures aren't a bare count
    bool exportJsonWritten = false;
    std::string exportJsonError;             // set only if writing export.json itself failed
    bool cancelled = false;
};

// Runs in two phases. First, re-streams the project's tasks via
// forEachLabelStudioTaskPage and writes just the ones in
// config.matchingTaskIds (via writeMatchingTasksAsJsonArrayElements) to
// `<destinationFolder>/export.json.partial`, renaming it to export.json
// once complete -- the shared task list no longer keeps every task's raw
// JSON in memory, so the export fetches what it needs itself. A cancel or
// fetch failure here leaves export.json.partial behind and skips the
// image downloads. Then downloads every task in config.tasksToExport's
// image into config.destinationFolder (named by original basename, same
// convention as fetchAndDownloadLabeledDataset). Mirrors LabelStudioPushWorker's
// shape (start/isRunning/requestCancel/progress/tryTakeResult). Unlike
// most download paths in this app, individual download failures also
// keep a capped sample of their actual error messages
// (DatasetExportResult::sampleErrors) rather than only a count, so a
// partial failure can be diagnosed without re-instrumenting the code.
// Not copyable; reuse one instance -- start() joins any previous thread
// first.
class DatasetExportWorker {
public:
    DatasetExportWorker() = default;
    ~DatasetExportWorker();
    DatasetExportWorker(const DatasetExportWorker&) = delete;
    DatasetExportWorker& operator=(const DatasetExportWorker&) = delete;

    void start(DatasetExportConfig config);
    bool isRunning() const { return running_.load(); }
    void requestCancel();
    DatasetExportProgress progress() const;
    bool tryTakeResult(DatasetExportResult& out);

private:
    void run(DatasetExportConfig config);
    void finish(DatasetExportResult result);

    std::thread thread_;
    std::atomic<int> phase_{0};
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    DatasetExportResult latestResult_;
};

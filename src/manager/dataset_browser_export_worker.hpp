#pragma once

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
    std::string destinationFolder;               // already exists by the time start() is called
    std::vector<DatasetTaskSummary> tasksToExport;
    nlohmann::json exportJson;                    // from buildDatasetExportJson, written verbatim to export.json
};

struct DatasetExportProgress {
    int completed = 0;
    int total = 0;
};

struct DatasetExportResult {
    int downloaded = 0;
    int downloadFailed = 0;
    std::vector<std::string> sampleErrors;   // up to 5 distinct download error messages, so failures aren't a bare count
    bool exportJsonWritten = false;
    std::string exportJsonError;             // set only if writing export.json itself failed
    bool cancelled = false;
};

// Downloads every task in config.tasksToExport's image into
// config.destinationFolder (named by original basename, same convention
// as fetchAndDownloadLabeledDataset), then writes config.exportJson to
// `<destinationFolder>/export.json`. Mirrors LabelStudioPushWorker's
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
    std::atomic<int> completed_{0};
    std::atomic<int> total_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    DatasetExportResult latestResult_;
};

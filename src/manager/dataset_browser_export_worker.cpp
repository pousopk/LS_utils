#include "manager/dataset_browser_export_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <filesystem>
#include <fstream>

namespace {
constexpr size_t kMaxSampleErrors = 5;
}

DatasetExportWorker::~DatasetExportWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DatasetExportWorker::start(DatasetExportConfig config) {
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
    thread_ = std::thread(&DatasetExportWorker::run, this, std::move(config));
}

void DatasetExportWorker::requestCancel() {
    cancelRequested_.store(true);
}

DatasetExportProgress DatasetExportWorker::progress() const {
    return DatasetExportProgress{completed_.load(), total_.load()};
}

bool DatasetExportWorker::tryTakeResult(DatasetExportResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void DatasetExportWorker::finish(DatasetExportResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void DatasetExportWorker::run(DatasetExportConfig config) {
    DatasetExportResult result;
    const int total = static_cast<int>(config.tasksToExport.size());
    total_.store(total);
    int completed = 0;

    for (const auto& task : config.tasksToExport) {
        if (cancelRequested_.load()) {
            result.cancelled = true;
            finish(std::move(result));
            return;
        }

        const std::string basename = std::filesystem::path(task.imagePath).filename().string();
        const std::string localPath = (std::filesystem::path(config.destinationFolder) / basename).string();

        std::string downloadError;
        if (downloadTaskImage(config.baseUrl, config.apiToken, task.imagePath, localPath, downloadError)) {
            result.downloaded++;
        } else {
            result.downloadFailed++;
            if (result.sampleErrors.size() < kMaxSampleErrors) {
                result.sampleErrors.push_back(downloadError);
            }
        }

        completed++;
        completed_.store(completed);
    }

    const std::string exportJsonPath = (std::filesystem::path(config.destinationFolder) / "export.json").string();
    std::ofstream file(exportJsonPath);
    if (file) {
        file << config.exportJson.dump(2);
        result.exportJsonWritten = static_cast<bool>(file);
    }
    if (!result.exportJsonWritten) {
        result.exportJsonError = "Could not write export.json to " + config.destinationFolder;
    }

    finish(std::move(result));
}

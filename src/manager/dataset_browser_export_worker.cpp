#include "manager/dataset_browser_export_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <filesystem>
#include <fstream>
#include <unordered_set>

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
    phase_.store(static_cast<int>(DatasetExportPhase::FetchingTasks));
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
    return DatasetExportProgress{static_cast<DatasetExportPhase>(phase_.load()), completed_.load(), total_.load()};
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

    const std::filesystem::path finalPath = std::filesystem::path(config.destinationFolder) / "export.json";
    const std::filesystem::path partialPath = std::filesystem::path(config.destinationFolder) / "export.json.partial";
    {
        std::ofstream file(partialPath);
        if (!file) {
            result.exportJsonError = "Could not write " + partialPath.string();
            finish(std::move(result));
            return;
        }
        std::unordered_set<int> remainingIds(config.matchingTaskIds.begin(), config.matchingTaskIds.end());
        bool wroteAny = false;
        file << "[\n";
        std::string fetchError;
        const LabelStudioPagedFetchOutcome outcome = forEachLabelStudioTaskPage(
            config.baseUrl, config.projectId, config.apiToken,
            [&](const nlohmann::json& pageTasks, const LabelStudioTaskPageProgress& progress) {
                writeMatchingTasksAsJsonArrayElements(pageTasks, remainingIds, file, wroteAny);
                completed_.store(progress.fetchedTasks);
                total_.store(progress.totalTasks.value_or(0));
                return static_cast<bool>(file);
            },
            &cancelRequested_, fetchError, config.createdAtBounds);
        file << "\n]\n";
        file.close();

        // Cancelled without a cancel request means the callback stopped
        // the fetch because the file stream failed -- a write error.
        if (outcome == LabelStudioPagedFetchOutcome::Cancelled && cancelRequested_.load()) {
            result.cancelled = true;
            finish(std::move(result));
            return;
        }
        if (outcome != LabelStudioPagedFetchOutcome::Completed || !file) {
            result.exportJsonError = outcome == LabelStudioPagedFetchOutcome::Failed
                ? "Fetching task data for export.json failed: " + fetchError
                : "Could not write " + partialPath.string();
            finish(std::move(result));
            return;
        }
        std::error_code renameError;
        std::filesystem::rename(partialPath, finalPath, renameError);
        if (renameError) {
            result.exportJsonError = "Could not rename export.json.partial: " + renameError.message();
            finish(std::move(result));
            return;
        }
        result.exportJsonWritten = true;
    }

    phase_.store(static_cast<int>(DatasetExportPhase::DownloadingImages));
    completed_.store(0);
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

    finish(std::move(result));
}

#include "manager/dataset_browser_tasklist_worker.hpp"

#include "manager/label_studio_client.hpp"

DatasetTaskListWorker::~DatasetTaskListWorker() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DatasetTaskListWorker::start(DatasetTaskListConfig config) {
    if (thread_.joinable()) {
        thread_.join();
    }
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hasResult_ = false;
    }
    running_.store(true);
    thread_ = std::thread(&DatasetTaskListWorker::run, this, std::move(config));
}

bool DatasetTaskListWorker::tryTakeResult(DatasetTaskListResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void DatasetTaskListWorker::finish(DatasetTaskListResult result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    latestResult_ = std::move(result);
    hasResult_ = true;
    running_.store(false);
}

void DatasetTaskListWorker::run(DatasetTaskListConfig config) {
    DatasetTaskListResult result;

    nlohmann::json allTasks;
    std::string error;
    if (!fetchAllLabelStudioTasksRaw(config.baseUrl, config.projectId, config.apiToken, allTasks, error)) {
        result.error = error;
        finish(std::move(result));
        return;
    }

    result.rawTasksJson = std::move(allTasks);
    result.summaries = summarizeDatasetTasks(result.rawTasksJson, config.dataImageKey);
    result.summaryIndexByTaskId = indexSummariesByTaskId(result.summaries);
    result.boxesByTaskId = buildBoxesByTaskId(result.rawTasksJson, config.rectangleLabelsFromName);
    result.masksByTaskId = buildMasksByTaskId(result.rawTasksJson, config.brushLabelsFromName);
    result.success = true;

    finish(std::move(result));
}

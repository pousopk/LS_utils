#include "manager/labeling_worker.hpp"

#include <filesystem>

LabelingWorker::~LabelingWorker() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void LabelingWorker::start(LabelingJobRequest request) {
    if (thread_.joinable()) {
        thread_.join();
    }
    running_ = true;
    thread_ = std::thread([this, request = std::move(request)]() mutable { run(std::move(request)); });
}

void LabelingWorker::finish(LabelingJobResult result) {
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        latestResult_ = std::move(result);
        hasResult_ = true;
    }
    running_ = false;
}

bool LabelingWorker::tryTakeResult(LabelingJobResult& out) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!hasResult_) {
        return false;
    }
    out = std::move(latestResult_);
    hasResult_ = false;
    return true;
}

void LabelingWorker::run(LabelingJobRequest request) {
    LabelingJobResult result;
    result.kind = request.kind;

    switch (request.kind) {
        case LabelingJobKind::FetchTaskList: {
            const auto& job = request.taskListJob;
            const auto listResult =
                fetchLabelStudioTaskSummaries(job.baseUrl, job.projectId, job.apiToken, job.dataImageKey);
            result.taskList = listResult.tasks;
            result.taskListError = listResult.error;
            break;
        }
        case LabelingJobKind::FetchTaskDetail: {
            const auto& job = request.taskDetailJob;
            result.taskDetail = fetchLabelStudioTaskById(job.baseUrl, job.apiToken, job.taskId, job.dataImageKey);
            if (!result.taskDetail.error.empty()) {
                result.taskDetailError = result.taskDetail.error;
                break;
            }
            const std::string extension = std::filesystem::path(result.taskDetail.imagePath).extension().string();
            const std::string localPath =
                job.scratchFolderPath + "/" + std::to_string(job.taskId) + (extension.empty() ? ".jpg" : extension);
            std::string downloadError;
            if (!downloadTaskImage(job.baseUrl, job.apiToken, result.taskDetail.imagePath, localPath, downloadError)) {
                result.taskDetailError = downloadError;
                break;
            }
            result.localImagePath = localPath;
            break;
        }
        case LabelingJobKind::SubmitAnnotation: {
            const auto& job = request.submitJob;
            const auto writeResult = job.existingAnnotationId.has_value()
                ? updateLabelStudioAnnotation(job.baseUrl, job.apiToken, *job.existingAnnotationId, job.resultArray)
                : createLabelStudioAnnotation(job.baseUrl, job.apiToken, job.taskId, job.resultArray);
            result.submitSuccess = writeResult.success;
            result.submitError = writeResult.error;
            break;
        }
    }

    finish(std::move(result));
}

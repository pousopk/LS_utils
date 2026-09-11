#pragma once

#include "manager/label_studio_client.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

enum class LabelingJobKind {
    FetchTaskList,
    FetchTaskDetail,
    SubmitAnnotation,
};

struct LabelingTaskListJob {
    std::string baseUrl;
    int projectId = 0;
    std::string apiToken;
    std::string dataImageKey;
};

struct LabelingTaskDetailJob {
    std::string baseUrl;
    std::string apiToken;
    int taskId = 0;
    std::string dataImageKey;
    std::string scratchFolderPath;   // already-created; the task's image is downloaded here as "<taskId><ext>"
};

struct LabelingSubmitJob {
    std::string baseUrl;
    std::string apiToken;
    int taskId = 0;
    std::optional<int> existingAnnotationId;   // set -> PATCH update; unset -> POST create
    nlohmann::json resultArray;
};

struct LabelingJobRequest {
    LabelingJobKind kind = LabelingJobKind::FetchTaskList;
    LabelingTaskListJob taskListJob;
    LabelingTaskDetailJob taskDetailJob;
    LabelingSubmitJob submitJob;
};

struct LabelingJobResult {
    LabelingJobKind kind = LabelingJobKind::FetchTaskList;

    std::vector<LabelStudioTaskSummary> taskList;   // FetchTaskList
    std::string taskListError;

    LabelStudioTaskDetail taskDetail;               // FetchTaskDetail
    std::string localImagePath;                     // FetchTaskDetail: downloaded image path, empty on failure
    std::string taskDetailError;

    bool submitSuccess = false;                     // SubmitAnnotation
    std::string submitError;
};

// Runs exactly one of FetchTaskList/FetchTaskDetail/SubmitAnnotation on a
// single background thread, following the same shape as
// TimestampSearchWorker/LabelAssistantWorker. Not copyable. Reuse one
// instance -- start() joins any previous thread first, so only one job is
// ever in flight, matching the UI's expectation that the user waits for
// one operation before starting another.
class LabelingWorker {
public:
    LabelingWorker() = default;
    ~LabelingWorker();
    LabelingWorker(const LabelingWorker&) = delete;
    LabelingWorker& operator=(const LabelingWorker&) = delete;

    void start(LabelingJobRequest request);
    bool isRunning() const { return running_.load(); }
    bool tryTakeResult(LabelingJobResult& out);

private:
    void run(LabelingJobRequest request);
    void finish(LabelingJobResult result);

    std::thread thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex resultMutex_;
    bool hasResult_ = false;
    LabelingJobResult latestResult_;
};

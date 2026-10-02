#include "manager/dataset_browser_tasklist_worker.hpp"

#include "manager/label_studio_client.hpp"

DatasetTaskListPage buildDatasetTaskListPage(const nlohmann::json& pageTasks, const DatasetTaskListConfig& config) {
    DatasetTaskListPage page;
    page.summaries = summarizeDatasetTasks(pageTasks, config.dataImageKey);
    if (!config.createdAtBounds.unbounded()) {
        const size_t before = page.summaries.size();
        std::erase_if(page.summaries, [&config](const DatasetTaskSummary& summary) {
            return !isTaskCreatedWithin(summary.createdAt, config.createdAtBounds);
        });
        page.droppedOutOfRange = static_cast<int>(before - page.summaries.size());
    }
    page.boxesByTaskId = buildBoxesByTaskId(pageTasks, config.rectangleLabelsFromName);
    page.masksByTaskId = buildEncodedMasksByTaskId(pageTasks, config.brushLabelsFromName);
    page.groundTruthByTaskId = buildGroundTruthByTaskId(pageTasks);
    return page;
}

DatasetTaskListWorker::~DatasetTaskListWorker() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DatasetTaskListWorker::stop() {
    requestCancel();
    if (thread_.joinable()) {
        thread_.join();
    }
    std::lock_guard<std::mutex> lock(queueMutex_);
    queuedPages_.clear();
    completion_.reset();
}

void DatasetTaskListWorker::start(DatasetTaskListConfig config) {
    stop();
    cancelRequested_.store(false);
    fetchedTasks_.store(0);
    totalTasks_.store(-1);
    running_.store(true);
    thread_ = std::thread(&DatasetTaskListWorker::run, this, std::move(config));
}

void DatasetTaskListWorker::requestCancel() {
    cancelRequested_.store(true);
}

bool DatasetTaskListWorker::isRunning() const {
    return running_.load();
}

DatasetTaskListProgress DatasetTaskListWorker::progress() const {
    DatasetTaskListProgress progress;
    progress.fetchedTasks = fetchedTasks_.load();
    const int total = totalTasks_.load();
    if (total >= 0) {
        progress.totalTasks = total;
    }
    return progress;
}

void DatasetTaskListWorker::poll(DatasetTaskListPoll& out) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    out.pages = std::move(queuedPages_);
    queuedPages_.clear();
    out.completion = std::move(completion_);
    completion_.reset();
}

void DatasetTaskListWorker::run(DatasetTaskListConfig config) {
    std::string error;
    const LabelStudioPagedFetchOutcome outcome = forEachLabelStudioTaskPage(
        config.baseUrl, config.projectId, config.apiToken,
        [this, &config](const nlohmann::json& pageTasks, const LabelStudioTaskPageProgress& progress) {
            DatasetTaskListPage page = buildDatasetTaskListPage(pageTasks, config);
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                queuedPages_.push_back(std::move(page));
            }
            fetchedTasks_.store(progress.fetchedTasks);
            totalTasks_.store(progress.totalTasks.value_or(-1));
            return !cancelRequested_.load();
        },
        &cancelRequested_, error, config.createdAtBounds);

    DatasetTaskListCompletion completion;
    completion.outcome = outcome == LabelStudioPagedFetchOutcome::Completed ? DatasetTaskListOutcome::Completed
        : outcome == LabelStudioPagedFetchOutcome::Cancelled               ? DatasetTaskListOutcome::Cancelled
                                                                            : DatasetTaskListOutcome::Failed;
    completion.error = error;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        completion_ = std::move(completion);
    }
    running_.store(false);
}

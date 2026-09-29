#pragma once

#include "manager/label_studio_client.hpp"
#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <optional>
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
    // Import-date window for this load; unbounded = the whole project.
    // Sent to Label Studio as a `query` filter and re-checked locally.
    TaskCreatedAtBounds createdAtBounds;
};

// One fetched page of the task list, already reduced to what the Label
// Studio tabs need -- the page's raw JSON is dropped as soon as this is
// built, so memory scales with these compact records, not with the
// project's full `fields=all` response.
struct DatasetTaskListPage {
    std::vector<DatasetTaskSummary> summaries;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetEncodedMasks> masksByTaskId;
    // Tasks Label Studio sent that fall outside config.createdAtBounds --
    // non-zero only when the server didn't apply the `query` filter.
    int droppedOutOfRange = 0;
};

// Pure function: summarizeDatasetTasks + buildBoxesByTaskId +
// buildEncodedMasksByTaskId over one page's tasks (all pure, safe on any
// thread), keeping only summaries whose createdAt passes
// isTaskCreatedWithin(config.createdAtBounds) and counting the rest in
// droppedOutOfRange.
DatasetTaskListPage buildDatasetTaskListPage(const nlohmann::json& pageTasks, const DatasetTaskListConfig& config);

enum class DatasetTaskListOutcome { Completed, Cancelled, Failed };

struct DatasetTaskListCompletion {
    DatasetTaskListOutcome outcome = DatasetTaskListOutcome::Completed;
    std::string error;   // set only for Failed
};

struct DatasetTaskListProgress {
    int fetchedTasks = 0;
    std::optional<int> totalTasks;   // unset until Label Studio reports a "total"
};

struct DatasetTaskListPoll {
    std::vector<DatasetTaskListPage> pages;
    std::optional<DatasetTaskListCompletion> completion;
};

// Streams the project's task list on a single background thread via
// forEachLabelStudioTaskPage: each page is converted to a
// DatasetTaskListPage on this thread and queued, and the page's raw JSON
// is freed before the next page is requested. The main thread calls
// poll() once per frame to take whatever pages have arrived (so tabs can
// show tasks while the rest are still loading) and, at the end, the
// run's completion. Reports numeric progress via progress().
// requestCancel() also aborts a request already in flight (see
// forEachLabelStudioTaskPage), so stop()/start()/the destructor -- which
// all cancel before joining -- return within about a second instead of
// waiting out the rest of the project. Not copyable; reuse one instance.
class DatasetTaskListWorker {
public:
    DatasetTaskListWorker() = default;
    ~DatasetTaskListWorker();
    DatasetTaskListWorker(const DatasetTaskListWorker&) = delete;
    DatasetTaskListWorker& operator=(const DatasetTaskListWorker&) = delete;

    // stop()s any previous run first, so none of its pages can leak into this one.
    void start(DatasetTaskListConfig config);
    // requestCancel() + join + discard any queued pages/completion.
    void stop();
    void requestCancel();
    bool isRunning() const;
    DatasetTaskListProgress progress() const;

    // Non-blocking: moves out every queued page and, if the run has
    // finished, its completion -- atomically, so a completion is never
    // observed before the pages that preceded it.
    void poll(DatasetTaskListPoll& out);

private:
    void run(DatasetTaskListConfig config);

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> cancelRequested_{false};
    std::atomic<int> fetchedTasks_{0};
    std::atomic<int> totalTasks_{-1};   // -1 = unknown

    mutable std::mutex queueMutex_;
    std::vector<DatasetTaskListPage> queuedPages_;
    std::optional<DatasetTaskListCompletion> completion_;
};

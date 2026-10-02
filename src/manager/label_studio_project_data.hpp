#pragma once

#include "manager/calendar_date.hpp"
#include "manager/dataset_browser_tasklist_worker.hpp"
#include "manager/label_studio_client.hpp"
#include "manager/label_studio_dataset_browser.hpp"
#include "manager/label_studio_session.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// One Label Studio project's task list + config, fetched once and shared
// by every tab that needs it (Labeling, Label Assistant, Find by
// Timestamp, Dataset Browser) -- refreshing from any one of them updates
// what every other one sees, instead of each independently re-fetching
// the whole project. DatasetTaskSummary/DatasetTaskListWorker/
// DatasetTaskListConfig/DatasetTaskListPage (used below) predate this
// file and still carry their original "Dataset" naming from when only
// the Dataset Browser used them; they're reused as-is here rather than
// renamed, since renaming every call site across four tabs for a
// naming-only change isn't worth the diff.
// Which tasks a load covers, by import date (a task's created_at), as
// whole local calendar days: `from` and `to` are both inclusive, and
// either unset means no limit on that side. Both unset -- the default --
// is the whole project.
struct TaskImportDateRange {
    std::optional<CalendarDate> from;
    std::optional<CalendarDate> to;

    bool operator==(const TaskImportDateRange&) const = default;
};

// from <= to when both are set; always true otherwise.
bool isValidTaskImportDateRange(const TaskImportDateRange& range);

// Local midnight of `from` up to (excluding) local midnight of the day
// after `to`, so the whole `to` day is included. Not pure (local
// timezone, via localMidnight).
TaskCreatedAtBounds toCreatedAtBounds(const TaskImportDateRange& range);

// Pure function: the note every Label Studio tab shows so a date-filtered
// list is never mistaken for the whole project -- "Tasks imported
// 2026/09/01 - 2026/09/15", "Tasks imported from 2026/09/01", "Tasks
// imported up to 2026/09/15", or "" for the whole project.
std::string describeTaskImportDateRange(const TaskImportDateRange& range);

// Where a refresh of an already-loaded project accumulates its pages, so
// the previous complete list stays visible (and intact, if the refresh
// fails) until the new one has fully arrived.
struct SharedTaskListStaging {
    std::vector<DatasetTaskSummary> summaries;
    std::unordered_map<int, size_t> summaryIndexByTaskId;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetEncodedMasks> masksByTaskId;
    std::unordered_map<int, ImageGroundTruth> groundTruthByTaskId;
};

struct SharedLabelStudioProjectData {
    std::vector<DatasetTaskSummary> summaries;
    std::unordered_map<int, size_t> summaryIndexByTaskId;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetEncodedMasks> masksByTaskId;
    // Each annotated task's ground truth (first annotation), as Benchmark's
    // Label Studio source evaluates against it.
    std::unordered_map<int, ImageGroundTruth> groundTruthByTaskId;

    LabelStudioProjectConfig projectConfig;
    std::string rectangleLabelsFromName;   // from_name of the first RectangleLabels tag, if any
    std::string brushLabelsFromName;       // from_name of the first BrushLabels tag, if any

    // Identity, not freshness: (baseUrl, activeProjectId, apiToken) as a
    // "|"-joined key. Changes only on a project switch or reconnect, never
    // on a plain refresh of the same project -- consumers that need to
    // tell "the same project just got refreshed" apart from "a different
    // project is now active" (e.g. to decide whether to clear a
    // window-local cache) compare against this, not against `version`.
    std::string lastFetchKey;
    // Bumped whenever summaries/the lookup maps change: once per page
    // during a first load (see streamingIntoLive), once when a refresh's
    // staged list is swapped in, and when a failed first load's partial
    // list is discarded for a retry. Consumers watching this know to
    // re-derive their own view from summaries/the lookup maps.
    uint64_t version = 0;
    // True only once a load has delivered the project's complete task
    // list. Stays false while a first load is still streaming, and after
    // one that failed or was cancelled partway -- so flows that must see
    // every task (Timestamp Search, Label Assistant) wait for it, while
    // the Dataset Browser can already browse `summaries`.
    bool loaded = false;
    bool loading = false;
    // Set on a failed fetch (config or task list). A failed or cancelled
    // refresh never touches the previous complete list -- it degrades to
    // "stale data plus an error", never a blank state. A failed or
    // cancelled first load keeps whatever pages arrived, and says how
    // many.
    std::string error;

    // Decided at the start of each load (beginSharedTaskListLoad): true
    // for a first load (nothing complete to preserve), so pages go
    // straight into summaries/the maps as they arrive; false for a
    // refresh, so pages go into `staging` instead.
    bool streamingIntoLive = false;
    SharedTaskListStaging staging;
    // Mirrors worker.progress() while `loading`.
    DatasetTaskListProgress loadProgress;

    // The import-date range the current list was loaded with (every load
    // uses it, including the automatic one on project select), and the
    // one the Connection tab's date pickers are editing -- they only take
    // effect via setSharedTaskImportDateRange ("Apply range").
    TaskImportDateRange appliedImportDateRange;
    TaskImportDateRange pendingImportDateRange;
    // Sum of DatasetTaskListPage::droppedOutOfRange for the current load.
    // Non-zero means Label Studio ignored the date filter and every task
    // is being downloaded and filtered locally (correct, but slow).
    int droppedOutOfRangeTasks = 0;

    DatasetTaskListWorker worker;
};

// Pure function: resets `state` for a new task-list load. Decides
// streamingIntoLive (= !loaded), and if streaming into live data, clears
// any partial list a previous failed first load left behind (bumping
// `version` if there was one). Always clears `staging`, `loadProgress`
// and `error`, and sets `loading = true`. Exposed separately from
// refreshSharedLabelStudioProjectData so the load transitions are
// unit-testable without a worker thread.
void beginSharedTaskListLoad(SharedLabelStudioProjectData& state);

// Pure function: appends one fetched page -- into summaries/the maps
// (bumping `version`) when streamingIntoLive, into `staging` otherwise.
// A task id already present in the destination is skipped, along with
// its boxes/masks: offset pagination can re-deliver a task when tasks are
// added or removed in Label Studio while a long load runs.
void applySharedTaskListPage(SharedLabelStudioProjectData& state, DatasetTaskListPage page);

// If `range` differs from appliedImportDateRange: stops any load in
// progress, makes `range` the applied range, and clears the list
// (summaries/the maps/staging, `loaded`/`loading` false, `version`
// bumped) so the next load -- which the caller starts via
// refreshSharedLabelStudioProjectData -- streams the new range from
// scratch rather than staging behind a list for a different range.
// Returns whether anything changed; a no-op for the same range.
bool setSharedTaskImportDateRange(SharedLabelStudioProjectData& state, const TaskImportDateRange& range);

// Pure function: applies the end of a load. Sets `loading = false`.
// Completed: a refresh's staging is swapped into live data (bumping
// `version`); `loaded = true`; `error` cleared. Failed/Cancelled: when
// streamingIntoLive, keeps the partial list, leaves `loaded` false and
// sets `error` to "Loaded N of M tasks, then: <error or "cancelled">"
// (" of M" omitted if the total is unknown, read from `loadProgress`);
// for a refresh, discards staging and sets `error`, leaving the previous
// list and `version` untouched.
void applySharedTaskListCompletion(SharedLabelStudioProjectData& state, const DatasetTaskListCompletion& completion);

// Pure function: applies a finished project-config fetch to `state`. On
// success, replaces `projectConfig` and re-derives
// rectangleLabelsFromName/brushLabelsFromName (first RectangleLabels/
// BrushLabels tag found, in document order -- "first one wins" if a
// project somehow has more than one of either), clearing `error`. On
// failure, leaves projectConfig/rectangleLabelsFromName/
// brushLabelsFromName untouched and sets `error` to
// "Project config: " + config.error.
void applySharedProjectConfigResult(SharedLabelStudioProjectData& state, const LabelStudioProjectConfig& config);

// Pure function: the "still loading" line every Label Studio tab shows
// while `state.loading` -- "Loading tasks... <fetched> / <total>", or
// just "Loading tasks... <fetched>" until Label Studio reports a total.
std::string describeSharedTaskListLoadProgress(const SharedLabelStudioProjectData& state);

// Called once per main-loop iteration while any Label Studio-consuming
// tab is open. The first time (session.baseUrl, session.activeProjectId,
// session.apiToken) is complete, or whenever that combination changes
// (project switch or reconnect -- tracked via `lastFetchKey`): stops any
// load still running for the old project, then clears
// summaries/the maps/staging/`loaded`/`projectConfig` (a project
// switch must not go on showing the OLD project's data or control tags
// while the new one loads -- unlike a same-project refresh failure,
// which preserves stale data instead, since there the prior data still
// legitimately describes the current project), fetches project config
// synchronously via fetchLabelStudioProjectConfigDetailed and applies it
// via applySharedProjectConfigResult, then (if that succeeded) starts the
// task-list fetch via refreshSharedLabelStudioProjectData. Also, every
// call, copies worker.progress() into `loadProgress` while loading and
// polls `worker`, applying each arrived page via applySharedTaskListPage
// and then any completion via applySharedTaskListCompletion.
void updateSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session);

// Starts a load (beginSharedTaskListLoad) and restarts `worker` against
// the current session's project -- called by
// any tab's "Refresh" button, and once automatically by
// updateSharedLabelStudioProjectData right after a successful config
// fetch. No-op if projectConfig.dataImageKey is still empty (config
// hasn't loaded yet).
void refreshSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session);

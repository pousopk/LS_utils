#pragma once

#include "manager/dataset_browser_tasklist_worker.hpp"
#include "manager/label_studio_client.hpp"
#include "manager/label_studio_dataset_browser.hpp"
#include "manager/label_studio_session.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// One Label Studio project's task list + config, fetched once and shared
// by every tab that needs it (Labeling, Label Assistant, Find by
// Timestamp, Dataset Browser) -- refreshing from any one of them updates
// what every other one sees, instead of each independently re-fetching
// the whole project. DatasetTaskSummary/DatasetTaskListWorker/
// DatasetTaskListConfig/DatasetTaskListResult (used below) predate this
// file and still carry their original "Dataset" naming from when only
// the Dataset Browser used them; they're reused as-is here rather than
// renamed, since renaming every call site across four tabs for a
// naming-only change isn't worth the diff.
struct SharedLabelStudioProjectData {
    nlohmann::json rawTasksJson;
    std::vector<DatasetTaskSummary> summaries;
    std::unordered_map<int, size_t> summaryIndexByTaskId;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetMasksToDraw> masksByTaskId;

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
    // Bumped on every successful task-list fetch (refresh or first load),
    // same project or not. Consumers watching this know to re-derive
    // their own view from rawTasksJson/summaries/the lookup maps.
    uint64_t version = 0;
    bool loaded = false;
    bool loading = false;
    // Set on a failed fetch (config or task list). Never clears
    // rawTasksJson/summaries/the maps/projectConfig/loaded/version from a
    // prior success -- a failed refresh degrades to "stale data plus an
    // error", never a blank state.
    std::string error;

    DatasetTaskListWorker worker;
};

// Pure function: applies a finished task-list fetch to `state`. On
// success, replaces rawTasksJson/summaries/the three lookup maps, clears
// `error`, sets `loaded = true`, and increments `version`. On failure,
// leaves rawTasksJson/summaries/the maps/`loaded`/`version` completely
// untouched and only sets `error`. Sets `loading = false` either way.
// Exposed separately from updateSharedLabelStudioProjectData so this
// transition logic is unit-testable without a worker thread.
void applySharedTaskListResult(SharedLabelStudioProjectData& state, const DatasetTaskListResult& result);

// Pure function: applies a finished project-config fetch to `state`. On
// success, replaces `projectConfig` and re-derives
// rectangleLabelsFromName/brushLabelsFromName (first RectangleLabels/
// BrushLabels tag found, in document order -- "first one wins" if a
// project somehow has more than one of either), clearing `error`. On
// failure, leaves projectConfig/rectangleLabelsFromName/
// brushLabelsFromName untouched and sets `error` to
// "Project config: " + config.error.
void applySharedProjectConfigResult(SharedLabelStudioProjectData& state, const LabelStudioProjectConfig& config);

// Called once per main-loop iteration while any Label Studio-consuming
// tab is open. The first time (session.baseUrl, session.activeProjectId,
// session.apiToken) is complete, or whenever that combination changes
// (project switch or reconnect -- tracked via `lastFetchKey`): clears
// rawTasksJson/summaries/the maps/`loaded`/`projectConfig` (a project
// switch must not go on showing the OLD project's data or control tags
// while the new one loads -- unlike a same-project refresh failure,
// which preserves stale data instead, since there the prior data still
// legitimately describes the current project), fetches project config
// synchronously via fetchLabelStudioProjectConfigDetailed and applies it
// via applySharedProjectConfigResult, then (if that succeeded) starts the
// task-list fetch via refreshSharedLabelStudioProjectData. Also polls
// `worker` for a finished result every call and applies it via
// applySharedTaskListResult. A no-op otherwise.
void updateSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session);

// Restarts `worker` against the current session's project -- called by
// any tab's "Refresh" button, and once automatically by
// updateSharedLabelStudioProjectData right after a successful config
// fetch. No-op if projectConfig.dataImageKey is still empty (config
// hasn't loaded yet).
void refreshSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session);

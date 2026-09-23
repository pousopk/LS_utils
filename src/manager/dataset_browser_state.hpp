#pragma once

#include "manager/dataset_browser_export_worker.hpp"
#include "manager/dataset_browser_tasklist_worker.hpp"
#include "manager/dataset_browser_thumbnail_worker.hpp"
#include "manager/dataset_thumbnail_lru.hpp"
#include "manager/label_studio_dataset_browser.hpp"
#include "manager/label_studio_session.hpp"

#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

enum class DatasetExportState { NotStarted, Running, Complete, Cancelled };

enum class DatasetOverlayMode { Annotations, Predictions };

struct DatasetThumbnailEntry {
    GLuint texture = 0;
    int textureWidth = 0;
    int textureHeight = 0;
    DatasetThumbnailStatus status = DatasetThumbnailStatus::Loading;
};

// Resident thumbnail textures, keyed by task id, plus their LRU order
// (front = least-recently-touched, back = most-recently-touched) --
// owned by the state layer, not the worker, since only the main thread
// may create/delete GL textures.
struct DatasetThumbnailCache {
    static constexpr size_t kResidentCap = 300;

    std::unordered_map<int, DatasetThumbnailEntry> entries;
    std::vector<int> lruOrder;

    // Moves (or inserts) `taskId` to the back of lruOrder -- call for
    // every id visible this frame, whether or not it already has a
    // texture.
    void touch(int taskId);

    // Deletes textures for whatever idsToEvict(lruOrder, currentlyVisible,
    // kResidentCap) returns, and removes them from entries/lruOrder.
    void evictIfNeeded(const std::vector<int>& currentlyVisible);

    // Deletes every resident texture and clears both containers -- used
    // when the connected project changes, since task ids from a
    // different project are meaningless here.
    void clear();
};

struct DatasetBrowserState {
    // Full task list for the active project, and the raw JSON it came
    // from (kept only so buildDatasetExportJson can filter it directly
    // on Export -- not re-fetched for that).
    std::vector<DatasetTaskSummary> summaries;
    nlohmann::json rawTasksJson;
    std::string taskListError;
    bool taskListLoaded = false;
    DatasetTaskListWorker taskListWorker;
    bool taskListLoading = false;

    // taskId -> O(1) lookups, rebuilt (via indexSummariesByTaskId/
    // buildBoxesByTaskId/buildMasksByTaskId) every time summaries/
    // rawTasksJson are, in refreshDatasetBrowserTaskList. Exist so the
    // grid doesn't re-scan the whole task list for every newly-visible
    // cell while scrolling -- it used to, via std::find_if over
    // summaries plus boxesToDrawForTask/masksToDrawForTask each
    // independently re-scanning rawTasksJson, and that's what made
    // scrolling visibly slow on any project with more than a couple
    // hundred tasks.
    std::unordered_map<int, size_t> summaryIndexByTaskId;
    std::unordered_map<int, DatasetBoxesToDraw> boxesByTaskId;
    std::unordered_map<int, DatasetMasksToDraw> masksByTaskId;

    DatasetFilterSpec filter;
    std::vector<int> matchingTaskIds;   // recomputed whenever summaries or filter changes

    // Auto-fetched from the project's labeling config, same
    // lastAutoFetchKey-gated pattern as label_assistant_state.cpp --
    // needed to know task.data's image-path field.
    std::string dataImageKey;
    // from_name of the project's first RectangleLabels/BrushLabels
    // control tag, if any -- passed to boxesToDrawForTask/
    // masksToDrawForTask so thumbnails/previews can draw annotation/
    // prediction boxes and masks. Empty if the project has no such tag,
    // in which case nothing of that kind is drawn (a classification-only
    // project has neither; a detection-only project has boxes but no
    // masks, etc.).
    std::string rectangleLabelsFromName;
    std::string brushLabelsFromName;
    std::string lastAutoFetchKey;

    DatasetThumbnailWorker thumbnailWorker;
    DatasetThumbnailCache thumbnailCache;
    std::optional<int> selectedTaskId;
    // Which set of boxes and masks gets baked into thumbnails --
    // annotations or predictions, not both (everything's baked into the
    // image at decode time and colored by class name, so showing both
    // sources at once would need coloring by source too, which is harder
    // to read than picking one set to look at). Defaults to Annotations.
    // Changing this doesn't retroactively redraw already-cached
    // thumbnails -- the caller (see dataset_browser_window.cpp) clears
    // thumbnailCache when it changes so everything re-fetches with the
    // new selection baked in.
    DatasetOverlayMode overlayMode = DatasetOverlayMode::Annotations;

    std::string exportDestinationFolder;
    bool exportFolderPickerOpen = false;
    std::string exportFolderPickerExplorerDir;
    std::string exportFolderPickerFilter;
    DatasetExportWorker exportWorker;
    DatasetExportState exportState = DatasetExportState::NotStarted;
    DatasetExportProgress lastExportProgress;
    std::string exportStatus;
};

// Starts state.taskListWorker to fetch the project's full task list and
// (on the worker thread) re-derive summaries/the O(1) lookup maps from
// it, returning immediately -- does not block. Call on window open and
// on an explicit "Refresh" click. Sets state.taskListLoading = true;
// updateDatasetBrowserState applies the result once the worker
// finishes. No-op (sets taskListError immediately, no worker started)
// if dataImageKey hasn't been auto-fetched yet.
void refreshDatasetBrowserTaskList(DatasetBrowserState& state, const LabelStudioSessionState& session);

// Re-runs filterDatasetTasks against the current summaries/filter and
// stores the result in state.matchingTaskIds. Call whenever a filter
// control changes.
void reapplyDatasetBrowserFilter(DatasetBrowserState& state);

// Starts state.exportWorker for every task in state.matchingTaskIds,
// writing into state.exportDestinationFolder (created first if it
// doesn't exist). No-op if matchingTaskIds is empty or a folder isn't
// set. Sets state.exportState = Running.
void startDatasetBrowserExport(DatasetBrowserState& state, const LabelStudioSessionState& session);

// Called once per frame while the Dataset Browser tab is open:
// auto-fetches dataImageKey (clearing the thumbnail cache and updating
// the thumbnail worker's connection if the project changed), polls
// taskListWorker and applies a finished result (rawTasksJson/summaries/
// the O(1) lookup maps, then reapplyDatasetBrowserFilter) or records its
// error, drains thumbnailWorker's decoded results and uploads them as
// textures into thumbnailCache, and polls exportWorker's progress/result.
void updateDatasetBrowserState(DatasetBrowserState& state, const LabelStudioSessionState& session);

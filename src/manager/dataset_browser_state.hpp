#pragma once

#include "manager/dataset_browser_export_worker.hpp"
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

struct DatasetThumbnailEntry {
    GLuint texture = 0;
    int textureWidth = 0;
    int textureHeight = 0;
    DatasetThumbnailStatus status = DatasetThumbnailStatus::Loading;
    // In this entry's own texture pixel space (i.e. consistent with
    // textureWidth/textureHeight) -- the window layer scales these once
    // more, to on-screen space, when drawing the overlay (see
    // dataset_browser_window.cpp, mirroring labeling_window.cpp's
    // drawBoxOverlay). Colored by colorForClassName, not by which list
    // they're in -- annotationBoxes/predictionBoxes only exist as
    // separate lists so showAnnotationBoxes/showPredictionBoxes can hide
    // either independently.
    std::vector<DraftDetectionBox> annotationBoxes;
    std::vector<DraftDetectionBox> predictionBoxes;
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

    DatasetFilterSpec filter;
    std::vector<int> matchingTaskIds;   // recomputed whenever summaries or filter changes

    // Auto-fetched from the project's labeling config, same
    // lastAutoFetchKey-gated pattern as label_assistant_state.cpp --
    // needed to know task.data's image-path field.
    std::string dataImageKey;
    // from_name of the project's first RectangleLabels control tag, if
    // any -- passed to boxesToDrawForTask so thumbnails/previews can draw
    // annotation/prediction boxes. Empty if the project has no such tag
    // (e.g. classification-only), in which case nothing is drawn.
    std::string rectangleLabelsFromName;
    std::string lastAutoFetchKey;

    DatasetThumbnailWorker thumbnailWorker;
    DatasetThumbnailCache thumbnailCache;
    std::optional<int> selectedTaskId;
    // Independent display toggles for the box overlay -- both default on.
    // Unlike `filter`, these don't change which tasks match; they only
    // control what the grid/detail-panel overlay draws.
    bool showAnnotationBoxes = true;
    bool showPredictionBoxes = true;

    std::string exportDestinationFolder;
    bool exportFolderPickerOpen = false;
    std::string exportFolderPickerExplorerDir;
    std::string exportFolderPickerFilter;
    DatasetExportWorker exportWorker;
    DatasetExportState exportState = DatasetExportState::NotStarted;
    DatasetExportProgress lastExportProgress;
    std::string exportStatus;
};

// Fetches the project's full task list (fetchAllLabelStudioTasksRaw) and
// re-derives summaries/matchingTaskIds from it. Call on window open and
// on an explicit "Refresh" click. Synchronous (a single blocking network
// call), matching this codebase's existing convention for connection/
// config-shaped fetches -- unlike downloading images, this is one JSON
// response, not a per-item loop, so it doesn't need a worker thread.
// No-op (sets taskListError) if dataImageKey hasn't been auto-fetched
// yet.
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
// the thumbnail worker's connection if the project changed), drains
// thumbnailWorker's decoded results and uploads them as textures into
// thumbnailCache, and polls exportWorker's progress/result.
void updateDatasetBrowserState(DatasetBrowserState& state, const LabelStudioSessionState& session);

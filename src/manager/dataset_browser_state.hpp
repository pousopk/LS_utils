#pragma once

#include "manager/dataset_browser_export_worker.hpp"
#include "manager/dataset_browser_thumbnail_worker.hpp"
#include "manager/dataset_thumbnail_lru.hpp"
#include "manager/label_studio_dataset_browser.hpp"
#include "manager/label_studio_project_data.hpp"
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
    DatasetFilterSpec filter;
    std::vector<int> matchingTaskIds;   // recomputed whenever the shared summaries or filter changes

    // Compared against SharedLabelStudioProjectData's own version/
    // lastFetchKey each frame (see updateDatasetBrowserState): `version`
    // changing means "re-derive matchingTaskIds", `lastFetchKey` changing
    // means "the project itself switched -- clear window-local caches
    // too" (thumbnailCache, in this case).
    uint64_t lastAppliedSharedVersion = 0;
    std::string lastSeenSharedFetchKey;

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

// Re-runs filterDatasetTasks against sharedData's current summaries and
// state.filter, storing the result in state.matchingTaskIds. Call
// whenever a filter control changes, or when sharedData.version changes
// (see updateDatasetBrowserState).
void reapplyDatasetBrowserFilter(DatasetBrowserState& state, const SharedLabelStudioProjectData& sharedData);

// Starts state.exportWorker for every task in state.matchingTaskIds,
// writing into state.exportDestinationFolder (created first if it
// doesn't exist), sourcing the raw JSON/summaries to export from
// sharedData. No-op if matchingTaskIds is empty or a folder isn't set.
// Sets state.exportState = Running.
void startDatasetBrowserExport(
    DatasetBrowserState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

// Called once per frame while the Dataset Browser tab is open. Compares
// sharedData.lastFetchKey against state.lastSeenSharedFetchKey: on a
// change (project switch or reconnect), clears thumbnailCache and points
// thumbnailWorker at the new connection. Compares sharedData.version
// against state.lastAppliedSharedVersion: on a change (any successful
// refresh), re-runs reapplyDatasetBrowserFilter. Also drains
// thumbnailWorker's decoded results into thumbnailCache as textures, and
// polls exportWorker's progress/result.
void updateDatasetBrowserState(
    DatasetBrowserState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

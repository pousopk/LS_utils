#include "manager/dataset_browser_state.hpp"

#include "ui_common/gl_texture.hpp"

#include <algorithm>
#include <filesystem>

void DatasetThumbnailCache::touch(int taskId) {
    lruOrder.erase(std::remove(lruOrder.begin(), lruOrder.end(), taskId), lruOrder.end());
    lruOrder.push_back(taskId);
}

void DatasetThumbnailCache::evictIfNeeded(const std::vector<int>& currentlyVisible) {
    const std::vector<int> toEvict = idsToEvict(lruOrder, currentlyVisible, kResidentCap);
    for (const int taskId : toEvict) {
        auto it = entries.find(taskId);
        if (it != entries.end()) {
            if (it->second.texture != 0) {
                glDeleteTextures(1, &it->second.texture);
            }
            entries.erase(it);
        }
        lruOrder.erase(std::remove(lruOrder.begin(), lruOrder.end(), taskId), lruOrder.end());
    }
}

void DatasetThumbnailCache::clear() {
    for (auto& [taskId, entry] : entries) {
        if (entry.texture != 0) {
            glDeleteTextures(1, &entry.texture);
        }
    }
    entries.clear();
    lruOrder.clear();
}

void reapplyDatasetBrowserFilter(DatasetBrowserState& state, const SharedLabelStudioProjectData& sharedData) {
    state.matchingTaskIds = filterDatasetTasks(sharedData.summaries, state.filter);
}

void startDatasetBrowserExport(
    DatasetBrowserState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
    if (state.matchingTaskIds.empty() || state.exportDestinationFolder.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(state.exportDestinationFolder, ec);

    DatasetExportConfig config;
    config.baseUrl = session.baseUrl;
    config.apiToken = session.apiToken;
    config.destinationFolder = state.exportDestinationFolder;
    config.projectId = session.activeProjectId;
    config.matchingTaskIds = state.matchingTaskIds;
    config.createdAtBounds = toCreatedAtBounds(sharedData.appliedImportDateRange);

    config.tasksToExport.reserve(state.matchingTaskIds.size());
    for (const int taskId : state.matchingTaskIds) {
        const auto summaryIndexIt = sharedData.summaryIndexByTaskId.find(taskId);
        if (summaryIndexIt != sharedData.summaryIndexByTaskId.end()) {
            config.tasksToExport.push_back(sharedData.summaries[summaryIndexIt->second]);
        }
    }

    state.exportStatus.clear();
    state.exportWorker.start(std::move(config));
    state.exportState = DatasetExportState::Running;
}

void updateDatasetBrowserState(
    DatasetBrowserState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
    if (sharedData.lastFetchKey != state.lastSeenSharedFetchKey) {
        state.lastSeenSharedFetchKey = sharedData.lastFetchKey;
        state.thumbnailCache.clear();
        state.filter.cls.className.clear();
        state.thumbnailWorker.setConnection(session.baseUrl, session.apiToken);
    }
    if (sharedData.version != state.lastAppliedSharedVersion) {
        state.lastAppliedSharedVersion = sharedData.version;
        state.availableClassNames = collectDatasetClassNames(sharedData.summaries);
        state.tasksPerDay = countTasksPerLocalDay(sharedData.summaries);
        reapplyDatasetBrowserFilter(state, sharedData);
    }

    std::vector<DatasetThumbnailResult> results;
    state.thumbnailWorker.drainResults(results, /*maxResults=*/16);
    for (auto& thumbnailResult : results) {
        DatasetThumbnailEntry entry;
        if (thumbnailResult.success) {
            glGenTextures(1, &entry.texture);
            uploadFrameToTexture(entry.texture, thumbnailResult.thumbnail, entry.textureWidth, entry.textureHeight);
            entry.status = DatasetThumbnailStatus::Loaded;
        } else {
            entry.status = DatasetThumbnailStatus::Failed;
        }
        state.thumbnailCache.entries[thumbnailResult.taskId] = entry;
        state.thumbnailCache.touch(thumbnailResult.taskId);
    }

    if (state.exportState == DatasetExportState::Running) {
        state.lastExportProgress = state.exportWorker.progress();

        DatasetExportResult exportResult;
        if (state.exportWorker.tryTakeResult(exportResult)) {
            state.exportState =
                exportResult.cancelled ? DatasetExportState::Cancelled : DatasetExportState::Complete;

            std::string status = "Downloaded " + std::to_string(exportResult.downloaded) + ", failed "
                + std::to_string(exportResult.downloadFailed);
            if (!exportResult.sampleErrors.empty()) {
                status += " (e.g. \"" + exportResult.sampleErrors.front() + "\")";
            }
            if (!exportResult.exportJsonWritten) {
                status += "; " + exportResult.exportJsonError;
            }
            state.exportStatus = status;
        }
    }
}

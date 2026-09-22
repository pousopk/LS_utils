#include "manager/dataset_browser_state.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_studio_client.hpp"

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

namespace {

std::string buildAutoFetchKey(const LabelStudioSessionState& session) {
    return session.baseUrl + "|" + std::to_string(session.activeProjectId) + "|" + session.apiToken;
}

void syncDatasetBrowserAutoFetch(DatasetBrowserState& state, const LabelStudioSessionState& session) {
    if (session.baseUrl.empty() || session.activeProjectId <= 0 || session.apiToken.empty()) {
        return;
    }
    const std::string key = buildAutoFetchKey(session);
    if (key == state.lastAutoFetchKey) {
        return;
    }
    state.lastAutoFetchKey = key;

    // Deliberately fetchLabelStudioProjectConfigDetailed, not
    // fetchLabelStudioLabelingConfig (the one label_assistant_state.cpp
    // uses): the latter needs an isDetection guess and errors out (leaving
    // dataImageKey empty too) if the project has no matching control tag
    // of that guessed type -- e.g. a classification-only project with
    // isDetection=true. fetchLabelStudioProjectConfigDetailed's
    // dataImageKey comes from the <Image> tag alone, independent of which
    // control tags exist, so it works regardless of task type.
    const LabelStudioProjectConfig config =
        fetchLabelStudioProjectConfigDetailed(session.baseUrl, session.activeProjectId, session.apiToken);
    if (!config.error.empty()) {
        state.taskListError = "Project config: " + config.error;
        return;
    }
    state.dataImageKey = config.dataImageKey;

    state.rectangleLabelsFromName.clear();
    state.brushLabelsFromName.clear();
    for (const auto& tag : config.controlTags) {
        if (tag.type == LabelStudioControlTagType::RectangleLabels && state.rectangleLabelsFromName.empty()) {
            state.rectangleLabelsFromName = tag.name;
        } else if (tag.type == LabelStudioControlTagType::BrushLabels && state.brushLabelsFromName.empty()) {
            state.brushLabelsFromName = tag.name;
        }
    }
    state.thumbnailCache.clear();
    state.thumbnailWorker.setConnection(session.baseUrl, session.apiToken);
}

} // namespace

void refreshDatasetBrowserTaskList(DatasetBrowserState& state, const LabelStudioSessionState& session) {
    if (state.dataImageKey.empty()) {
        state.taskListError = "Waiting on the project's data image key -- try again in a moment.";
        return;
    }

    nlohmann::json allTasks;
    std::string error;
    if (!fetchAllLabelStudioTasksRaw(session.baseUrl, session.activeProjectId, session.apiToken, allTasks, error)) {
        state.taskListError = error;
        return;
    }

    state.taskListError.clear();
    state.rawTasksJson = std::move(allTasks);
    state.summaries = summarizeDatasetTasks(state.rawTasksJson, state.dataImageKey);
    state.summaryIndexByTaskId = indexSummariesByTaskId(state.summaries);
    state.boxesByTaskId = buildBoxesByTaskId(state.rawTasksJson, state.rectangleLabelsFromName);
    state.masksByTaskId = buildMasksByTaskId(state.rawTasksJson, state.brushLabelsFromName);
    state.taskListLoaded = true;
    state.thumbnailCache.clear();
    reapplyDatasetBrowserFilter(state);
}

void reapplyDatasetBrowserFilter(DatasetBrowserState& state) {
    state.matchingTaskIds = filterDatasetTasks(state.summaries, state.filter);
}

void startDatasetBrowserExport(DatasetBrowserState& state, const LabelStudioSessionState& session) {
    if (state.matchingTaskIds.empty() || state.exportDestinationFolder.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(state.exportDestinationFolder, ec);

    DatasetExportConfig config;
    config.baseUrl = session.baseUrl;
    config.apiToken = session.apiToken;
    config.destinationFolder = state.exportDestinationFolder;
    config.exportJson = buildDatasetExportJson(state.rawTasksJson, state.matchingTaskIds);

    config.tasksToExport.reserve(state.matchingTaskIds.size());
    for (const auto& summary : state.summaries) {
        if (std::find(state.matchingTaskIds.begin(), state.matchingTaskIds.end(), summary.taskId)
            != state.matchingTaskIds.end()) {
            config.tasksToExport.push_back(summary);
        }
    }

    state.exportStatus.clear();
    state.exportWorker.start(std::move(config));
    state.exportState = DatasetExportState::Running;
}

void updateDatasetBrowserState(DatasetBrowserState& state, const LabelStudioSessionState& session) {
    syncDatasetBrowserAutoFetch(state, session);

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

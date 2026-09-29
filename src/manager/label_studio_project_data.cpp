#include "manager/label_studio_project_data.hpp"

namespace {

void clearLiveTaskData(SharedLabelStudioProjectData& state) {
    state.summaries.clear();
    state.summaryIndexByTaskId.clear();
    state.boxesByTaskId.clear();
    state.masksByTaskId.clear();
}

// Appends `page` into the given containers, skipping any task id already indexed.
void appendPage(
    DatasetTaskListPage& page, std::vector<DatasetTaskSummary>& summaries,
    std::unordered_map<int, size_t>& summaryIndexByTaskId,
    std::unordered_map<int, DatasetBoxesToDraw>& boxesByTaskId,
    std::unordered_map<int, DatasetEncodedMasks>& masksByTaskId) {
    for (auto& summary : page.summaries) {
        const int taskId = summary.taskId;
        if (summaryIndexByTaskId.count(taskId) != 0) {
            continue;
        }
        summaryIndexByTaskId[taskId] = summaries.size();
        summaries.push_back(std::move(summary));
        if (auto boxesIt = page.boxesByTaskId.find(taskId); boxesIt != page.boxesByTaskId.end()) {
            boxesByTaskId[taskId] = std::move(boxesIt->second);
        }
        if (auto masksIt = page.masksByTaskId.find(taskId); masksIt != page.masksByTaskId.end()) {
            masksByTaskId[taskId] = std::move(masksIt->second);
        }
    }
}

std::string describeStoppedLoad(const SharedLabelStudioProjectData& state, const std::string& reason) {
    std::string text = "Loaded " + std::to_string(state.summaries.size());
    if (state.loadProgress.totalTasks) {
        text += " of " + std::to_string(*state.loadProgress.totalTasks);
    }
    return text + " tasks, then: " + reason;
}

} // namespace

bool isValidTaskImportDateRange(const TaskImportDateRange& range) {
    return !(range.from && range.to && *range.to < *range.from);
}

TaskCreatedAtBounds toCreatedAtBounds(const TaskImportDateRange& range) {
    TaskCreatedAtBounds bounds;
    if (range.from) {
        bounds.fromInclusive = localMidnight(*range.from);
    }
    if (range.to) {
        bounds.toExclusive = localMidnight(nextDay(*range.to));
    }
    return bounds;
}

std::string describeTaskImportDateRange(const TaskImportDateRange& range) {
    if (range.from && range.to) {
        return "Tasks imported " + formatCalendarDate(*range.from) + " - " + formatCalendarDate(*range.to);
    }
    if (range.from) {
        return "Tasks imported from " + formatCalendarDate(*range.from);
    }
    if (range.to) {
        return "Tasks imported up to " + formatCalendarDate(*range.to);
    }
    return "";
}

bool setSharedTaskImportDateRange(SharedLabelStudioProjectData& state, const TaskImportDateRange& range) {
    if (range == state.appliedImportDateRange) {
        return false;
    }
    state.worker.stop();
    state.appliedImportDateRange = range;
    clearLiveTaskData(state);
    state.staging = SharedTaskListStaging{};
    state.loaded = false;
    state.loading = false;
    state.droppedOutOfRangeTasks = 0;
    state.version++;
    return true;
}

void beginSharedTaskListLoad(SharedLabelStudioProjectData& state) {
    state.streamingIntoLive = !state.loaded;
    if (state.streamingIntoLive && !state.summaries.empty()) {
        clearLiveTaskData(state);
        state.version++;   // consumers must drop views built from the discarded partial list
    }
    state.staging = SharedTaskListStaging{};
    state.loadProgress = DatasetTaskListProgress{};
    state.droppedOutOfRangeTasks = 0;
    state.error.clear();
    state.loading = true;
}

void applySharedTaskListPage(SharedLabelStudioProjectData& state, DatasetTaskListPage page) {
    state.droppedOutOfRangeTasks += page.droppedOutOfRange;
    if (state.streamingIntoLive) {
        appendPage(page, state.summaries, state.summaryIndexByTaskId, state.boxesByTaskId, state.masksByTaskId);
        state.version++;
    } else {
        appendPage(
            page, state.staging.summaries, state.staging.summaryIndexByTaskId, state.staging.boxesByTaskId,
            state.staging.masksByTaskId);
    }
}

void applySharedTaskListCompletion(SharedLabelStudioProjectData& state, const DatasetTaskListCompletion& completion) {
    state.loading = false;
    if (completion.outcome == DatasetTaskListOutcome::Completed) {
        if (!state.streamingIntoLive) {
            state.summaries = std::move(state.staging.summaries);
            state.summaryIndexByTaskId = std::move(state.staging.summaryIndexByTaskId);
            state.boxesByTaskId = std::move(state.staging.boxesByTaskId);
            state.masksByTaskId = std::move(state.staging.masksByTaskId);
            state.version++;
        }
        state.staging = SharedTaskListStaging{};
        state.error.clear();
        state.loaded = true;
        return;
    }

    const std::string reason = completion.outcome == DatasetTaskListOutcome::Cancelled ? "cancelled" : completion.error;
    if (state.streamingIntoLive) {
        state.error = describeStoppedLoad(state, reason);
    } else {
        state.error = "Refresh failed (showing previous list): " + reason;
    }
    state.staging = SharedTaskListStaging{};
}

std::string describeSharedTaskListLoadProgress(const SharedLabelStudioProjectData& state) {
    std::string text = "Loading tasks... " + std::to_string(state.loadProgress.fetchedTasks);
    if (state.loadProgress.totalTasks) {
        text += " / " + std::to_string(*state.loadProgress.totalTasks);
    }
    return text;
}

void applySharedProjectConfigResult(SharedLabelStudioProjectData& state, const LabelStudioProjectConfig& config) {
    if (!config.error.empty()) {
        state.error = "Project config: " + config.error;
        return;
    }
    state.error.clear();
    state.projectConfig = config;

    state.rectangleLabelsFromName.clear();
    state.brushLabelsFromName.clear();
    for (const auto& tag : config.controlTags) {
        if (tag.type == LabelStudioControlTagType::RectangleLabels && state.rectangleLabelsFromName.empty()) {
            state.rectangleLabelsFromName = tag.name;
        } else if (tag.type == LabelStudioControlTagType::BrushLabels && state.brushLabelsFromName.empty()) {
            state.brushLabelsFromName = tag.name;
        }
    }
}

namespace {
std::string buildSharedFetchKey(const LabelStudioSessionState& session) {
    return session.baseUrl + "|" + std::to_string(session.activeProjectId) + "|" + session.apiToken;
}
} // namespace

void updateSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session) {
    if (!session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty()) {
        const std::string key = buildSharedFetchKey(session);
        if (key != state.lastFetchKey) {
            state.lastFetchKey = key;
            state.worker.stop();
            state.loaded = false;
            state.loading = false;
            clearLiveTaskData(state);
            state.staging = SharedTaskListStaging{};
            // Reset projectConfig too, not just the task list -- a project
            // switch whose config fetch then fails must not go on showing
            // the PREVIOUS project's control tags/dataImageKey (that would
            // be silently wrong, not merely stale). applySharedProjectConfigResult's
            // "preserve on failure" behavior exists for a same-project
            // refresh failure, where the prior data legitimately still
            // describes the current project -- that assumption doesn't
            // hold across a project switch, so it's cleared here first.
            state.projectConfig = LabelStudioProjectConfig{};
            state.rectangleLabelsFromName.clear();
            state.brushLabelsFromName.clear();

            const LabelStudioProjectConfig config =
                fetchLabelStudioProjectConfigDetailed(session.baseUrl, session.activeProjectId, session.apiToken);
            applySharedProjectConfigResult(state, config);
            if (config.error.empty()) {
                refreshSharedLabelStudioProjectData(state, session);
            }
        }
    }

    if (state.loading) {
        state.loadProgress = state.worker.progress();
    }
    DatasetTaskListPoll poll;
    state.worker.poll(poll);
    for (auto& page : poll.pages) {
        applySharedTaskListPage(state, std::move(page));
    }
    if (poll.completion) {
        applySharedTaskListCompletion(state, *poll.completion);
    }
}

void refreshSharedLabelStudioProjectData(SharedLabelStudioProjectData& state, const LabelStudioSessionState& session) {
    if (state.projectConfig.dataImageKey.empty()) {
        return;
    }

    DatasetTaskListConfig config;
    config.baseUrl = session.baseUrl;
    config.projectId = session.activeProjectId;
    config.apiToken = session.apiToken;
    config.dataImageKey = state.projectConfig.dataImageKey;
    config.rectangleLabelsFromName = state.rectangleLabelsFromName;
    config.brushLabelsFromName = state.brushLabelsFromName;
    config.createdAtBounds = toCreatedAtBounds(state.appliedImportDateRange);

    beginSharedTaskListLoad(state);
    state.worker.start(std::move(config));
}

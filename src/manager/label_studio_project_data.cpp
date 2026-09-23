#include "manager/label_studio_project_data.hpp"

void applySharedTaskListResult(SharedLabelStudioProjectData& state, const DatasetTaskListResult& result) {
    state.loading = false;
    if (!result.success) {
        state.error = result.error;
        return;
    }
    state.error.clear();
    state.rawTasksJson = result.rawTasksJson;
    state.summaries = result.summaries;
    state.summaryIndexByTaskId = result.summaryIndexByTaskId;
    state.boxesByTaskId = result.boxesByTaskId;
    state.masksByTaskId = result.masksByTaskId;
    state.loaded = true;
    state.version++;
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
            state.loaded = false;
            state.rawTasksJson = nlohmann::json::array();
            state.summaries.clear();
            state.summaryIndexByTaskId.clear();
            state.boxesByTaskId.clear();
            state.masksByTaskId.clear();
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

    DatasetTaskListResult result;
    if (state.worker.tryTakeResult(result)) {
        applySharedTaskListResult(state, result);
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

    state.error.clear();
    state.worker.start(std::move(config));
    state.loading = true;
}

#include "manager/timestamp_search_state.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>

namespace {

// Fixed, hidden scratch location for downloaded thumbnails -- cleared and
// recreated at the start of every search, so no leftover files
// accumulate across searches and nothing needs cleaning up on window
// close (same convention as label_assistant_state.cpp's scratch folder).
std::string timestampSearchScratchFolder() {
    return (std::filesystem::temp_directory_path() / "vision_app_timestamp_search_download").string();
}

void clearResultTextures(TimestampSearchState& state) {
    for (auto& group : state.resultGroups) {
        for (auto& candidateView : group.candidates) {
            if (candidateView.texture != 0) {
                glDeleteTextures(1, &candidateView.texture);
            }
        }
    }
}

std::string candidateImageFilePath(const std::string& scratchFolder, const TimestampMatchCandidate& candidate) {
    const std::string extension = std::filesystem::path(candidate.imagePath).extension().string();
    return (std::filesystem::path(scratchFolder) / (std::to_string(candidate.taskId) + extension)).string();
}

void buildResultGroups(TimestampSearchState& state, const TimestampSearchRunResult& runResult) {
    state.resultGroups.clear();
    for (size_t i = 0; i < state.lastPerQueryMatches.size() && i < state.lastSearchedEntries.size(); ++i) {
        TimestampSearchResultGroup group;
        group.entry = state.lastSearchedEntries[i];

        for (const auto& candidate : state.lastPerQueryMatches[i]) {
            TimestampSearchCandidateView view;
            view.candidate = candidate;

            const cv::Mat frame = cv::imread(candidateImageFilePath(runResult.scratchFolderPath, candidate));
            if (!frame.empty()) {
                glGenTextures(1, &view.texture);
                uploadFrameToTexture(view.texture, frame, view.textureWidth, view.textureHeight);
            }
            group.candidates.push_back(view);
        }
        state.resultGroups.push_back(std::move(group));
    }
}

} // namespace

void startTimestampSearch(
    TimestampSearchState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
    std::vector<TimestampMatchQuery> queries;
    std::vector<TimestampSearchEntry> searchedEntries;
    for (const auto& entry : state.entries) {
        const std::optional<std::time_t> parsed = parseTypedLocalTimestamp(entry.rawText);
        if (!parsed) {
            continue;
        }
        TimestampMatchQuery query;
        query.timestamp = *parsed;
        query.toleranceSeconds = static_cast<long long>(state.toleranceMinutes) * 60;
        queries.push_back(query);
        searchedEntries.push_back(entry);
    }

    clearResultTextures(state);
    state.resultGroups.clear();
    state.resultError.clear();
    state.lastSearchedEntries = std::move(searchedEntries);

    state.lastPerQueryMatches =
        matchTasksToTimestamps(sharedData.rawTasksJson, sharedData.projectConfig.dataImageKey, queries);
    const std::vector<TimestampMatchCandidate> candidates = dedupTimestampMatchCandidates(state.lastPerQueryMatches);

    const std::string scratchFolder = timestampSearchScratchFolder();
    std::error_code ec;
    std::filesystem::remove_all(scratchFolder, ec);
    std::filesystem::create_directories(scratchFolder, ec);

    TimestampSearchRunConfig config;
    config.labelStudioBaseUrl = session.baseUrl;
    config.labelStudioApiToken = session.apiToken;
    config.candidates = candidates;
    config.scratchFolderPath = scratchFolder;

    state.worker.start(std::move(config));
    state.runState = TimestampSearchRunState::Running;
}

void updateTimestampSearchState(TimestampSearchState& state, const LabelStudioSessionState& session) {
    if (state.runState == TimestampSearchRunState::Running) {
        state.lastProgress = state.worker.progress();

        TimestampSearchRunResult runResult;
        if (state.worker.tryTakeResult(runResult)) {
            buildResultGroups(state, runResult);
            state.runState = TimestampSearchRunState::Complete;
        }
    }
}

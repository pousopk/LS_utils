#pragma once

#include "manager/timestamp_search_worker.hpp"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>

// One row the user typed in, before it's known whether it parses.
struct TimestampSearchEntry {
    std::string rawText;   // e.g. "2026/09/10 12:00:00" -- UTC, "YYYY/MM/DD HH:MM:SS"
};

struct TimestampSearchCandidateView {
    TimestampMatchCandidate candidate;
    GLuint texture = 0;        // 0 if the image failed to download/decode
    int textureWidth = 0;
    int textureHeight = 0;
};

struct TimestampSearchResultGroup {
    TimestampSearchEntry entry;                             // the typed entry this group is for
    std::vector<TimestampSearchCandidateView> candidates;    // sorted by closeness (from matchTasksToTimestamps)
};

enum class TimestampSearchRunState {
    NotStarted,
    Running,
    Complete,
};

struct TimestampSearchState {
    std::string labelStudioBaseUrl;
    int labelStudioProjectId = 0;
    std::string labelStudioApiToken;
    std::string labelStudioDataImageKey;   // auto-fetched, see updateTimestampSearchState
    std::string lastAutoFetchKey;
    std::string labelStudioAutoFetchStatus;

    int toleranceMinutes = 2;

    std::vector<TimestampSearchEntry> entries;
    std::string newEntryText;

    // The entries actually sent to the worker for the run currently
    // shown in resultGroups -- entries that failed to parse at Search
    // time are excluded, so this can be a subset of `entries`.
    std::vector<TimestampSearchEntry> lastSearchedEntries;

    TimestampSearchWorker worker;
    TimestampSearchRunState runState = TimestampSearchRunState::NotStarted;
    TimestampSearchProgress lastProgress;
    std::vector<TimestampSearchResultGroup> resultGroups;
    std::string resultError;
};

// Builds a TimestampSearchRunConfig from state.entries (skipping any that
// fail to parse -- they stay in state.entries for the user to fix, just
// excluded from this run) and state.toleranceMinutes, clears any previous
// result and its GL textures, (re)creates the hidden scratch download
// folder, and starts state.worker. Sets state.runState = Running.
void startTimestampSearch(TimestampSearchState& state);

// Called once per main-loop iteration while the window is open: polls
// worker.progress()/tryTakeResult(); on completion, builds resultGroups
// (loading one GL texture per candidate via cv::imread + uploadFrameToTexture)
// or sets resultError. Also lazily auto-fetches labelStudioDataImageKey
// from the Label Studio project once the connection fields are all set,
// via fetchLabelStudioLabelingConfig, re-fetching only when
// (baseUrl, projectId, apiToken) actually changes.
void updateTimestampSearchState(TimestampSearchState& state);

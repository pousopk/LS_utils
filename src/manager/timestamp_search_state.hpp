#pragma once

#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"
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
    int toleranceMinutes = 2;

    std::vector<TimestampSearchEntry> entries;
    std::string newEntryText;

    // The entries actually matched for the run currently shown in
    // resultGroups -- entries that failed to parse at Search time are
    // excluded, so this can be a subset of `entries`.
    std::vector<TimestampSearchEntry> lastSearchedEntries;
    // Matching happens synchronously on the main thread in
    // startTimestampSearch (against the shared project data, not a fresh
    // fetch); stored here so buildResultGroups can pair each candidate
    // back up with its query/entry once the worker's download phase
    // finishes -- the worker's own result no longer carries this.
    std::vector<std::vector<TimestampMatchCandidate>> lastPerQueryMatches;

    TimestampSearchWorker worker;
    TimestampSearchRunState runState = TimestampSearchRunState::NotStarted;
    TimestampSearchProgress lastProgress;
    std::vector<TimestampSearchResultGroup> resultGroups;
    std::string resultError;
};

// Matches state.entries (skipping any that fail to parse -- they stay in
// state.entries for the user to fix, just excluded from this run) against
// sharedData's already-loaded task list via matchTasksToTimestamps
// (synchronous, no network call), storing the result in
// state.lastPerQueryMatches. Clears any previous result and its GL
// textures, (re)creates the hidden scratch download folder, deduplicates
// the matches (dedupTimestampMatchCandidates) and starts state.worker to
// download just those images. Sets state.runState = Running.
void startTimestampSearch(
    TimestampSearchState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData);

// Called once per main-loop iteration while the tab is open: polls
// worker.progress()/tryTakeResult(); on completion, builds resultGroups
// from state.lastPerQueryMatches (loading one GL texture per candidate via
// cv::imread + uploadFrameToTexture).
void updateTimestampSearchState(TimestampSearchState& state, const LabelStudioSessionState& session);

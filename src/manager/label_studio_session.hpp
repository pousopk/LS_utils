#pragma once

#include "manager/label_studio_client.hpp"

#include <string>
#include <vector>

enum class LabelStudioSessionStatus {
    Disconnected,
    Connecting,
    Connected,
    Error,
};

// Shared, in-memory-only Label Studio connection: one base URL + API
// token, the project list fetched from it, and one globally active
// project -- consumed by every window that talks to Label Studio
// (Labeling, Label Assistant, Timestamp Search, Model Evaluation/Batch)
// in place of each window's own former baseUrl/projectId/apiToken
// fields. Not persisted to disk; resets to Disconnected on app restart.
struct LabelStudioSessionState {
    std::string baseUrl;
    std::string apiToken;
    std::vector<LabelStudioProjectSummary> projects;
    int activeProjectId = 0;
    std::string activeProjectTitle;
    LabelStudioSessionStatus status = LabelStudioSessionStatus::Disconnected;
    std::string error;   // set when status == Error
};

// Pure function: applies a project-list fetch result to `session`. On
// success, populates `projects`, sets status to Connected, clears
// `error`, and resets any previously active project (its id may not
// exist in the new list, e.g. after reconnecting with different
// credentials) -- the user re-picks from the panel. On failure, clears
// `projects` and the active project, sets status to Error with `error`
// set. Exposed separately from connectLabelStudioSession so this
// transition logic is unit-testable without a network call.
void applyProjectListResult(LabelStudioSessionState& session, const LabelStudioProjectListResult& result);

// Fetches the project list for (session.baseUrl, session.apiToken) via
// fetchLabelStudioProjects and applies it via applyProjectListResult.
// Synchronous (blocking network call), called once from the Label
// Studio panel's Connect button -- matches this codebase's existing
// convention of blocking one-shot network calls for connection/config
// fetches, rather than adding a worker thread for a single quick GET.
void connectLabelStudioSession(LabelStudioSessionState& session);

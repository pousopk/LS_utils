#include "manager/label_studio_session.hpp"

void applyProjectListResult(LabelStudioSessionState& session, const LabelStudioProjectListResult& result) {
    if (!result.error.empty()) {
        session.status = LabelStudioSessionStatus::Error;
        session.error = result.error;
        session.projects.clear();
        session.activeProjectId = 0;
        session.activeProjectTitle.clear();
        return;
    }
    session.status = LabelStudioSessionStatus::Connected;
    session.error.clear();
    session.projects = result.projects;
    session.activeProjectId = 0;
    session.activeProjectTitle.clear();
}

void connectLabelStudioSession(LabelStudioSessionState& session) {
    session.status = LabelStudioSessionStatus::Connecting;
    const LabelStudioProjectListResult result = fetchLabelStudioProjects(session.baseUrl, session.apiToken);
    applyProjectListResult(session, result);
}

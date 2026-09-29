#pragma once

#include "manager/label_assistant_state.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>
#include <string>

using LabelTaskCallback = std::function<void(int taskId)>;

// Content for the Label Assistant tab of the merged Label Studio window
// (no Begin/End of its own).
void drawLabelAssistantTabContent(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData,
    const LabelTaskCallback& onLabelTask, const std::function<void()>& onOpenLabelStudioWindow);

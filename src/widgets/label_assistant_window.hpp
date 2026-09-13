#pragma once

#include "manager/label_assistant_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>
#include <string>

using LabelTaskCallback = std::function<void(const std::string& baseUrl, int projectId, const std::string& apiToken, int taskId)>;

void drawLabelAssistantWindow(
    bool* show, LabelAssistantState& state, const LabelStudioSessionState& session,
    const LabelTaskCallback& onLabelTask, const std::function<void()>& onOpenLabelStudioWindow);

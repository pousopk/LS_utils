#pragma once

#include "manager/label_studio_session.hpp"
#include "manager/model_evaluation_state.hpp"

#include <imgui.h>

#include <functional>

// The Model Evaluation tab of the main window. Must be called inside a
// BeginTabBar; draws nothing while *show is false, and the tab's close
// button clears it.
void drawModelEvaluationTab(
    bool* show, ImGuiTabItemFlags flags, ModelEvaluationState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow);

#pragma once

#include "manager/label_studio_session.hpp"
#include "manager/model_evaluation_state.hpp"

#include <functional>

void drawModelEvaluationWindow(
    bool* show, ModelEvaluationState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow);

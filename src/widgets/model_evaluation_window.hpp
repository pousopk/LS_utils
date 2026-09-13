#pragma once

#include "manager/app_runtime.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/model_evaluation_state.hpp"

#include <functional>
#include <vector>

void drawModelEvaluationWindow(
    bool* show, ModelEvaluationState& state, std::vector<CameraSession>& sessions,
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow);

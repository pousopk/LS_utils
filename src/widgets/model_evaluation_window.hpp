#pragma once

#include "manager/app_runtime.hpp"
#include "manager/model_evaluation_state.hpp"

#include <vector>

void drawModelEvaluationWindow(bool* show, ModelEvaluationState& state, std::vector<CameraSession>& sessions);

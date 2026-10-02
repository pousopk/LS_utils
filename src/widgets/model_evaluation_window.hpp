#pragma once

#include "manager/label_studio_session.hpp"
#include "manager/model_evaluation_state.hpp"

#include <functional>

// Content for the Benchmark tab of the Models tab (no Begin/End of its own):
// one or two ONNX models evaluated over a batch of images, with metrics and
// a per-image drill-down.
void drawBenchmarkTabContent(
    ModelEvaluationState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow);

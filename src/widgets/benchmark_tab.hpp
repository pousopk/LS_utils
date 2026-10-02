#pragma once

#include "manager/label_studio_session.hpp"
#include "manager/benchmark_state.hpp"

#include <functional>

// Content for the Benchmark tab of the Models tab (no Begin/End of its own):
// one or two ONNX models evaluated over a batch of images, with metrics and
// a per-image drill-down.
void drawBenchmarkTabContent(
    BenchmarkState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData,
    const std::function<void()>& onOpenLabelStudioWindow);

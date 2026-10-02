#pragma once

#include "manager/onnx_metadata.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum class DetectedTaskMode {
    Unknown,
    Detection,
    ObbDetection,
    Classification,
    Anomaly,
};

struct ResolvedModelTask {
    DetectedTaskMode mode = DetectedTaskMode::Unknown;
    std::string error;  // set iff mode == Unknown
};

// Pure: decides what kind of model this is. A recognized `task` metadata
// value (Ultralytics "detect"/"obb"/"classify", or "vad" for anomaly)
// wins; "segment"/"pose" are rejected as unsupported. Otherwise falls back
// to `outputShape`, the dummy forward pass's first output (empty if that
// pass couldn't run): [1, C, N] is a detector -- axis-aligned when
// C == 4 + classCount, OBB when C == 5 + classCount, anything else is
// rejected -- [1, K] a classifier, and [1, 1, inputHeight, inputWidth] an
// anomaly map.
ResolvedModelTask resolveModelTask(
    const std::string& taskHint, const std::vector<int64_t>& outputShape, int inputWidth, int inputHeight,
    size_t classCount);

struct ModelAutoDetectResult {
    DetectedTaskMode mode = DetectedTaskMode::Unknown;
    int inputWidth = 0;
    int inputHeight = 0;
    std::vector<std::string> classNames;
    OnnxPreprocessingHints hints;
    AnomalyScoreHints anomalyHints;  // only meaningful when mode == Anomaly
    std::string error;               // set iff the model can't be used
};

// Reads everything needed to load an ONNX model from the file itself:
// declared input shape, preprocessing hints, class names, anomaly score
// calibration, and the task (see resolveModelTask -- the dummy forward
// pass only runs when the `task` metadata doesn't decide it). Fails
// (sets `error`, mode stays Unknown) on a dynamic input shape, an
// undeterminable task, or a detection/classification model with no
// embedded class names.
ModelAutoDetectResult autoDetectModel(const std::string& onnxPath);

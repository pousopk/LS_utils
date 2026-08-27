#pragma once

#include "manager/onnx_metadata.hpp"

#include <string>
#include <vector>

enum class DetectedTaskMode {
    Unknown,
    Detection,
    Classification,
    Anomaly,
};

struct ModelAutoDetectResult {
    bool shapeDetected = false;
    int inputWidth = 0;
    int inputHeight = 0;
    DetectedTaskMode suggestedMode = DetectedTaskMode::Unknown;
    std::vector<std::string> classNames;
    OnnxPreprocessingHints hints;
    AnomalyScoreHints anomalyHints;  // only meaningful when suggestedMode == Anomaly
    std::string error;
};

// Reads an ONNX model's declared input shape, preprocessing hints, and
// embedded class names (via onnx_metadata), then -- if a shape was found
// -- runs one cheap dummy forward pass at that shape to guess whether
// it's a detector or a classifier from the output tensor's rank. A failed
// shape read sets `error`; a failed/ambiguous mode guess just leaves
// suggestedMode at Unknown (not an error -- shape detection alone is
// still useful).
ModelAutoDetectResult autoDetectModel(const std::string& onnxPath);

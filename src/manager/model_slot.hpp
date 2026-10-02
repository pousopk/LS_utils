#pragma once

#include "manager/anomaly_inference.hpp"
#include "manager/classification_inference.hpp"
#include "manager/model_task.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// One loaded model plus its run-time thresholds -- Benchmark's Model A/B and
// Label Assistant's model. Everything describing the model itself (task,
// input size, class names, preprocessing, anomaly calibration) is read from
// the ONNX file on load; only the run-time thresholds are user-tunable.
struct ModelSlotConfig {
    std::string onnxPath;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    float anomalyThreshold = 0.5f;  // overwritten by the model's auto_threshold metadata on load, if any

    // Set on successful load; exactly one of the model pointers is non-null
    // while loaded, matching `task`.
    ModelTask task = ModelTask::Detection;
    std::shared_ptr<YoloModel> detectionModel;
    std::shared_ptr<ClassificationModel> classificationModel;
    std::shared_ptr<AnomalyModel> anomalyModel;
    std::string summary;  // e.g. "Detection (OBB) | 640x640 | 3 classes | GPU"
    std::string loadError;
};

bool isModelSlotLoaded(const ModelSlotConfig& slot);

// (Re)loads slot's model from onnxPath, reading everything it needs from
// the file via autoDetectModel and refusing (loadError set, nothing
// loaded) if that fails. Clears any previous model/summary/error first.
// No-op if onnxPath is empty.
void loadModelSlot(ModelSlotConfig& slot);

// Draws boxes + labels for `detections` onto a clone of `frame`. Box/label
// size scales up with the frame's own resolution relative to an 800px-wide
// baseline (never down), and every label is drawn on a solid background so
// it stays readable regardless of the photo's own colors.
cv::Mat annotateDetections(const cv::Mat& frame, const std::vector<Detection>& detections);

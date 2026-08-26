#pragma once

#include "manager/onnx_metadata.hpp"
#include "manager/onnx_runtime_env.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

struct AnomalyScore {
    float rawScore = 0.0f;
    float score = 0.0f;
    bool isAnomalous = false;
};

// Computes rawScore (the max value in `heatmap`, a CV_32F single-channel
// Mat), normalizes it into [scoreMin,scoreMax] clamped to [0,1] (if
// scoreMax <= scoreMin, treats rawScore as already normalized and just
// clamps it directly, avoiding a divide-by-zero), and thresholds it.
// Pure function -- no ONNX/inference dependency. An empty `heatmap`
// returns a default-constructed AnomalyScore (rawScore 0, not
// anomalous).
AnomalyScore scoreAnomalyHeatmap(const cv::Mat& heatmap, float scoreMin, float scoreMax, float threshold);

struct AnomalyResult {
    cv::Mat heatmap;         // CV_32F, single channel, original-frame resolution
    float rawScore = 0.0f;   // max over the raw (pre-crop) model output
    float score = 0.0f;      // rawScore normalized to [scoreMin,scoreMax], clamped to [0,1]
    bool isAnomalous = false;
};

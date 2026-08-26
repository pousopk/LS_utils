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

class AnomalyModel {
public:
    // `scoreMin`/`scoreMax` calibrate the normalized `score` -- pass
    // AnomalyScoreHints's fields (auto-detected or manually entered).
    // Threshold is intentionally NOT taken here: like YoloModel's
    // confThreshold/nmsThreshold, it's a call-time infer() parameter
    // instead, so a run can sweep/adjust it without reloading the
    // model. Construction never throws; on failure isValid() is false
    // and errorOut holds a human-readable reason.
    AnomalyModel(
        const std::string& onnxPath,
        int inputWidth,
        int inputHeight,
        const OnnxPreprocessingHints& hints,
        float scoreMin,
        float scoreMax,
        std::string& errorOut);

    bool isValid() const { return valid_; }

    // True if this model's session is using the CUDA execution provider
    // (only meaningful once isValid() is true).
    bool isGpuActive() const { return gpuActive_; }

    // Runs letterbox -> forward pass -> reshape to a heatmap -> score
    // -> un-letterbox on `frame`, setting isAnomalous = (score >=
    // threshold). Returns a default-constructed AnomalyResult (empty
    // heatmap, score 0, isAnomalous false) if the model isn't valid or
    // `frame` is empty.
    AnomalyResult infer(const cv::Mat& frame, float threshold);

private:
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    int inputWidth_ = 1024;
    int inputHeight_ = 1024;
    OnnxPreprocessingHints hints_;
    float scoreMin_ = 0.0f;
    float scoreMax_ = 1.0f;
    bool gpuActive_ = false;
    bool valid_ = false;
};

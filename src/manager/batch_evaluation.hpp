#pragma once

#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <string>
#include <vector>

struct TimingStats {
    double meanMs = 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
    int count = 0;
};

// Computes mean, median (50th percentile), and 95th percentile (linear
// interpolation between ranks) over `inferenceMsSamples`. count=0 (all
// fields left at 0) if empty.
TimingStats computeTimingStats(std::vector<double> inferenceMsSamples);

struct BatchImageResult {
    std::string imageFilename;
    std::vector<Detection> detections;              // Detection mode
    std::vector<GroundTruthBox> groundTruthBoxes;    // Detection mode, matched ground truth (empty if none)
    std::vector<ClassPrediction> predictions;        // Classification mode
    std::string groundTruthLabel;                    // Classification mode, matched ground truth (empty if none)
    double inferenceMs = 0.0;
    bool hasGroundTruth = false;
};

struct BatchEvaluationResult {
    std::vector<BatchImageResult> images;
    TimingStats timing;
    int imagesFound = 0;
    int imagesWithGroundTruth = 0;
    std::string error;  // hard failure: folder doesn't exist / no recognized images
};

// Scans `imageFolderPath` for recognized image files, cross-references
// each by basename against `groundTruth` (pass nullptr for none -- images
// are still evaluated, just with hasGroundTruth=false), and calls `infer`
// once per image. Synchronous -- run this off the UI thread if needed.
BatchEvaluationResult runDetectionBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<Detection>(const cv::Mat&)>& infer);

BatchEvaluationResult runClassificationBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& infer);

// Convenience conversions from a BatchEvaluationResult's per-image data
// into the metrics functions' input shape (only images with ground truth
// contribute).
std::vector<DetectionEvaluationItem> toDetectionEvaluationItems(const BatchEvaluationResult& result);
std::vector<ClassificationEvaluationItem> toClassificationEvaluationItems(const BatchEvaluationResult& result);

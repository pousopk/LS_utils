#pragma once

#include "manager/anomaly_inference.hpp"
#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
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

// Lightweight, persisted per-image anomaly result -- no heatmap (a
// 1024x1024 float heatmap is 4MB; keeping one per image for a whole
// batch run would balloon memory). The full heatmap is recomputed on
// demand for preview only.
struct BatchAnomalyResult {
    float rawScore = 0.0f;
    float score = 0.0f;
    bool isAnomalous = false;
};

struct BatchImageResult {
    std::string imageFilename;
    std::vector<Detection> detections;              // Detection mode
    std::vector<GroundTruthBox> groundTruthBoxes;    // Detection mode, matched ground truth (empty if none)
    std::vector<ClassPrediction> predictions;        // Classification mode
    std::string groundTruthLabel;                    // Classification mode, matched ground truth (empty if none)
    std::optional<BatchAnomalyResult> anomalyResult;  // Anomaly mode
    double inferenceMs = 0.0;
    bool hasGroundTruth = false;
};

struct BatchEvaluationResult {
    std::vector<BatchImageResult> images;
    TimingStats timing;
    int imagesFound = 0;
    int imagesWithGroundTruth = 0;
    int totalFilesInFolder = 0;  // count before sampling was applied (== imagesFound when sampling is off)
    std::string error;  // hard failure: folder doesn't exist / no recognized images
};

// Shuffles `files` (Fisher-Yates via `seed`) and keeps only the first
// `sampleSize`, re-sorted back to their original (alphabetical) order for
// a stable, readable image list. A no-op returning `files` unchanged if
// `sampleSize <= 0` or `sampleSize >= files.size()`. Pure and
// deterministic for a given `seed` -- callers wanting a different sample
// each run should draw `seed` from a real entropy source (e.g.
// `std::random_device`) themselves.
std::vector<std::filesystem::path> sampleImageFiles(
    std::vector<std::filesystem::path> files, int sampleSize, unsigned seed);

// Scans `imageFolderPath` for recognized image files, optionally samples
// down to `sampleSize` of them (0 = evaluate all -- see `sampleImageFiles`),
// cross-references each by basename against `groundTruth` (pass nullptr
// for none -- images are still evaluated, just with hasGroundTruth=false),
// and calls `infer` once per image. Synchronous -- run this off the UI
// thread if needed. `onProgress` (if non-null) is called once per
// processed image with (completed, total) -- `total` reflects the sampled
// count, not the full folder. `cancelRequested` (if non-null and observed
// true) stops the loop early, returning whatever was completed so far with
// `error` left empty (a cancelled run is not a hard failure). Callers
// comparing two models against the same run must pass the same `sampleSeed`
// to both calls, or they'll be evaluated on different random subsets.
BatchEvaluationResult runDetectionBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<Detection>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr,
    int sampleSize = 0,
    unsigned sampleSeed = 0);

BatchEvaluationResult runClassificationBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr,
    int sampleSize = 0,
    unsigned sampleSeed = 0);

// groundTruth is accepted for signature symmetry with the other two
// run*BatchEvaluation functions (and to avoid a signature change when
// sub-project 2 adds anomaly ground truth) but is unused this
// sub-project -- Anomaly mode has no ground truth ingestion yet.
BatchEvaluationResult runAnomalyBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<AnomalyResult(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr,
    int sampleSize = 0,
    unsigned sampleSeed = 0);

// Convenience conversions from a BatchEvaluationResult's per-image data
// into the metrics functions' input shape (only images with ground truth
// contribute).
std::vector<DetectionEvaluationItem> toDetectionEvaluationItems(const BatchEvaluationResult& result);
std::vector<ClassificationEvaluationItem> toClassificationEvaluationItems(const BatchEvaluationResult& result);

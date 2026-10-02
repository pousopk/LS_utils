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
struct BenchmarkAnomalyResult {
    float rawScore = 0.0f;
    float score = 0.0f;
    bool isAnomalous = false;
};

struct BenchmarkImageResult {
    std::string imageFilename;
    std::vector<Detection> detections;              // Detection mode
    std::vector<GroundTruthBox> groundTruthBoxes;    // Detection mode, matched ground truth (empty if none)
    std::vector<ClassPrediction> predictions;        // Classification mode
    std::string groundTruthLabel;                    // Classification mode, matched ground truth (empty if none)
    std::optional<BenchmarkAnomalyResult> anomalyResult;  // Anomaly mode
    double inferenceMs = 0.0;
    bool hasGroundTruth = false;
};

struct BenchmarkResult {
    std::vector<BenchmarkImageResult> images;
    TimingStats timing;
    int imagesFound = 0;
    int imagesWithGroundTruth = 0;
    // Images available before sampling (folder files, or labeled tasks);
    // == imagesFound when sampling is off. Set by the caller, not run*Benchmark.
    int totalAvailableImages = 0;
    std::string error;  // hard failure: no images to evaluate
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

// Recognized image files directly inside `folderPath`, sorted by path.
// Empty if the folder doesn't exist or holds no images.
std::vector<std::filesystem::path> listImageFiles(const std::string& folderPath);

// Evaluates each of `files` in order (unreadable files are skipped),
// cross-referencing each by its filename against `groundTruth` (pass
// nullptr for none -- images are still evaluated, just with
// hasGroundTruth=false), and calls `infer` once per image. Synchronous --
// run this off the UI thread. `onProgress` (if non-null) is called once
// per processed image with (completed, files.size()). `cancelRequested`
// (if non-null and observed true) stops the loop early, returning
// whatever was completed so far with `error` left empty (a cancelled run
// is not a hard failure). Choosing/sampling `files` is the caller's job;
// two models being compared must be given the same list.
BenchmarkResult runDetectionBenchmark(
    const std::vector<std::filesystem::path>& files,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<Detection>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

BenchmarkResult runClassificationBenchmark(
    const std::vector<std::filesystem::path>& files,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

// Anomaly mode has no ground truth ingestion; scores only.
BenchmarkResult runAnomalyBenchmark(
    const std::vector<std::filesystem::path>& files,
    const std::function<AnomalyResult(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

// Convenience conversions from a BenchmarkResult's per-image data
// into the metrics functions' input shape (only images with ground truth
// contribute).
std::vector<DetectionEvaluationItem> toDetectionEvaluationItems(const BenchmarkResult& result);
std::vector<ClassificationEvaluationItem> toClassificationEvaluationItems(const BenchmarkResult& result);

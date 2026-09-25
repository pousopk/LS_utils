#pragma once

#include "manager/batch_evaluation.hpp"
#include "manager/comparison_task_mode.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/yolo_inference.hpp"

#include <optional>
#include <string>
#include <vector>

enum class BatchEvalImageSortMode {
    Filename,
    ConfidenceAscending,
    ConfidenceDescending,
};

enum class BatchEvalConfidenceFilterMode {
    None,
    LessThan,
    GreaterThan,
};

// Detection mode only -- whether an image produced at least one detected
// box. Meaningless for classification (a run always produces a top-1
// prediction unless inference itself failed), so it's ignored there.
enum class BatchEvalDetectionPresenceFilter {
    Any,
    HasDetections,
    NoDetections,
};

// Pure per-image filter/sort logic for the Model Evaluation window's batch
// image list. Deliberately free of GL/worker/Label Studio dependencies so
// it can be unit tested on its own (see tests/batch_eval_filters_tests.cpp).

struct DetectionMatch {
    std::vector<bool> predictionMatched;   // per prediction: true = TP, false = FP
    std::vector<bool> groundTruthMatched;  // per GT box: true = matched, false = missed
};

// Greedy same-class matching: predictions are processed in their stored
// order, each claiming the highest-IoU *unmatched* ground-truth box of the
// same class with rotation-aware IoU >= iouThreshold. Same rule the batch
// image list has always used to decide a detection "mismatch".
DetectionMatch matchDetections(
    const std::vector<Detection>& predictions, const std::vector<GroundTruthBox>& groundTruth,
    float iouThreshold = 0.5f);

// Detection mode only: which single number summarizes an image's boxes.
// Mean is the historical behavior; Lowest surfaces uncertain images and
// Highest surfaces confident false positives.
enum class BatchEvalConfidenceBasis {
    Mean,
    Lowest,
    Highest,
};

// Representative confidence of one slot's result for an image: the
// classification top-1 probability, the anomaly score, or (detection) the
// mean/min/max box confidence per `basis`. std::nullopt if `image` is null
// or has no prediction/detection/anomaly result. `basis` is ignored
// outside detection mode.
std::optional<float> batchEvalImageConfidence(
    ComparisonTaskMode mode, const BatchImageResult* image, BatchEvalConfidenceBasis basis);

// Sort key for confidence sorting: the lower of the two slots'
// confidences; a slot with no value is ignored; 0.0f if neither has one
// (so such images sort first ascending / last descending).
float batchEvalImageSortConfidence(
    ComparisonTaskMode mode, const BatchImageResult* imageA, const BatchImageResult* imageB,
    BatchEvalConfidenceBasis basis);

// Which ground-truth errors an image must have to be listed. Detection
// uses all four; classification only offers Any/AnyError ("Misclassified")
// -- FalsePositives/Missed are treated as AnyError there. Ignored when the
// run has no ground truth, and in anomaly mode.
enum class BatchEvalErrorFilter {
    Any,
    AnyError,
    FalsePositives,
    Missed,
};

// A clicked classification confusion-matrix cell: list only images where
// slot `slotIndex` (0 = model A, 1 = model B) predicted `predictedLabel`
// for true label `trueLabel`.
struct BatchEvalConfusionCellFilter {
    std::string trueLabel;
    std::string predictedLabel;
    int slotIndex = 0;
};

// Every image-list filter except the filename search. A default-constructed
// value passes every image.
struct BatchEvalImageFilters {
    BatchEvalErrorFilter errorFilter = BatchEvalErrorFilter::Any;
    std::string classFilter;  // empty = all classes
    BatchEvalConfidenceBasis confidenceBasis = BatchEvalConfidenceBasis::Mean;
    BatchEvalConfidenceFilterMode confidenceFilterMode = BatchEvalConfidenceFilterMode::None;
    float confidenceFilterThreshold = 0.5f;
    BatchEvalDetectionPresenceFilter detectionPresenceFilter = BatchEvalDetectionPresenceFilter::Any;
    bool modelsDisagreeOnly = false;  // compare mode only
    std::optional<BatchEvalConfusionCellFilter> confusionCell;
};

// AND over every filter in `filters` for one image. `imageA`/`imageB` are
// the two slots' results for the same image (either may be null -- single
// model mode leaves slot B empty); per-slot filters pass if either
// non-null slot passes. `hasGroundTruth` is the run-level flag: without it
// the error filter is ignored.
bool batchEvalImagePassesFilters(
    ComparisonTaskMode mode, bool hasGroundTruth,
    const BatchImageResult* imageA, const BatchImageResult* imageB,
    const BatchEvalImageFilters& filters);

// Sorted, de-duplicated class names across both slots' ground truth
// (boxes / label) and predictions (detections / top-1) -- populates the
// image list's class filter combo. Empty in anomaly mode.
std::vector<std::string> collectBatchEvalClassNames(
    ComparisonTaskMode mode, const BatchEvaluationResult& resultA, const BatchEvaluationResult& resultB);

// Final precision (TP / (TP + FP)) and recall (TP / numGroundTruth) for a
// class after every prediction is processed -- i.e. at the model's
// configured confidence threshold, not a point on the PR curve.
// std::nullopt when the denominator is 0.
std::optional<float> classPrecision(const ClassAveragePrecision& metrics);
std::optional<float> classRecall(const ClassAveragePrecision& metrics);

// Linear lookup of an image's result by filename; nullptr if absent.
const BatchImageResult* findBatchImage(const BatchEvaluationResult& result, const std::string& filename);

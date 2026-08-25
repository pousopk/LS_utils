#pragma once

#include "manager/label_studio_import.hpp"
#include "manager/yolo_inference.hpp"

#include <string>
#include <vector>

struct DetectionEvaluationItem {
    std::vector<Detection> predictions;
    std::vector<GroundTruthBox> groundTruth;
};

struct ClassAveragePrecision {
    std::string className;
    float averagePrecision = 0.0f;
    int numGroundTruth = 0;
    int numPredictions = 0;
    // Final counts after processing every prediction for this class --
    // lets a caller derive overall precision = truePositives /
    // (truePositives + falsePositives) and recall = truePositives /
    // numGroundTruth directly, not just the AP curve.
    int truePositives = 0;
    int falsePositives = 0;
};

struct DetectionMetrics {
    std::vector<ClassAveragePrecision> perClass;
    float meanAveragePrecision = 0.0f;  // mean AP over classes with >=1 ground-truth instance
};

// Computes mAP at a single IoU threshold (default 0.5) across all
// evaluation items (one per image). Per class: predictions are processed
// in descending-confidence order; each is greedily matched to the
// highest-IoU *unmatched* ground-truth box of the same class in the same
// image -- IoU >= iouThreshold is a true positive (consuming that box so
// it can't match again), otherwise a false positive. AP is the
// all-points-interpolated area under the resulting precision/recall
// curve. A class with zero ground-truth instances is still reported (for
// visibility) but excluded from the mean.
DetectionMetrics computeDetectionMetrics(
    const std::vector<DetectionEvaluationItem>& items, float iouThreshold = 0.5f);

#include "manager/detection_metrics.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace {

struct PredictionRecord {
    float confidence = 0.0f;
    int imageIndex = 0;
    cv::Rect box;
};

} // namespace

DetectionMetrics computeDetectionMetrics(const std::vector<DetectionEvaluationItem>& items, float iouThreshold) {
    std::map<std::string, std::vector<PredictionRecord>> predictionsByClass;
    std::map<std::string, int> groundTruthCountByClass;

    for (size_t imgIdx = 0; imgIdx < items.size(); ++imgIdx) {
        for (const auto& det : items[imgIdx].predictions) {
            predictionsByClass[det.className].push_back({det.confidence, static_cast<int>(imgIdx), det.box});
        }
        for (const auto& gt : items[imgIdx].groundTruth) {
            groundTruthCountByClass[gt.className]++;
        }
    }

    std::set<std::string> allClasses;
    for (const auto& kv : predictionsByClass) {
        allClasses.insert(kv.first);
    }
    for (const auto& kv : groundTruthCountByClass) {
        allClasses.insert(kv.first);
    }

    DetectionMetrics metrics;
    float apSum = 0.0f;
    int apCount = 0;

    for (const auto& className : allClasses) {
        const auto gtCountIt = groundTruthCountByClass.find(className);
        const int numGroundTruth = gtCountIt != groundTruthCountByClass.end() ? gtCountIt->second : 0;
        const auto predsIt = predictionsByClass.find(className);
        const int numPredictions = predsIt != predictionsByClass.end() ? static_cast<int>(predsIt->second.size()) : 0;

        ClassAveragePrecision classAP;
        classAP.className = className;
        classAP.numGroundTruth = numGroundTruth;
        classAP.numPredictions = numPredictions;

        if (numGroundTruth == 0) {
            // Undefined AP -- reported for visibility, excluded from the mean.
            metrics.perClass.push_back(classAP);
            continue;
        }
        if (numPredictions == 0) {
            classAP.averagePrecision = 0.0f;
            metrics.perClass.push_back(classAP);
            apSum += 0.0f;
            apCount++;
            continue;
        }

        std::vector<PredictionRecord> predictions = predsIt->second;
        std::sort(predictions.begin(), predictions.end(),
            [](const PredictionRecord& a, const PredictionRecord& b) { return a.confidence > b.confidence; });

        // Ground-truth boxes of this class, indexed by image, so each can
        // be matched at most once.
        std::vector<std::vector<int>> gtIndicesPerImage(items.size());
        std::vector<std::vector<bool>> usedGroundTruth(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            usedGroundTruth[i].assign(items[i].groundTruth.size(), false);
            for (size_t g = 0; g < items[i].groundTruth.size(); ++g) {
                if (items[i].groundTruth[g].className == className) {
                    gtIndicesPerImage[i].push_back(static_cast<int>(g));
                }
            }
        }

        std::vector<float> precision(predictions.size());
        std::vector<float> recall(predictions.size());
        int cumulativeTP = 0;
        int cumulativeFP = 0;

        for (size_t p = 0; p < predictions.size(); ++p) {
            const auto& pred = predictions[p];
            int bestGtIdx = -1;
            float bestIoU = iouThreshold;
            for (int gIdx : gtIndicesPerImage[pred.imageIndex]) {
                if (usedGroundTruth[pred.imageIndex][gIdx]) {
                    continue;
                }
                const float iou = computeIoU(pred.box, items[pred.imageIndex].groundTruth[gIdx].box);
                if (iou >= bestIoU) {
                    bestIoU = iou;
                    bestGtIdx = gIdx;
                }
            }

            if (bestGtIdx >= 0) {
                usedGroundTruth[pred.imageIndex][bestGtIdx] = true;
                cumulativeTP++;
            } else {
                cumulativeFP++;
            }

            precision[p] = static_cast<float>(cumulativeTP) / static_cast<float>(cumulativeTP + cumulativeFP);
            recall[p] = static_cast<float>(cumulativeTP) / static_cast<float>(numGroundTruth);
        }

        // All-points interpolation: precision monotonically non-increasing from the right.
        for (int i = static_cast<int>(precision.size()) - 2; i >= 0; --i) {
            precision[i] = std::max(precision[i], precision[i + 1]);
        }

        float ap = 0.0f;
        float prevRecall = 0.0f;
        for (size_t p = 0; p < precision.size(); ++p) {
            ap += (recall[p] - prevRecall) * precision[p];
            prevRecall = recall[p];
        }

        classAP.averagePrecision = ap;
        classAP.truePositives = cumulativeTP;
        classAP.falsePositives = cumulativeFP;
        metrics.perClass.push_back(classAP);
        apSum += ap;
        apCount++;
    }

    metrics.meanAveragePrecision = apCount > 0 ? apSum / static_cast<float>(apCount) : 0.0f;
    return metrics;
}

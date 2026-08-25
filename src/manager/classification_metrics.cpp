#include "manager/classification_metrics.hpp"

ClassificationMetrics computeClassificationMetrics(const std::vector<ClassificationEvaluationItem>& items) {
    ClassificationMetrics metrics;
    int correct = 0;

    for (const auto& item : items) {
        metrics.confusionMatrix[item.trueLabel][item.predictedLabel]++;
        if (item.predictedLabel == item.trueLabel) {
            correct++;
        }
    }

    metrics.totalEvaluated = static_cast<int>(items.size());
    metrics.accuracy = metrics.totalEvaluated > 0
        ? static_cast<float>(correct) / static_cast<float>(metrics.totalEvaluated)
        : 0.0f;
    return metrics;
}

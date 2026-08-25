#pragma once

#include <map>
#include <string>
#include <vector>

struct ClassificationEvaluationItem {
    std::string predictedLabel;
    std::string trueLabel;
};

struct ClassificationMetrics {
    float accuracy = 0.0f;
    int totalEvaluated = 0;
    // confusionMatrix[trueLabel][predictedLabel] = count
    std::map<std::string, std::map<std::string, int>> confusionMatrix;
};

ClassificationMetrics computeClassificationMetrics(const std::vector<ClassificationEvaluationItem>& items);

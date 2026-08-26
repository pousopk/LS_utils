#include "manager/anomaly_inference.hpp"

#include <algorithm>

AnomalyScore scoreAnomalyHeatmap(const cv::Mat& heatmap, float scoreMin, float scoreMax, float threshold) {
    AnomalyScore result;
    if (heatmap.empty()) {
        return result;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(heatmap, &minVal, &maxVal);
    result.rawScore = static_cast<float>(maxVal);

    const float range = scoreMax - scoreMin;
    result.score = (range > 0.0f) ? std::clamp((result.rawScore - scoreMin) / range, 0.0f, 1.0f)
                                   : std::clamp(result.rawScore, 0.0f, 1.0f);
    result.isAnomalous = result.score >= threshold;
    return result;
}

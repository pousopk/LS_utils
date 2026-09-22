#include "manager/mask_overlay.hpp"

#include "manager/label_color.hpp"

#include <opencv2/imgproc.hpp>

cv::Mat compositeMaskOverlay(const cv::Mat& baseImage, const std::vector<DraftBrushRegion>& regions) {
    cv::Mat result = baseImage.clone();
    constexpr double kMaskAlpha = 0.45;

    for (const auto& region : regions) {
        if (region.mask.empty() || region.mask.size() != baseImage.size()) {
            continue;
        }
        const LabelColor color = colorForClassName(region.className);
        cv::Mat colorLayer(baseImage.size(), baseImage.type(), cv::Scalar(color.b, color.g, color.r));
        cv::Mat blended;
        cv::addWeighted(result, 1.0 - kMaskAlpha, colorLayer, kMaskAlpha, 0.0, blended);
        blended.copyTo(result, region.mask);
    }

    return result;
}

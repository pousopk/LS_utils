#include "ui_common/image_fit.hpp"

#include <algorithm>

ImVec2 fitImageToRegion(int imageWidth, int imageHeight, float maxWidth, float maxHeight) {
    const float safeMaxW = std::max(1.0f, maxWidth);
    const float safeMaxH = std::max(1.0f, maxHeight);

    if (imageWidth <= 0 || imageHeight <= 0) {
        return ImVec2(safeMaxW, safeMaxH);
    }

    const float imageAspect = static_cast<float>(imageWidth) / static_cast<float>(imageHeight);

    float drawW = safeMaxW;
    float drawH = drawW / imageAspect;

    if (drawH > safeMaxH) {
        drawH = safeMaxH;
        drawW = drawH * imageAspect;
    }

    return ImVec2(std::max(1.0f, drawW), std::max(1.0f, drawH));
}

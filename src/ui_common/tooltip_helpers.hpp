#pragma once

#include "ui_common/image_fit.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

inline void showDelayedItemTooltip(const char* text) {
    if (text == nullptr || text[0] == '\0') {
        return;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::SetTooltip("%s", text);
    }
}

// Shows `texture` enlarged (fit to maxEnlargedDim x maxEnlargedDim) in a
// tooltip when the item drawn immediately before this call (an ImGui::Image
// at `baseSize`) is hovered -- a lightweight "hover to zoom in" for any
// preview image, with no extra state to track. A no-op if the enlarged size
// wouldn't actually be bigger than `baseSize` (e.g. a preview pane already
// showing the image at 500x300 gains nothing from an 480x480 popup).
inline void drawHoverEnlargedImage(
    GLuint texture, int textureWidth, int textureHeight, const ImVec2& baseSize, float maxEnlargedDim = 480.0f) {
    if (texture == 0 || textureWidth <= 0 || textureHeight <= 0) {
        return;
    }
    const ImVec2 enlargedSize = fitImageToRegion(textureWidth, textureHeight, maxEnlargedDim, maxEnlargedDim);
    if (enlargedSize.x <= baseSize.x && enlargedSize.y <= baseSize.y) {
        return;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::BeginTooltip();
        ImGui::Image((void*)(intptr_t)texture, enlargedSize);
        ImGui::EndTooltip();
    }
}

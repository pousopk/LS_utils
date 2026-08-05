#pragma once

#include <imgui.h>

inline void showDelayedItemTooltip(const char* text) {
    if (text == nullptr || text[0] == '\0') {
        return;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::SetTooltip("%s", text);
    }
}

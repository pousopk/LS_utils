#include "widgets/ui_theme.hpp"

#include <imgui.h>

namespace {

// Light, warm-white background with a vibrant amber/orange accent.
constexpr ImVec4 kBgDarkest(0.975f, 0.965f, 0.950f, 1.00f);   // main window bg (warm off-white)
constexpr ImVec4 kBgDark(0.992f, 0.988f, 0.980f, 1.00f);      // popups (slightly lighter)
constexpr ImVec4 kBgMid(0.906f, 0.878f, 0.835f, 1.00f);       // table header / plot bg
constexpr ImVec4 kBorder(0.780f, 0.740f, 0.680f, 0.60f);

constexpr ImVec4 kAccent(0.902f, 0.451f, 0.086f, 1.00f);        // vibrant amber/orange
constexpr ImVec4 kAccentHovered(0.980f, 0.549f, 0.157f, 1.00f);
constexpr ImVec4 kAccentActive(0.745f, 0.361f, 0.055f, 1.00f);
constexpr ImVec4 kAccentMuted(0.902f, 0.451f, 0.086f, 0.35f);

// Resting-state tints (opaque, warm apricot) so color reads even before
// hovering/focusing anything, not only on interaction.
constexpr ImVec4 kButtonRest(0.965f, 0.784f, 0.596f, 1.00f);
constexpr ImVec4 kFrameRest(0.945f, 0.898f, 0.827f, 1.00f);
constexpr ImVec4 kHeaderRest(0.933f, 0.663f, 0.427f, 0.70f);

constexpr ImVec4 kText(0.140f, 0.120f, 0.100f, 1.00f);
constexpr ImVec4 kTextDisabled(0.520f, 0.490f, 0.450f, 1.00f);

} // namespace

void applyProfessionalTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    colors[ImGuiCol_Text] = kText;
    colors[ImGuiCol_TextDisabled] = kTextDisabled;
    colors[ImGuiCol_WindowBg] = kBgDarkest;
    colors[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    colors[ImGuiCol_PopupBg] = kBgDark;
    colors[ImGuiCol_Border] = kBorder;
    colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

    colors[ImGuiCol_FrameBg] = kFrameRest;
    colors[ImGuiCol_FrameBgHovered] = kAccentMuted;
    colors[ImGuiCol_FrameBgActive] = kAccentActive;

    colors[ImGuiCol_TitleBg] = kBgDarkest;
    colors[ImGuiCol_TitleBgActive] = kAccentActive;
    colors[ImGuiCol_TitleBgCollapsed] = kBgDarkest;
    colors[ImGuiCol_MenuBarBg] = kFrameRest;

    colors[ImGuiCol_ScrollbarBg] = kBgDarkest;
    colors[ImGuiCol_ScrollbarGrab] = kFrameRest;
    colors[ImGuiCol_ScrollbarGrabHovered] = kAccentMuted;
    colors[ImGuiCol_ScrollbarGrabActive] = kAccent;

    colors[ImGuiCol_CheckMark] = kAccentHovered;
    colors[ImGuiCol_SliderGrab] = kAccent;
    colors[ImGuiCol_SliderGrabActive] = kAccentHovered;

    colors[ImGuiCol_Button] = kButtonRest;
    colors[ImGuiCol_ButtonHovered] = kAccentHovered;
    colors[ImGuiCol_ButtonActive] = kAccentActive;

    colors[ImGuiCol_Header] = kHeaderRest;
    colors[ImGuiCol_HeaderHovered] = kAccentHovered;
    colors[ImGuiCol_HeaderActive] = kAccentActive;

    colors[ImGuiCol_Separator] = kBorder;
    colors[ImGuiCol_SeparatorHovered] = kAccentHovered;
    colors[ImGuiCol_SeparatorActive] = kAccentActive;

    colors[ImGuiCol_ResizeGrip] = kAccentMuted;
    colors[ImGuiCol_ResizeGripHovered] = kAccentHovered;
    colors[ImGuiCol_ResizeGripActive] = kAccentActive;

    colors[ImGuiCol_Tab] = kFrameRest;
    colors[ImGuiCol_TabHovered] = kAccentHovered;
    colors[ImGuiCol_TabSelected] = kAccentActive;
    colors[ImGuiCol_TabSelectedOverline] = kAccent;
    colors[ImGuiCol_TabDimmed] = ImVec4(0.930f, 0.912f, 0.888f, 1.00f);
    colors[ImGuiCol_TabDimmedSelected] = kButtonRest;
    colors[ImGuiCol_TabDimmedSelectedOverline] = kAccentMuted;

    colors[ImGuiCol_PlotLines] = kAccent;
    colors[ImGuiCol_PlotLinesHovered] = kAccentHovered;
    colors[ImGuiCol_PlotHistogram] = kAccent;
    colors[ImGuiCol_PlotHistogramHovered] = kAccentHovered;

    colors[ImGuiCol_TableHeaderBg] = kBgDark;
    colors[ImGuiCol_TableBorderStrong] = kBorder;
    colors[ImGuiCol_TableBorderLight] = kBorder;
    colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(0, 0, 0, 0.035f);

    colors[ImGuiCol_TextSelectedBg] = kAccentMuted;
    colors[ImGuiCol_DragDropTarget] = kAccentHovered;
    colors[ImGuiCol_NavHighlight] = kAccent;
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0, 0, 0, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.2f, 0.2f, 0.2f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.05f, 0.06f, 0.07f, 0.45f);

    // Subtle rounding/spacing so it reads as a deliberate theme, not just recolored defaults.
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(6.0f, 4.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
}

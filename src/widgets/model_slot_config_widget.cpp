#include "widgets/model_slot_config_widget.hpp"

#include <imgui.h>

bool drawModelSlotConfigFields(
    const std::string& onnxPath,
    const std::string& classNamesPath,
    int& inputWidth,
    int& inputHeight,
    float* confThreshold,
    float* nmsThreshold,
    const std::string& autoDetectStatus,
    const std::string& loadError,
    const std::function<void()>& onBrowseModel,
    const std::function<void()>& onBrowseClasses) {
    ImGui::TextWrapped("Model: %s", onnxPath.empty() ? "(none)" : onnxPath.c_str());
    if (ImGui::Button("Browse Model...") && onBrowseModel) {
        onBrowseModel();
    }

    ImGui::TextWrapped("Classes: %s", classNamesPath.empty() ? "(index only)" : classNamesPath.c_str());
    if (ImGui::Button("Browse Classes...") && onBrowseClasses) {
        onBrowseClasses();
    }

    ImGui::InputInt("Input Width", &inputWidth);
    ImGui::InputInt("Input Height", &inputHeight);
    if (confThreshold != nullptr && nmsThreshold != nullptr) {
        ImGui::SliderFloat("Confidence", confThreshold, 0.05f, 0.95f);
        ImGui::SliderFloat("NMS IoU", nmsThreshold, 0.05f, 0.95f);
    }

    if (!autoDetectStatus.empty()) {
        ImGui::TextDisabled("%s", autoDetectStatus.c_str());
    }

    const bool loadClicked = ImGui::Button("Load Model");

    if (!loadError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", loadError.c_str());
    }

    return loadClicked;
}

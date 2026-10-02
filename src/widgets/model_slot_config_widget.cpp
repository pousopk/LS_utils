#include "widgets/model_slot_config_widget.hpp"

#include <imgui.h>

void drawModelSlotConfigFields(ModelSlotConfig& slot, const std::function<void()>& onBrowseModel) {
    ImGui::TextWrapped("Model: %s", slot.onnxPath.empty() ? "(none)" : slot.onnxPath.c_str());
    if (ImGui::Button("Browse Model...") && onBrowseModel) {
        onBrowseModel();
    }

    if (!slot.loadError.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.1f, 0.1f, 1.0f));
        ImGui::TextWrapped("%s", slot.loadError.c_str());
        ImGui::PopStyleColor();
    }
    if (!isModelSlotLoaded(slot)) {
        return;
    }
    ImGui::TextDisabled("%s", slot.summary.c_str());

    if (slot.task == ModelTask::Detection) {
        ImGui::SliderFloat("Confidence", &slot.confThreshold, 0.05f, 0.95f);
        ImGui::SliderFloat("NMS IoU", &slot.nmsThreshold, 0.05f, 0.95f);
    } else if (slot.task == ModelTask::Anomaly) {
        ImGui::SliderFloat("Anomaly Threshold", &slot.anomalyThreshold, 0.0f, 1.0f, "%.2f");
    }
}

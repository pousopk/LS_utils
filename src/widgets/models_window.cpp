#include "widgets/models_window.hpp"

#include "widgets/label_assistant_window.hpp"
#include "widgets/ml_app_ui.hpp"
#include "widgets/benchmark_tab.hpp"

#include <imgui.h>

void drawModelsTab(MlAppUi& ui, ImGuiTabItemFlags flags) {
    if (!ui.showModelsWindow) {
        return;
    }

    if (!ImGui::BeginTabItem("Models", &ui.showModelsWindow, flags)) {
        return;
    }

    auto onOpenConnectionTab = [&ui] { ui.openLabelStudioTab(LabelStudioTab::Connection); };
    auto onLabelTask = [&ui](int taskId) { ui.openLabelingForTask(taskId); };

    if (ImGui::BeginTabBar("ModelsTabs")) {
        ImGuiTabItemFlags benchmarkFlags = ImGuiTabItemFlags_None;
        if (ui.pendingModelsTab == ModelsTab::Benchmark) {
            benchmarkFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Benchmark", nullptr, benchmarkFlags)) {
            drawBenchmarkTabContent(ui.benchmarkState, ui.labelStudioSession, ui.labelStudioProjectData, onOpenConnectionTab);
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags labelAssistantFlags = ImGuiTabItemFlags_None;
        if (ui.pendingModelsTab == ModelsTab::LabelAssistant) {
            labelAssistantFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Label Assistant", nullptr, labelAssistantFlags)) {
            drawLabelAssistantTabContent(
                ui.labelAssistantState, ui.labelStudioSession, ui.labelStudioProjectData, onLabelTask,
                onOpenConnectionTab);
            ImGui::EndTabItem();
        }

        ui.pendingModelsTab.reset();
        ImGui::EndTabBar();
    }

    ImGui::EndTabItem();
}

#include "widgets/ml_app_ui.hpp"

#include "widgets/label_studio_window.hpp"
#include "widgets/models_window.hpp"

#include <imgui.h>

#include <optional>
#include <utility>

void MlAppUi::update() {
    // The shared task list feeds Label Assistant (Models tab) as well as the
    // Label Studio tab's own tools, so it refreshes while either tab is open
    // -- and before any of the states below read it.
    if (showLabelStudioWindow || showModelsWindow) {
        updateSharedLabelStudioProjectData(labelStudioProjectData, labelStudioSession);
    }
    if (showModelsWindow) {
        updateBatchRuntime(
            modelEvaluationState.taskMode, modelEvaluationState.slots, modelEvaluationState.batch.imageFolderPath,
            modelEvaluationState.batch, labelStudioSession);
        updateLabelAssistantState(labelAssistantState, labelStudioSession, labelStudioProjectData);
    }
    if (showLabelStudioWindow) {
        updateLabelingState(labelingState, labelStudioSession, labelStudioProjectData);
        updateDatasetBrowserState(datasetBrowserState, labelStudioSession, labelStudioProjectData);
    }
}

void MlAppUi::drawMainLayout() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Label Studio", nullptr, showLabelStudioWindow)) {
                openLabelStudioTab();
            }
            if (ImGui::MenuItem("Models", nullptr, showModelsWindow)) {
                openModelsTab();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("About", nullptr, false, false);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // The host fills the viewport below the menu bar and never comes to the
    // front, so the app's popups and modals always draw above it.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    constexpr ImGuiWindowFlags kHostFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
                                            | ImGuiWindowFlags_NoSavedSettings
                                            | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const bool hostVisible = ImGui::Begin("##MainHost", nullptr, kHostFlags);
    ImGui::PopStyleVar(2);
    if (hostVisible && ImGui::BeginTabBar("MainTabs")) {
        // Taken up front, not reset afterwards: the tabs drawn below can
        // request a switch themselves (e.g. Benchmark's "open the
        // connection tab" button), and that request has to survive to
        // the next frame to take effect.
        const std::optional<MainTab> pending = std::exchange(pendingMainTab, std::nullopt);
        auto selectFlags = [pending](MainTab tab) {
            return pending == tab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
        };
        drawLabelStudioTab(*this, selectFlags(MainTab::LabelStudio));
        drawModelsTab(*this, selectFlags(MainTab::Models));
        ImGui::EndTabBar();
    }
    ImGui::End();
}

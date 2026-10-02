#include "widgets/ml_app_ui.hpp"

#include "widgets/label_studio_window.hpp"
#include "widgets/model_evaluation_window.hpp"

#include <imgui.h>

#include <optional>
#include <utility>

void MlAppUi::update() {
    if (showModelEvaluationWindow) {
        updateBatchRuntime(
            modelEvaluationState.taskMode, modelEvaluationState.slots, modelEvaluationState.batch.imageFolderPath,
            modelEvaluationState.batch, labelStudioSession);
    }
    if (showLabelStudioWindow) {
        updateSharedLabelStudioProjectData(labelStudioProjectData, labelStudioSession);
        updateLabelingState(labelingState, labelStudioSession, labelStudioProjectData);
        updateLabelAssistantState(labelAssistantState, labelStudioSession, labelStudioProjectData);
        updateDatasetBrowserState(datasetBrowserState, labelStudioSession, labelStudioProjectData);
    }
}

void MlAppUi::drawMainLayout() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Label Studio", nullptr, showLabelStudioWindow)) {
                openLabelStudioTab();
            }
            if (ImGui::MenuItem("Model Evaluation", nullptr, showModelEvaluationWindow)) {
                openModelEvaluationTab();
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
        // request a switch themselves (e.g. Model Evaluation's "open the
        // connection tab" button), and that request has to survive to
        // the next frame to take effect.
        const std::optional<MainTab> pending = std::exchange(pendingMainTab, std::nullopt);
        auto selectFlags = [pending](MainTab tab) {
            return pending == tab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
        };
        drawLabelStudioTab(*this, selectFlags(MainTab::LabelStudio));
        drawModelEvaluationTab(
            &showModelEvaluationWindow, selectFlags(MainTab::ModelEvaluation), modelEvaluationState,
            labelStudioSession, [this] { openLabelStudioTab(LabelStudioTab::Connection); });
        ImGui::EndTabBar();
    }
    ImGui::End();
}

#include "widgets/ml_app_ui.hpp"

#include "widgets/label_studio_window.hpp"
#include "widgets/model_evaluation_window.hpp"

#include <imgui.h>

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
        updateTimestampSearchState(timestampSearchState, labelStudioSession);
        updateDatasetBrowserState(datasetBrowserState, labelStudioSession, labelStudioProjectData);
    }
}

void MlAppUi::drawMainLayout() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Label Studio", nullptr, showLabelStudioWindow)) {
                showLabelStudioWindow = true;
            }
            if (ImGui::MenuItem("Model Evaluation", nullptr, showModelEvaluationWindow)) {
                showModelEvaluationWindow = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("About", nullptr, false, false);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    drawModelEvaluationWindow(
        &showModelEvaluationWindow, modelEvaluationState, labelStudioSession,
        [this] { openLabelStudioTab(LabelStudioTab::Connection); });
    drawLabelStudioWindow(*this);
}

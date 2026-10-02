#pragma once

#include "manager/dataset_browser_state.hpp"
#include "manager/label_assistant_state.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/labeling_state.hpp"
#include "manager/model_evaluation_state.hpp"

#include <optional>

// The Label Studio window's tabs.
enum class LabelStudioTab {
    Connection,
    Labeling,
    LabelAssistant,
    DatasetBrowser,
};

// The main window's top-level tabs.
enum class MainTab {
    LabelStudio,
    ModelEvaluation,
};

// Everything the ML app draws: one host window filling the main viewport,
// whose top-level tab bar holds the Label Studio tab (its four sub-tabs
// share one session and one loaded task list) and the batch Model
// Evaluation tab. Both open at startup; the Window menu reopens them.
struct MlAppUi {
    bool showModelEvaluationWindow = true;
    ModelEvaluationState modelEvaluationState;

    bool showLabelStudioWindow = true;
    LabelStudioSessionState labelStudioSession;
    SharedLabelStudioProjectData labelStudioProjectData;
    LabelingState labelingState;
    LabelAssistantState labelAssistantState;
    DatasetBrowserState datasetBrowserState;
    // Which tab to force-select on the next draw (consumed and cleared by drawLabelStudioTab).
    std::optional<LabelStudioTab> pendingLabelStudioTab;
    // Which top-level tab to force-select on the next draw (consumed and cleared by drawMainLayout).
    std::optional<MainTab> pendingMainTab;

    // Opens the Model Evaluation tab (if closed) and switches to it on the next draw.
    void openModelEvaluationTab() {
        showModelEvaluationWindow = true;
        pendingMainTab = MainTab::ModelEvaluation;
    }

    // Opens the Label Studio tab (if closed) and switches it to `tab` on its next draw.
    void openLabelStudioTab(LabelStudioTab tab) {
        showLabelStudioWindow = true;
        pendingMainTab = MainTab::LabelStudio;
        pendingLabelStudioTab = tab;
    }

    // Opens the Label Studio tab (if closed) and switches to it, keeping its current sub-tab.
    void openLabelStudioTab() {
        showLabelStudioWindow = true;
        pendingMainTab = MainTab::LabelStudio;
    }

    // Opens the Labeling tab scoped to a specific task.
    void openLabelingForTask(int taskId) {
        labelingState.focusTaskId = taskId;
        openLabelStudioTab(LabelStudioTab::Labeling);
    }

    // Per-frame state updates (no ImGui calls), for whichever windows are open.
    void update();

    // Main menu bar plus the host window and its tabs.
    void drawMainLayout();
};

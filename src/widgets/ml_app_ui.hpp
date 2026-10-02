#pragma once

#include "manager/dataset_browser_state.hpp"
#include "manager/label_assistant_state.hpp"
#include "manager/label_studio_profiles_state.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/labeling_state.hpp"
#include "manager/model_evaluation_state.hpp"

#include <optional>

// The Label Studio window's tabs.
enum class LabelStudioTab {
    Connection,
    Labeling,
    DatasetBrowser,
};

// The Models tab's sub-tabs.
enum class ModelsTab {
    Benchmark,
    LabelAssistant,
};

// The main window's top-level tabs.
enum class MainTab {
    LabelStudio,
    Models,
};

// Everything the ML app draws: one host window filling the main viewport,
// whose top-level tab bar holds the Label Studio tab (Connection, Labeling
// and Dataset Browser, sharing one session and one loaded task list) and the
// Models tab (Benchmark and Label Assistant, the tools that run ONNX models;
// Label Assistant reads the same session and task list). Both open at
// startup; the Window menu reopens them.
struct MlAppUi {
    bool showModelsWindow = true;
    ModelEvaluationState modelEvaluationState;
    // Which Models sub-tab to force-select on the next draw (consumed and cleared by drawModelsTab).
    std::optional<ModelsTab> pendingModelsTab;

    bool showLabelStudioWindow = true;
    LabelStudioSessionState labelStudioSession;
    // Saved connection profiles for the Connection tab; read from disk once, here.
    LabelStudioProfilesState labelStudioProfiles = loadLabelStudioProfilesState();
    SharedLabelStudioProjectData labelStudioProjectData;
    LabelingState labelingState;
    LabelAssistantState labelAssistantState;
    DatasetBrowserState datasetBrowserState;
    // Which tab to force-select on the next draw (consumed and cleared by drawLabelStudioTab).
    std::optional<LabelStudioTab> pendingLabelStudioTab;
    // Which top-level tab to force-select on the next draw (consumed and cleared by drawMainLayout).
    std::optional<MainTab> pendingMainTab;

    // Opens the Models tab (if closed) and switches to it, keeping its current sub-tab.
    void openModelsTab() {
        showModelsWindow = true;
        pendingMainTab = MainTab::Models;
    }

    // Opens the Models tab (if closed) and switches it to `tab` on its next draw.
    void openModelsTab(ModelsTab tab) {
        openModelsTab();
        pendingModelsTab = tab;
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

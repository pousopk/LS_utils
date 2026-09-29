#pragma once

#include "manager/dataset_browser_state.hpp"
#include "manager/label_assistant_state.hpp"
#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/labeling_state.hpp"
#include "manager/model_evaluation_state.hpp"
#include "manager/timestamp_search_state.hpp"

#include <optional>

// The Label Studio window's tabs.
enum class LabelStudioTab {
    Connection,
    Labeling,
    LabelAssistant,
    TimestampSearch,
    DatasetBrowser,
};

// Everything the ML app draws: the Label Studio window (its five tabs
// share one session and one loaded task list) and the batch Model
// Evaluation window. Both open at startup; the Window menu reopens them.
struct MlAppUi {
    bool showModelEvaluationWindow = true;
    ModelEvaluationState modelEvaluationState;

    bool showLabelStudioWindow = true;
    LabelStudioSessionState labelStudioSession;
    SharedLabelStudioProjectData labelStudioProjectData;
    LabelingState labelingState;
    LabelAssistantState labelAssistantState;
    TimestampSearchState timestampSearchState;
    DatasetBrowserState datasetBrowserState;
    // Which tab to force-select on the next draw (consumed and cleared by drawLabelStudioWindow).
    std::optional<LabelStudioTab> pendingLabelStudioTab;

    // Opens the Label Studio window (if not already open) and switches it to `tab` on its next draw.
    void openLabelStudioTab(LabelStudioTab tab) {
        showLabelStudioWindow = true;
        pendingLabelStudioTab = tab;
    }

    // Opens the Labeling tab scoped to a specific task.
    void openLabelingForTask(int taskId) {
        labelingState.focusTaskId = taskId;
        openLabelStudioTab(LabelStudioTab::Labeling);
    }

    // Per-frame state updates (no ImGui calls), for whichever windows are open.
    void update();

    // Main menu bar plus both windows.
    void drawMainLayout();
};

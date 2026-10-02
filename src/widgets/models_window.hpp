#pragma once

#include <imgui.h>

struct MlAppUi;

// The Models tab of the main window: one closable tab item wrapping a nested
// tab bar with the tools that run ONNX models. Must be called inside a
// BeginTabBar; draws nothing while ui.showModelsWindow is false. Takes
// MlAppUi& directly for the same reason drawLabelStudioTab does -- it
// coordinates several of MlAppUi's own states (modelEvaluationState,
// labelAssistantState, labelStudioSession, pendingModelsTab) and its tab
// switching.
void drawModelsTab(MlAppUi& ui, ImGuiTabItemFlags flags);

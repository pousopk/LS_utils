#pragma once

#include "manager/label_studio_project_data.hpp"
#include "manager/label_studio_session.hpp"

#include <imgui.h>

#include <functional>

struct MlAppUi;

// The merged Label Studio tab of the main window: one closable tab item
// wrapping a nested tab bar with Connection/Labeling/Dataset Browser tabs.
// Must be called inside a BeginTabBar; draws
// nothing while ui.showLabelStudioWindow is false. Takes MlAppUi& directly -- like
// drawConnectionSidebar, this widget's whole job is coordinating several of
// MlAppUi's own states (labelStudioSession, labelingState,
// datasetBrowserState, pendingMainTab, pendingLabelStudioTab), so that coupling is
// intrinsic rather than something to hide behind narrower parameters.
void drawLabelStudioTab(MlAppUi& ui, ImGuiTabItemFlags flags);

// Read-only summary shown by every LS-consuming tab/window in place of its
// own connection-field inputs: "Connected: <project>" with a button to
// reopen the connection tab (e.g. to switch projects), or "not connected"
// with a button to open it for the first time. Both buttons invoke the same
// onOpenLabelStudioWindow callback -- opening the connection tab is opening
// the connection tab, whether connecting for the first time or changing
// project.
// One line under a tab's task count whenever the shared list is limited
// to an import-date range ("Tasks imported 2026/09/01 - 2026/09/15"), so
// a filtered list is never mistaken for the whole project -- plus a
// warning when Label Studio ignored the date filter and every task is
// being downloaded and filtered locally instead. Draws nothing for the
// whole project.
void drawSharedTaskRangeNote(const SharedLabelStudioProjectData& sharedData);

void drawLabelStudioSessionSummary(
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow);

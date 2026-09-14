#pragma once

#include "manager/label_studio_session.hpp"

#include <functional>

struct AppUi;

// The merged Label Studio window: one Begin/End wrapping a tab bar with
// Connection/Labeling/Label Assistant/Find by Timestamp tabs (formerly four
// separate top-level windows). Takes AppUi& directly -- like
// drawConnectionSidebar, this widget's whole job is coordinating several of
// AppUi's own states (labelStudioSession, labelingState, labelAssistantState,
// timestampSearchState, pendingLabelStudioTab), so that coupling is
// intrinsic rather than something to hide behind narrower parameters.
void drawLabelStudioWindow(AppUi& ui);

// Read-only summary shown by every LS-consuming tab/window in place of its
// own connection-field inputs: "Connected: <project>" with a button to
// reopen the connection tab (e.g. to switch projects), or "not connected"
// with a button to open it for the first time. Both buttons invoke the same
// onOpenLabelStudioWindow callback -- opening the connection tab is opening
// the connection tab, whether connecting for the first time or changing
// project.
void drawLabelStudioSessionSummary(
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow);

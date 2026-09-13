#pragma once

#include "manager/label_studio_session.hpp"

#include <functional>

void drawLabelStudioWindow(bool* show, LabelStudioSessionState& session);

// Read-only summary shown by each of the four LS-consuming windows in
// place of their former own connection-field inputs: "Connected: <project>"
// with a button to reopen the panel (e.g. to switch projects), or "not
// connected" with a button to open it for the first time. Both buttons
// invoke the same onOpenLabelStudioWindow callback -- opening the panel
// is opening the panel, whether connecting for the first time or
// changing project.
void drawLabelStudioSessionSummary(
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow);

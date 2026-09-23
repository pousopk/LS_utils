#pragma once

#include "manager/labeling_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>

// Content for the Labeling tab of the merged Label Studio window (no
// Begin/End of its own). `mergedWindowOpen` points at the merged window's
// own open flag -- used only to veto/complete a close of the whole merged
// window while this tab has unsaved edits (see the unsaved-changes prompt
// handling in the .cpp); it does not gate whether this function runs.
void drawLabelingTabContent(
    bool* mergedWindowOpen, LabelingState& state, const LabelStudioSessionState& session,
    SharedLabelStudioProjectData& sharedData, const std::function<void()>& onOpenLabelStudioWindow);

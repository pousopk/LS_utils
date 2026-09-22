#pragma once

#include "manager/dataset_browser_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>

// Content for the Dataset Browser tab of the merged Label Studio window
// (no Begin/End of its own) -- filter controls, a virtualized thumbnail
// grid of the currently-matching tasks, a click-to-enlarge single
// preview, and export to a local folder.
void drawDatasetBrowserTabContent(
    DatasetBrowserState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow);

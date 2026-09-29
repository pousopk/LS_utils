#pragma once

#include "manager/dataset_browser_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>

// Content for the Dataset Browser tab of the merged Label Studio window
// (no Begin/End of its own) -- filter controls, a virtualized thumbnail
// grid of the currently-matching tasks, a click-to-enlarge single
// preview, and export to a local folder. `sharedData` is mutable only so
// the Refresh button can call refreshSharedLabelStudioProjectData.
void drawDatasetBrowserTabContent(
    DatasetBrowserState& state, const LabelStudioSessionState& session, SharedLabelStudioProjectData& sharedData,
    const std::function<void()>& onOpenLabelStudioWindow);

#pragma once

#include "manager/timestamp_search_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>
#include <string>

using LabelTaskCallback = std::function<void(int taskId)>;

// Content for the Find by Timestamp tab of the merged Label Studio window
// (no Begin/End of its own).
void drawTimestampSearchTabContent(
    TimestampSearchState& state, const LabelStudioSessionState& session, const LabelTaskCallback& onLabelTask,
    const std::function<void()>& onOpenLabelStudioWindow);

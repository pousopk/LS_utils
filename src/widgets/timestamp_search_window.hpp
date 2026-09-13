#pragma once

#include "manager/timestamp_search_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>
#include <string>

using LabelTaskCallback = std::function<void(int taskId)>;

void drawTimestampSearchWindow(
    bool* show, TimestampSearchState& state, const LabelStudioSessionState& session,
    const LabelTaskCallback& onLabelTask, const std::function<void()>& onOpenLabelStudioWindow);

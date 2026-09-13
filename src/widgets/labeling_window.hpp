#pragma once

#include "manager/labeling_state.hpp"
#include "manager/label_studio_session.hpp"

#include <functional>

void drawLabelingWindow(
    bool* show, LabelingState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow);

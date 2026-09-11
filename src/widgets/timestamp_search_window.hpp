#pragma once

#include "manager/timestamp_search_state.hpp"

#include <functional>
#include <string>

using LabelTaskCallback = std::function<void(const std::string& baseUrl, int projectId, const std::string& apiToken, int taskId)>;

void drawTimestampSearchWindow(bool* show, TimestampSearchState& state, const LabelTaskCallback& onLabelTask);

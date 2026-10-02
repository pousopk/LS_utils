#pragma once

#include "manager/item_filters.hpp"

#include <map>
#include <string>
#include <vector>

// ImGui controls for the item_filters blocks, so every filterable list
// draws the same control the same way. Each returns true on the frame
// its value changes (callers that cache a filtered result re-filter
// then). `label`/`id` also scopes the widget's internal IDs, so it must
// be unique within the current ID stack.

// "None / < threshold / > threshold" combo plus a 0-1 threshold slider,
// disabled while None.
bool drawConfidenceFilter(const char* label, ConfidenceFilter& filter);

bool drawPresenceFilter(
    const char* label, PresenceFilter& filter, const char* anyLabel = "Any", const char* hasLabel = "Has",
    const char* lacksLabel = "Lacks");

// Combo with "All classes" first, then `options`. A selected class
// missing from `options` is still shown as the preview.
bool drawClassFilter(const char* label, ClassFilter& filter, const std::vector<std::string>& options);

bool drawTextSearch(const char* id, const char* hint, TextSearch& search);

// `noneLabel` names the unsorted order ("Filename" for file lists).
bool drawConfidenceSort(const char* label, ConfidenceSort& sort, const char* noneLabel = "Filename");

// Any time / Around / Range radio, then a date picker button plus an
// HH:MM:SS box per time (Around: one, plus +/- minutes; Range: From and
// To, where a blank box means the whole day), then a red note from
// describeTimeWindowProblem when the filter is inactive because of its
// input.
// `markedDays` (optional) is passed to every date picker so days with
// data are tinted in the calendar.
bool drawTimeWindowFilter(
    const char* id, TimeWindowFilter& filter, const std::map<CalendarDate, int>* markedDays = nullptr);

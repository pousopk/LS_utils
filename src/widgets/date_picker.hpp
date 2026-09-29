#pragma once

#include "manager/calendar_date.hpp"

#include <optional>

// A button showing `value` as "YYYY/MM/DD" (or `emptyText` when unset)
// that opens a month-view calendar popup: << < Month Year > >> to move
// by year/month, a Monday-first grid of day cells (today tinted, the
// selected day highlighted), and Today / Clear buttons. Dear ImGui has
// no date widget of its own; this is built only from public ImGui calls
// (popup, table, selectable), with the date math in calendar_date.hpp.
// `label` is drawn to the right, ImGui-style, and also scopes the popup's
// ID, so it must be unique within the current ID stack. Returns true on
// the frame `value` changes.
bool DatePickerButton(const char* label, std::optional<CalendarDate>& value, const char* emptyText = "Any");

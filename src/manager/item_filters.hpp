#pragma once

#include "manager/calendar_date.hpp"

#include <algorithm>
#include <ctime>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Filter building blocks shared by every filterable list in the app
// (Dataset Browser, Benchmark's image list, Label Assistant's
// drafts). Each block tests one value; each tool composes the blocks it
// needs into its own spec struct alongside its tool-specific filters.
// Pure -- no ImGui; the matching controls are in widgets/filter_widgets.
// Every block default-constructs to "passes everything".

enum class ThresholdMode { None, LessThan, GreaterThan };

struct ConfidenceFilter {
    ThresholdMode mode = ThresholdMode::None;
    float threshold = 0.5f;
    // None: always true. Active: false for a missing value, otherwise a
    // strict < / > against threshold.
    bool passes(std::optional<float> value) const;
};

enum class Presence { Any, Has, Lacks };

struct PresenceFilter {
    Presence presence = Presence::Any;
    bool passes(bool has) const;
};

struct ClassFilter {
    std::string className;   // empty = all classes
    // Exact, case-sensitive match against any element / the one class.
    bool passes(const std::vector<std::string>& classNames) const;
    bool passes(std::string_view singleClass) const;
};

struct TextSearch {
    std::string text;   // empty = all
    // Case-insensitive substring against the name's file-name part --
    // exactly fileNameMatchesFilter's rule.
    bool passes(std::string_view name) const;
};

enum class TimeWindowMode { Off, AroundTime, Range };

// Epoch seconds, both ends inclusive; an unset end is open.
struct TimeWindow {
    std::optional<std::time_t> start;
    std::optional<std::time_t> end;
    std::optional<std::time_t> center;   // AroundTime only
};

// One time window on an item's timestamp, in local time: either a
// center date + time +/- toleranceMinutes, or a From/To range. Dates come
// from the date picker; times are typed "HH:MM" or "HH:MM:SS". In Range
// mode a side without a date is open, and a blank time covers the whole
// day (From = 00:00:00, To = 23:59:59); a To time without seconds covers
// the whole minute.
struct TimeWindowFilter {
    TimeWindowMode mode = TimeWindowMode::Off;
    std::optional<CalendarDate> centerDate;
    std::string centerTime;
    int toleranceMinutes = 2;   // negative is treated as 0
    std::optional<CalendarDate> fromDate;
    std::string fromTime;
    std::optional<CalendarDate> toDate;
    std::string toTime;
    // std::nullopt means "inactive": Off, a missing date or time Around
    // needs, a time that doesn't parse, Range with no date on either
    // side, or From after To. Callers must treat it as passing
    // everything -- an unfinished entry must not empty the list.
    std::optional<TimeWindow> resolve() const;
};

// Why an active filter resolves to nothing, for display next to the
// controls; empty when Off or when it resolves fine.
std::string describeTimeWindowProblem(const TimeWindowFilter& filter);

// True with no window. With one, false for a missing time, otherwise
// start <= t <= end on whichever sides are set.
bool passesTimeWindow(const std::optional<TimeWindow>& window, std::optional<std::time_t> t);

enum class ConfidenceSort { None, Ascending, Descending };

// Stable sort of `items` by key(item) -> std::optional<float>. Missing
// keys sort first ascending and last descending. None leaves the order
// alone.
template <class Item, class KeyFn>
void sortByConfidence(std::vector<Item>& items, ConfidenceSort sort, KeyFn key) {
    if (sort == ConfidenceSort::None) {
        return;
    }
    const bool ascending = sort == ConfidenceSort::Ascending;
    std::stable_sort(items.begin(), items.end(), [&](const Item& a, const Item& b) {
        const std::optional<float> keyA = key(a);
        const std::optional<float> keyB = key(b);
        if (!keyA || !keyB) {
            return ascending ? (!keyA && keyB.has_value()) : (keyA.has_value() && !keyB);
        }
        return ascending ? *keyA < *keyB : *keyA > *keyB;
    });
}

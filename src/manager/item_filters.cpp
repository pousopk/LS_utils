#include "manager/item_filters.hpp"

#include "ui_common/file_browser_utils.hpp"

#include <filesystem>
#include <regex>

bool ConfidenceFilter::passes(std::optional<float> value) const {
    if (mode == ThresholdMode::None) {
        return true;
    }
    if (!value) {
        return false;
    }
    return mode == ThresholdMode::LessThan ? *value < threshold : *value > threshold;
}

bool PresenceFilter::passes(bool has) const {
    switch (presence) {
    case Presence::Any:
        return true;
    case Presence::Has:
        return has;
    case Presence::Lacks:
        return !has;
    }
    return true;
}

bool ClassFilter::passes(const std::vector<std::string>& classNames) const {
    return className.empty() || std::find(classNames.begin(), classNames.end(), className) != classNames.end();
}

bool ClassFilter::passes(std::string_view singleClass) const {
    return className.empty() || singleClass == className;
}

bool TextSearch::passes(std::string_view name) const {
    return fileNameMatchesFilter(std::filesystem::path(std::string(name)), text);
}

namespace {

constexpr const char* kTimeFormatProblem = "Time must be HH:MM or HH:MM:SS";

struct TimeOfDay {
    int hour = 0;
    int minute = 0;
    std::optional<int> second;   // unset when typed as HH:MM
};

std::optional<TimeOfDay> parseTimeOfDay(const std::string& text) {
    static const std::regex kShape(R"(\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*)");
    std::smatch match;
    if (!std::regex_match(text, match, kShape)) {
        return std::nullopt;
    }
    TimeOfDay time;
    time.hour = std::stoi(match[1].str());
    time.minute = std::stoi(match[2].str());
    if (match[3].matched) {
        time.second = std::stoi(match[3].str());
    }
    if (time.hour > 23 || time.minute > 59 || time.second.value_or(0) > 59) {
        return std::nullopt;
    }
    return time;
}

// Local wall-clock time on `date` -> epoch, letting mktime decide DST for
// that moment (adding seconds to localMidnight would be off by an hour
// on DST-change days).
std::time_t localDateTime(const CalendarDate& date, int hour, int minute, int second) {
    std::tm tm{};
    tm.tm_year = date.year - 1900;
    tm.tm_mon = date.month - 1;
    tm.tm_mday = date.day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
    tm.tm_isdst = -1;
    return std::mktime(&tm);
}

// One Range side. No date -> open side (true, `out` unset). Blank time ->
// whole day. Returns false only when a time is typed and doesn't parse.
bool resolveRangeSide(
    const std::optional<CalendarDate>& date, const std::string& timeText, bool isEnd, std::optional<std::time_t>& out) {
    out.reset();
    if (!date) {
        return true;
    }
    const bool blank = timeText.find_first_not_of(" \t") == std::string::npos;
    if (blank) {
        out = isEnd ? localDateTime(*date, 23, 59, 59) : localDateTime(*date, 0, 0, 0);
        return true;
    }
    const std::optional<TimeOfDay> time = parseTimeOfDay(timeText);
    if (!time) {
        return false;
    }
    out = localDateTime(*date, time->hour, time->minute, time->second.value_or(isEnd ? 59 : 0));
    return true;
}

} // namespace

std::optional<TimeWindow> TimeWindowFilter::resolve() const {
    switch (mode) {
    case TimeWindowMode::Off:
        return std::nullopt;
    case TimeWindowMode::AroundTime: {
        if (!centerDate) {
            return std::nullopt;
        }
        const std::optional<TimeOfDay> time = parseTimeOfDay(centerTime);
        if (!time) {
            return std::nullopt;
        }
        const std::time_t center = localDateTime(*centerDate, time->hour, time->minute, time->second.value_or(0));
        const std::time_t tolerance = static_cast<std::time_t>(std::max(0, toleranceMinutes)) * 60;
        return TimeWindow{center - tolerance, center + tolerance, center};
    }
    case TimeWindowMode::Range: {
        std::optional<std::time_t> from;
        std::optional<std::time_t> to;
        if (!resolveRangeSide(fromDate, fromTime, /*isEnd=*/false, from)
            || !resolveRangeSide(toDate, toTime, /*isEnd=*/true, to)) {
            return std::nullopt;
        }
        if (!from && !to) {
            return std::nullopt;
        }
        if (from && to && *from > *to) {
            return std::nullopt;
        }
        return TimeWindow{from, to, std::nullopt};
    }
    }
    return std::nullopt;
}

std::string describeTimeWindowProblem(const TimeWindowFilter& filter) {
    if (filter.mode == TimeWindowMode::Off || filter.resolve().has_value()) {
        return "";
    }
    if (filter.mode == TimeWindowMode::AroundTime) {
        if (!filter.centerDate) {
            return "Pick a date";
        }
        return kTimeFormatProblem;
    }
    if (!filter.fromDate && !filter.toDate) {
        return "Pick a From or To date";
    }
    std::optional<std::time_t> from;
    std::optional<std::time_t> to;
    if (!resolveRangeSide(filter.fromDate, filter.fromTime, false, from)
        || !resolveRangeSide(filter.toDate, filter.toTime, true, to)) {
        return kTimeFormatProblem;
    }
    return "From is after To";
}

bool passesTimeWindow(const std::optional<TimeWindow>& window, std::optional<std::time_t> t) {
    if (!window) {
        return true;
    }
    if (!t) {
        return false;
    }
    if (window->start && *t < *window->start) {
        return false;
    }
    if (window->end && *t > *window->end) {
        return false;
    }
    return true;
}

#pragma once

#include <compare>
#include <ctime>
#include <string>

// A plain local calendar day (no time, no timezone) -- what the date
// picker edits and what the task import-date range is expressed in.
// Month and day are 1-based.
struct CalendarDate {
    int year = 1970;
    int month = 1;
    int day = 1;

    auto operator<=>(const CalendarDate&) const = default;
};

// Pure function: 28-31, Gregorian leap-year rules.
int daysInMonth(int year, int month);

// Pure function: 0 = Monday ... 6 = Sunday (the date picker's column
// order), via Sakamoto's method -- no dependency on the C library's
// timezone state.
int weekdayMondayFirst(const CalendarDate& date);

// Pure function: shifts year/month by `deltaMonths` (negative goes back),
// clamping the day to the target month's length (Jan 31 + 1 -> Feb 28).
CalendarDate addMonths(const CalendarDate& date, int deltaMonths);

// Pure function: the following calendar day.
CalendarDate nextDay(const CalendarDate& date);

// Pure function: "YYYY/MM/DD", the same order Timestamp Search's typed
// timestamps use.
std::string formatCalendarDate(const CalendarDate& date);

// The epoch time of 00:00:00 local time on `date` (mktime, DST decided
// by the system's timezone rules). Not pure: depends on the process's
// timezone setting, like parseTypedLocalTimestamp.
std::time_t localMidnight(const CalendarDate& date);

// Today's date in local time. Not pure (reads the clock).
CalendarDate todayLocal();

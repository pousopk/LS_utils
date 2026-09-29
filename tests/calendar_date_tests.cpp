#include "manager/calendar_date.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

void test_daysInMonth_handlesLeapYears() {
    CHECK(daysInMonth(2026, 1) == 31);
    CHECK(daysInMonth(2026, 2) == 28);
    CHECK(daysInMonth(2024, 2) == 29);
    CHECK(daysInMonth(1900, 2) == 28);   // divisible by 100, not 400
    CHECK(daysInMonth(2000, 2) == 29);   // divisible by 400
    CHECK(daysInMonth(2026, 4) == 30);
    CHECK(daysInMonth(2026, 12) == 31);
}

void test_weekdayMondayFirst_knownDates() {
    CHECK(weekdayMondayFirst((CalendarDate{2026, 9, 29})) == 1);   // Tuesday
    CHECK(weekdayMondayFirst((CalendarDate{2026, 9, 1})) == 1);    // Tuesday
    CHECK(weekdayMondayFirst((CalendarDate{2024, 2, 29})) == 3);   // Thursday
    CHECK(weekdayMondayFirst((CalendarDate{2000, 1, 1})) == 5);    // Saturday
    CHECK(weekdayMondayFirst((CalendarDate{2026, 3, 1})) == 6);    // Sunday
}

void test_addMonths_wrapsYearsAndClampsDay() {
    CHECK((addMonths(CalendarDate{2026, 12, 15}, 1) == (CalendarDate{2027, 1, 15})));
    CHECK((addMonths(CalendarDate{2026, 1, 15}, -1) == (CalendarDate{2025, 12, 15})));
    CHECK((addMonths(CalendarDate{2026, 1, 31}, 1) == (CalendarDate{2026, 2, 28})));
    CHECK((addMonths(CalendarDate{2026, 3, 10}, -14) == (CalendarDate{2025, 1, 10})));
}

void test_nextDay_crossesMonthAndYear() {
    CHECK((nextDay(CalendarDate{2026, 9, 29}) == (CalendarDate{2026, 9, 30})));
    CHECK((nextDay(CalendarDate{2026, 9, 30}) == (CalendarDate{2026, 10, 1})));
    CHECK((nextDay(CalendarDate{2026, 12, 31}) == (CalendarDate{2027, 1, 1})));
    CHECK((nextDay(CalendarDate{2024, 2, 28}) == (CalendarDate{2024, 2, 29})));
}

void test_compareCalendarDates() {
    CHECK((CalendarDate{2026, 9, 1} < CalendarDate{2026, 9, 2}));
    CHECK((CalendarDate{2025, 12, 31} < CalendarDate{2026, 1, 1}));
    CHECK((!(CalendarDate{2026, 9, 1} < CalendarDate{2026, 9, 1})));
}

void test_formatCalendarDate_zeroPads() {
    CHECK(formatCalendarDate((CalendarDate{2026, 9, 1})) == "2026/09/01");
    CHECK(formatCalendarDate((CalendarDate{2026, 12, 31})) == "2026/12/31");
}

void test_localMidnight_isStartOfThatLocalDay() {
    // Pin TZ so the expected epoch doesn't depend on the test machine.
    const char* previous = std::getenv("TZ");
    const std::string saved = previous ? previous : "";
    setenv("TZ", "Europe/Madrid", 1);
    tzset();
    // 2026-09-01 00:00 CEST (UTC+2) == 2026-08-31 22:00:00 UTC == 1788213600
    CHECK(localMidnight((CalendarDate{2026, 9, 1})) == static_cast<std::time_t>(1788213600));
    // 2026-01-15 00:00 CET (UTC+1) == 2026-01-14 23:00:00 UTC == 1768431600
    CHECK(localMidnight((CalendarDate{2026, 1, 15})) == static_cast<std::time_t>(1768431600));
    if (previous) {
        setenv("TZ", saved.c_str(), 1);
    } else {
        unsetenv("TZ");
    }
    tzset();
}

} // namespace

int main() {
    test_daysInMonth_handlesLeapYears();
    test_weekdayMondayFirst_knownDates();
    test_addMonths_wrapsYearsAndClampsDay();
    test_nextDay_crossesMonthAndYear();
    test_compareCalendarDates();
    test_formatCalendarDate_zeroPads();
    test_localMidnight_isStartOfThatLocalDay();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

#include "manager/calendar_date.hpp"

#include <cstdio>

int daysInMonth(int year, int month) {
    static constexpr int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2) {
        const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        return leap ? 29 : 28;
    }
    return kDays[month - 1];
}

int weekdayMondayFirst(const CalendarDate& date) {
    static constexpr int kOffsets[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    const int y = date.month < 3 ? date.year - 1 : date.year;
    const int sundayFirst = (y + y / 4 - y / 100 + y / 400 + kOffsets[date.month - 1] + date.day) % 7;
    return (sundayFirst + 6) % 7;
}

CalendarDate addMonths(const CalendarDate& date, int deltaMonths) {
    int monthIndex = date.year * 12 + (date.month - 1) + deltaMonths;
    CalendarDate out;
    out.year = monthIndex / 12;
    out.month = monthIndex % 12 + 1;
    const int maxDay = daysInMonth(out.year, out.month);
    out.day = date.day > maxDay ? maxDay : date.day;
    return out;
}

CalendarDate nextDay(const CalendarDate& date) {
    CalendarDate out = date;
    if (out.day < daysInMonth(out.year, out.month)) {
        out.day++;
        return out;
    }
    out.day = 1;
    if (out.month < 12) {
        out.month++;
    } else {
        out.month = 1;
        out.year++;
    }
    return out;
}

std::string formatCalendarDate(const CalendarDate& date) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04d/%02d/%02d", date.year, date.month, date.day);
    return buffer;
}

std::time_t localMidnight(const CalendarDate& date) {
    std::tm tm{};
    tm.tm_year = date.year - 1900;
    tm.tm_mon = date.month - 1;
    tm.tm_mday = date.day;
    tm.tm_isdst = -1;   // let mktime decide DST for this date from the system's timezone rules
    return std::mktime(&tm);
}

CalendarDate localCalendarDate(std::time_t t) {
    std::tm local{};
    localtime_r(&t, &local);
    return CalendarDate{local.tm_year + 1900, local.tm_mon + 1, local.tm_mday};
}

CalendarDate todayLocal() {
    return localCalendarDate(std::time(nullptr));
}

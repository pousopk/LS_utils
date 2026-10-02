#include "manager/item_filters.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

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

// 2026-09-10T12:00:00Z; every test runs with TZ=UTC (set in main), so
// typed "local" timestamps equal UTC.
constexpr std::time_t kNoon = 1789041600;

void test_confidence_noneAlwaysPasses() {
    const ConfidenceFilter f;
    CHECK(f.passes(0.1f));
    CHECK(f.passes(std::nullopt));
}

void test_confidence_strictThresholds() {
    ConfidenceFilter f;
    f.threshold = 0.5f;
    f.mode = ThresholdMode::LessThan;
    CHECK(f.passes(0.49f));
    CHECK(!f.passes(0.5f));
    f.mode = ThresholdMode::GreaterThan;
    CHECK(f.passes(0.51f));
    CHECK(!f.passes(0.5f));
}

void test_confidence_activeFilterRejectsMissingValue() {
    ConfidenceFilter f;
    f.mode = ThresholdMode::GreaterThan;
    f.threshold = 0.0f;
    CHECK(!f.passes(std::nullopt));
}

void test_presence() {
    PresenceFilter f;
    CHECK(f.passes(true) && f.passes(false));
    f.presence = Presence::Has;
    CHECK(f.passes(true) && !f.passes(false));
    f.presence = Presence::Lacks;
    CHECK(!f.passes(true) && f.passes(false));
}

void test_class_emptyPassesExactMatchOnly() {
    ClassFilter f;
    CHECK(f.passes(std::vector<std::string>{}));
    CHECK(f.passes(std::string_view("anything")));
    f.className = "Cat";
    CHECK(f.passes(std::vector<std::string>{"Dog", "Cat"}));
    CHECK(!f.passes(std::vector<std::string>{"Cats"}));
    CHECK(!f.passes(std::vector<std::string>{"cat"}));
    CHECK(f.passes(std::string_view("Cat")));
    CHECK(!f.passes(std::string_view("Ca")));
}

void test_textSearch_caseInsensitiveSubstringOnFileName() {
    TextSearch s;
    CHECK(s.passes("whatever.png"));
    s.text = "IMG_01";
    CHECK(s.passes("img_0123.jpg"));
    CHECK(!s.passes("img_02.jpg"));
    // Matches the file-name part only, like fileNameMatchesFilter.
    s.text = "folder";
    CHECK(!s.passes("folder/a.jpg"));
}

void test_timeWindow_offResolvesToNothing() {
    const TimeWindowFilter f;
    CHECK(!f.resolve().has_value());
    CHECK(describeTimeWindowProblem(f).empty());
}

constexpr CalendarDate kDay{2026, 9, 10};

void test_timeWindow_aroundTime() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::AroundTime;
    f.centerDate = kDay;
    f.centerTime = "12:00:00";
    f.toleranceMinutes = 2;
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->start == kNoon - 120);
    CHECK(w->end == kNoon + 120);
    CHECK(w->center == kNoon);
    CHECK(describeTimeWindowProblem(f).empty());
}

void test_timeWindow_aroundTimeAcceptsHoursAndMinutes() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::AroundTime;
    f.centerDate = kDay;
    f.centerTime = "12:00";
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->center == kNoon);
}

void test_timeWindow_negativeToleranceClampedToZero() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::AroundTime;
    f.centerDate = kDay;
    f.centerTime = "12:00:00";
    f.toleranceMinutes = -5;
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->start == kNoon);
    CHECK(w->end == kNoon);
}

void test_timeWindow_aroundTimeNeedsDateAndValidTime() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::AroundTime;
    f.centerTime = "12:00:00";
    CHECK(!f.resolve().has_value());   // no date picked
    CHECK(!describeTimeWindowProblem(f).empty());
    f.centerDate = kDay;
    for (const char* bad : {"", "12", "12:0", "12:00:0", "24:00", "12:60", "12:00:60", "12:00x", "ab:cd"}) {
        f.centerTime = bad;
        CHECK(!f.resolve().has_value());
        CHECK(!describeTimeWindowProblem(f).empty());
    }
}

void test_timeWindow_rangeOpenSides() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromDate = kDay;
    f.fromTime = "12:00:00";
    auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->start == kNoon);
    CHECK(!w->end.has_value());
    CHECK(!w->center.has_value());

    f.fromDate.reset();
    f.toDate = kDay;
    f.toTime = "12:00:00";
    w = f.resolve();
    CHECK(w.has_value());
    CHECK(!w->start.has_value());
    CHECK(w->end == kNoon);
}

void test_timeWindow_rangeBlankTimesCoverWholeDays() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromDate = kDay;
    f.toDate = kDay;
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->start == kNoon - 12 * 3600);
    CHECK(w->end == kNoon + 12 * 3600 - 1);
}

void test_timeWindow_rangeToWithoutSecondsIncludesWholeMinute() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromDate = kDay;
    f.fromTime = "12:00";
    f.toDate = kDay;
    f.toTime = "12:00";
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(w->start == kNoon);
    CHECK(w->end == kNoon + 59);
}

void test_timeWindow_rangeTimeWithoutDateIsIgnored() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromTime = "12:00";   // no From date -> open start
    f.toDate = kDay;
    const auto w = f.resolve();
    CHECK(w.has_value());
    CHECK(!w->start.has_value());
    CHECK(w->end == kNoon + 12 * 3600 - 1);
}

void test_timeWindow_rangeBothEmptyIsInactive() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    CHECK(!f.resolve().has_value());
    CHECK(!describeTimeWindowProblem(f).empty());
}

void test_timeWindow_rangeFromAfterToIsInactive() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromDate = kDay;
    f.fromTime = "13:00:00";
    f.toDate = kDay;
    f.toTime = "12:00:00";
    CHECK(!f.resolve().has_value());
    CHECK(describeTimeWindowProblem(f) == "From is after To");
}

void test_timeWindow_rangeOneSideUnparseableIsInactive() {
    TimeWindowFilter f;
    f.mode = TimeWindowMode::Range;
    f.fromDate = kDay;
    f.toDate = kDay;
    f.toTime = "garbage";
    CHECK(!f.resolve().has_value());
    CHECK(!describeTimeWindowProblem(f).empty());
}

void test_passesTimeWindow() {
    CHECK(passesTimeWindow(std::nullopt, std::nullopt));
    CHECK(passesTimeWindow(std::nullopt, kNoon));
    const std::optional<TimeWindow> w = TimeWindow{kNoon - 10, kNoon + 10, kNoon};
    CHECK(passesTimeWindow(w, kNoon - 10));   // inclusive
    CHECK(passesTimeWindow(w, kNoon + 10));   // inclusive
    CHECK(!passesTimeWindow(w, kNoon + 11));
    CHECK(!passesTimeWindow(w, std::nullopt));
    const std::optional<TimeWindow> openEnd = TimeWindow{kNoon, std::nullopt, std::nullopt};
    CHECK(passesTimeWindow(openEnd, kNoon + 1000000));
    CHECK(!passesTimeWindow(openEnd, kNoon - 1));
}

using Row = std::pair<int, std::optional<float>>;
const auto kRowKey = [](const Row& r) { return r.second; };

std::vector<int> ids(const std::vector<Row>& rows) {
    std::vector<int> out;
    for (const auto& r : rows) {
        out.push_back(r.first);
    }
    return out;
}

void test_sortByConfidence_noneKeepsOrder() {
    std::vector<Row> rows = {{1, 0.9f}, {2, 0.1f}};
    sortByConfidence(rows, ConfidenceSort::None, kRowKey);
    CHECK(ids(rows) == (std::vector<int>{1, 2}));
}

void test_sortByConfidence_directionsAndMissingKeys() {
    std::vector<Row> rows = {{1, 0.5f}, {2, std::nullopt}, {3, 0.1f}, {4, 0.9f}};
    sortByConfidence(rows, ConfidenceSort::Ascending, kRowKey);
    CHECK(ids(rows) == (std::vector<int>{2, 3, 1, 4}));
    sortByConfidence(rows, ConfidenceSort::Descending, kRowKey);
    CHECK(ids(rows) == (std::vector<int>{4, 1, 3, 2}));
}

void test_sortByConfidence_stableForEqualKeys() {
    std::vector<Row> rows = {{1, 0.5f}, {2, 0.5f}, {3, 0.5f}, {4, 0.2f}};
    sortByConfidence(rows, ConfidenceSort::Ascending, kRowKey);
    CHECK(ids(rows) == (std::vector<int>{4, 1, 2, 3}));
    sortByConfidence(rows, ConfidenceSort::Descending, kRowKey);
    CHECK(ids(rows) == (std::vector<int>{1, 2, 3, 4}));
}

} // namespace

int main() {
    setenv("TZ", "UTC", 1);
    tzset();

    test_confidence_noneAlwaysPasses();
    test_confidence_strictThresholds();
    test_confidence_activeFilterRejectsMissingValue();
    test_presence();
    test_class_emptyPassesExactMatchOnly();
    test_textSearch_caseInsensitiveSubstringOnFileName();
    test_timeWindow_offResolvesToNothing();
    test_timeWindow_aroundTime();
    test_timeWindow_aroundTimeAcceptsHoursAndMinutes();
    test_timeWindow_negativeToleranceClampedToZero();
    test_timeWindow_aroundTimeNeedsDateAndValidTime();
    test_timeWindow_rangeOpenSides();
    test_timeWindow_rangeBlankTimesCoverWholeDays();
    test_timeWindow_rangeToWithoutSecondsIncludesWholeMinute();
    test_timeWindow_rangeTimeWithoutDateIsIgnored();
    test_timeWindow_rangeBothEmptyIsInactive();
    test_timeWindow_rangeFromAfterToIsInactive();
    test_timeWindow_rangeOneSideUnparseableIsInactive();
    test_passesTimeWindow();
    test_sortByConfidence_noneKeepsOrder();
    test_sortByConfidence_directionsAndMissingKeys();
    test_sortByConfidence_stableForEqualKeys();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

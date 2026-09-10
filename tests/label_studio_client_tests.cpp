#include "manager/label_studio_client.hpp"

#include <cstdio>

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

void test_parseIso8601Utc_standardFormat() {
    const auto result = parseIso8601Utc("2024-05-01T12:34:56Z");
    CHECK(result.has_value());
    CHECK(*result == 1714566896);
}

void test_parseIso8601Utc_withFractionalSecondsAndZ() {
    const auto result = parseIso8601Utc("2024-05-01T12:34:56.789012Z");
    CHECK(result.has_value());
    CHECK(*result == 1714566896);
}

void test_parseIso8601Utc_malformedReturnsNullopt() {
    CHECK(!parseIso8601Utc("not a timestamp").has_value());
    CHECK(!parseIso8601Utc("").has_value());
}

void test_parseTypedUtcTimestamp_standardFormat() {
    const auto result = parseTypedUtcTimestamp("2026/09/10 12:00:00");
    CHECK(result.has_value());
    CHECK(*result == 1789041600);
}

void test_parseTypedUtcTimestamp_malformedReturnsNullopt() {
    CHECK(!parseTypedUtcTimestamp("2026-09-10 12:00:00").has_value()); // wrong separators
    CHECK(!parseTypedUtcTimestamp("garbage").has_value());
    CHECK(!parseTypedUtcTimestamp("").has_value());
}

void test_matchTasksToTimestamps_withinToleranceAndSorted() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}},
        {"id": 2, "created_at": "2026-09-10T12:01:30Z", "data": {"image": "/a/2.jpg"}},
        {"id": 3, "created_at": "2026-09-10T12:05:00Z", "data": {"image": "/a/3.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600; // 2026-09-10T12:00:00Z
    query.toleranceSeconds = 120; // +/- 2 min

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results.size() == 1);
    CHECK(results[0].size() == 2); // task 3 (5 min away) excluded
    CHECK(results[0][0].taskId == 1); // closest first
    CHECK(results[0][0].deltaSeconds == 0);
    CHECK(results[0][1].taskId == 2);
    CHECK(results[0][1].deltaSeconds == 90);
}

void test_matchTasksToTimestamps_toleranceBoundaryInclusive() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:02:00Z", "data": {"image": "/a/1.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600;
    query.toleranceSeconds = 120;

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results[0].size() == 1); // exactly at the boundary -- included
}

void test_matchTasksToTimestamps_skipsTaskMissingCreatedAtOrDataKey() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "data": {"image": "/a/1.jpg"}},
        {"id": 2, "created_at": "2026-09-10T12:00:00Z", "data": {}},
        {"id": 3, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/3.jpg"}}
    ])");

    TimestampMatchQuery query;
    query.timestamp = 1789041600;
    query.toleranceSeconds = 60;

    const auto results = matchTasksToTimestamps(tasks, "image", {query});
    CHECK(results[0].size() == 1);
    CHECK(results[0][0].taskId == 3);
}

void test_matchTasksToTimestamps_taskMatchesMultipleQueriesIndependently() {
    const auto tasks = nlohmann::json::parse(R"([
        {"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}}
    ])");

    TimestampMatchQuery queryA;
    queryA.timestamp = 1789041600;
    queryA.toleranceSeconds = 60;
    TimestampMatchQuery queryB;
    queryB.timestamp = 1789041630; // 30s later, overlapping window
    queryB.toleranceSeconds = 60;

    const auto results = matchTasksToTimestamps(tasks, "image", {queryA, queryB});
    CHECK(results.size() == 2);
    CHECK(results[0].size() == 1);
    CHECK(results[1].size() == 1);
    CHECK(results[0][0].taskId == 1);
    CHECK(results[1][0].taskId == 1);
}

void test_matchTasksToTimestamps_emptyQueriesReturnsEmpty() {
    const auto tasks = nlohmann::json::parse(R"([{"id": 1, "created_at": "2026-09-10T12:00:00Z", "data": {"image": "/a/1.jpg"}}])");
    CHECK(matchTasksToTimestamps(tasks, "image", {}).empty());
}

} // namespace

int main() {
    test_parseIso8601Utc_standardFormat();
    test_parseIso8601Utc_withFractionalSecondsAndZ();
    test_parseIso8601Utc_malformedReturnsNullopt();
    test_parseTypedUtcTimestamp_standardFormat();
    test_parseTypedUtcTimestamp_malformedReturnsNullopt();
    test_matchTasksToTimestamps_withinToleranceAndSorted();
    test_matchTasksToTimestamps_toleranceBoundaryInclusive();
    test_matchTasksToTimestamps_skipsTaskMissingCreatedAtOrDataKey();
    test_matchTasksToTimestamps_taskMatchesMultipleQueriesIndependently();
    test_matchTasksToTimestamps_emptyQueriesReturnsEmpty();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

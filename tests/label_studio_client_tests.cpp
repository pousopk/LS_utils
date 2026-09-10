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

} // namespace

int main() {
    test_parseIso8601Utc_standardFormat();
    test_parseIso8601Utc_withFractionalSecondsAndZ();
    test_parseIso8601Utc_malformedReturnsNullopt();
    test_parseTypedUtcTimestamp_standardFormat();
    test_parseTypedUtcTimestamp_malformedReturnsNullopt();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

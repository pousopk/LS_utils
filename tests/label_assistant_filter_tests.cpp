#include "manager/label_assistant_state.hpp"

#include <cstdio>
#include <string>
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

const std::vector<LabelAssistantImageEntry> kEntries = {
    {"b_002.jpg", 0.9f, "cat"},
    {"a_001.jpg", 0.3f, "dog"},
    {"c_003.jpg", 0.3f, "cat"},
    {"B_004.jpg", 0.6f, "dog"},
};

std::vector<std::string> names(const std::vector<const LabelAssistantImageEntry*>& entries) {
    std::vector<std::string> out;
    for (const auto* entry : entries) {
        out.push_back(entry->filename);
    }
    return out;
}

void test_defaultsKeepEverythingInOrder() {
    const auto result = filterLabelAssistantEntries(kEntries, TextSearch{}, ConfidenceFilter{}, ConfidenceSort::None);
    CHECK(names(result) == (std::vector<std::string>{"b_002.jpg", "a_001.jpg", "c_003.jpg", "B_004.jpg"}));
}

void test_filenameSearchIsCaseInsensitive() {
    const auto result = filterLabelAssistantEntries(kEntries, TextSearch{"b_"}, ConfidenceFilter{}, ConfidenceSort::None);
    CHECK(names(result) == (std::vector<std::string>{"b_002.jpg", "B_004.jpg"}));
}

void test_confidenceFilter() {
    const ConfidenceFilter below{ThresholdMode::LessThan, 0.5f};
    const auto result = filterLabelAssistantEntries(kEntries, TextSearch{}, below, ConfidenceSort::None);
    CHECK(names(result) == (std::vector<std::string>{"a_001.jpg", "c_003.jpg"}));
}

void test_sortIsStableForEqualConfidence() {
    const auto ascending = filterLabelAssistantEntries(kEntries, TextSearch{}, ConfidenceFilter{}, ConfidenceSort::Ascending);
    CHECK(names(ascending) == (std::vector<std::string>{"a_001.jpg", "c_003.jpg", "B_004.jpg", "b_002.jpg"}));
    const auto descending = filterLabelAssistantEntries(kEntries, TextSearch{}, ConfidenceFilter{}, ConfidenceSort::Descending);
    CHECK(names(descending) == (std::vector<std::string>{"b_002.jpg", "B_004.jpg", "a_001.jpg", "c_003.jpg"}));
}

} // namespace

int main() {
    test_defaultsKeepEverythingInOrder();
    test_filenameSearchIsCaseInsensitive();
    test_confidenceFilter();
    test_sortIsStableForEqualConfidence();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

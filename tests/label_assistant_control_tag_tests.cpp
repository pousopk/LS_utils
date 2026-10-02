#include "manager/label_assistant_control_tag.hpp"

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

void test_resolveLabelAssistantControlTag_findsRectangleLabelsForDetection() {
    const std::vector<LabelStudioControlTag> controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "choice_label", "image", {"Cat"}},
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "rect_label", "image", {"Cat", "Dog"}},
    };
    const auto resolution = resolveLabelAssistantControlTag(controlTags, ModelTask::Detection);
    CHECK(resolution.error.empty());
    CHECK(resolution.fromName == "rect_label");
    CHECK(resolution.toName == "image");
}

void test_resolveLabelAssistantControlTag_findsChoicesForClassification() {
    const std::vector<LabelStudioControlTag> controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "choice_label", "image", {"Cat"}},
    };
    const auto resolution = resolveLabelAssistantControlTag(controlTags, ModelTask::Classification);
    CHECK(resolution.error.empty());
    CHECK(resolution.fromName == "choice_label");
    CHECK(resolution.toName == "image");
}

void test_resolveLabelAssistantControlTag_setsErrorWhenNoMatchingTagType() {
    const std::vector<LabelStudioControlTag> controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::Choices, "choice_label", "image", {"Cat"}},
    };
    const auto resolution = resolveLabelAssistantControlTag(controlTags, ModelTask::Detection);
    CHECK(!resolution.error.empty());
    CHECK(resolution.fromName.empty());
    CHECK(resolution.toName.empty());
}

void test_resolveLabelAssistantControlTag_firstMatchingTagWinsWhenDuplicated() {
    const std::vector<LabelStudioControlTag> controlTags = {
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "first", "image", {}},
        LabelStudioControlTag{LabelStudioControlTagType::RectangleLabels, "second", "image", {}},
    };
    const auto resolution = resolveLabelAssistantControlTag(controlTags, ModelTask::Detection);
    CHECK(resolution.fromName == "first");
}

} // namespace

int main() {
    test_resolveLabelAssistantControlTag_findsRectangleLabelsForDetection();
    test_resolveLabelAssistantControlTag_findsChoicesForClassification();
    test_resolveLabelAssistantControlTag_setsErrorWhenNoMatchingTagType();
    test_resolveLabelAssistantControlTag_firstMatchingTagWinsWhenDuplicated();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

#include "manager/label_assistant_control_tag.hpp"

LabelAssistantControlTagResolution resolveLabelAssistantControlTag(
    const std::vector<LabelStudioControlTag>& controlTags, ComparisonTaskMode mode) {
    const LabelStudioControlTagType neededType = mode == ComparisonTaskMode::Detection
        ? LabelStudioControlTagType::RectangleLabels
        : LabelStudioControlTagType::Choices;

    for (const auto& tag : controlTags) {
        if (tag.type == neededType) {
            LabelAssistantControlTagResolution resolution;
            resolution.fromName = tag.name;
            resolution.toName = tag.toName;
            return resolution;
        }
    }

    LabelAssistantControlTagResolution resolution;
    resolution.error = mode == ComparisonTaskMode::Detection
        ? "This project has no RectangleLabels control tag."
        : "This project has no Choices control tag.";
    return resolution;
}

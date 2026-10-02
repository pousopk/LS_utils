#include "widgets/filter_widgets.hpp"

#include "widgets/date_picker.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>

namespace {

constexpr ImVec4 kErrorColor(1.0f, 0.4f, 0.4f, 1.0f);
constexpr float kTimeFieldWidth = 90.0f;

// The time box drawn after a date picker button on the same line.
bool drawTimeField(const char* id, const char* hint, std::string& text) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(kTimeFieldWidth);
    return ImGui::InputTextWithHint(id, hint, &text);
}

// "From [date] at [time]" on one line; a blank time box shows `timeHint`
// (the whole-day default).
bool drawRangeSide(
    const char* name, std::optional<CalendarDate>& date, std::string& time, const char* timeHint,
    const std::map<CalendarDate, int>* markedDays) {
    ImGui::PushID(name);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(name);
    ImGui::SameLine();
    bool changed = DatePickerButton("at", date, "Any", markedDays);
    changed |= drawTimeField("##Time", timeHint, time);
    ImGui::PopID();
    return changed;
}

} // namespace

bool drawConfidenceFilter(const char* label, ConfidenceFilter& filter) {
    bool changed = false;
    ImGui::PushID(label);
    static const char* kLabels[] = {"None", "< threshold", "> threshold"};
    int index = static_cast<int>(filter.mode);
    if (ImGui::Combo(label, &index, kLabels, IM_ARRAYSIZE(kLabels))) {
        filter.mode = static_cast<ThresholdMode>(index);
        changed = true;
    }
    ImGui::BeginDisabled(filter.mode == ThresholdMode::None);
    if (ImGui::SliderFloat("Threshold", &filter.threshold, 0.0f, 1.0f, "%.2f")) {
        changed = true;
    }
    ImGui::EndDisabled();
    ImGui::PopID();
    return changed;
}

bool drawPresenceFilter(
    const char* label, PresenceFilter& filter, const char* anyLabel, const char* hasLabel, const char* lacksLabel) {
    const char* labels[] = {anyLabel, hasLabel, lacksLabel};
    int index = static_cast<int>(filter.presence);
    if (ImGui::Combo(label, &index, labels, IM_ARRAYSIZE(labels))) {
        filter.presence = static_cast<Presence>(index);
        return true;
    }
    return false;
}

bool drawClassFilter(const char* label, ClassFilter& filter, const std::vector<std::string>& options) {
    bool changed = false;
    if (ImGui::BeginCombo(label, filter.className.empty() ? "All classes" : filter.className.c_str())) {
        if (ImGui::Selectable("All classes", filter.className.empty()) && !filter.className.empty()) {
            filter.className.clear();
            changed = true;
        }
        for (const auto& option : options) {
            if (ImGui::Selectable(option.c_str(), filter.className == option) && filter.className != option) {
                filter.className = option;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool drawTextSearch(const char* id, const char* hint, TextSearch& search) {
    return ImGui::InputTextWithHint(id, hint, &search.text);
}

bool drawConfidenceSort(const char* label, ConfidenceSort& sort, const char* noneLabel) {
    const char* labels[] = {noneLabel, "Confidence (low first)", "Confidence (high first)"};
    int index = static_cast<int>(sort);
    if (ImGui::Combo(label, &index, labels, IM_ARRAYSIZE(labels))) {
        sort = static_cast<ConfidenceSort>(index);
        return true;
    }
    return false;
}

bool drawTimeWindowFilter(const char* id, TimeWindowFilter& filter, const std::map<CalendarDate, int>* markedDays) {
    bool changed = false;
    ImGui::PushID(id);
    ImGui::TextUnformatted("Created");
    ImGui::SameLine();
    int mode = static_cast<int>(filter.mode);
    changed |= ImGui::RadioButton("Any time", &mode, static_cast<int>(TimeWindowMode::Off));
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Around", &mode, static_cast<int>(TimeWindowMode::AroundTime));
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Range", &mode, static_cast<int>(TimeWindowMode::Range));
    filter.mode = static_cast<TimeWindowMode>(mode);

    // DatePickerButton draws its label verbatim after the button, so each
    // reads "[date] at [time]"; PushID keeps From's and To's apart.
    if (filter.mode == TimeWindowMode::AroundTime) {
        changed |= DatePickerButton("at", filter.centerDate, "Pick date", markedDays);
        changed |= drawTimeField("##AroundTime", "HH:MM:SS", filter.centerTime);
        if (ImGui::InputInt("+/- minutes", &filter.toleranceMinutes)) {
            filter.toleranceMinutes = std::max(0, filter.toleranceMinutes);
            changed = true;
        }
    } else if (filter.mode == TimeWindowMode::Range) {
        changed |= drawRangeSide("From", filter.fromDate, filter.fromTime, "00:00:00", markedDays);
        changed |= drawRangeSide("To", filter.toDate, filter.toTime, "23:59:59", markedDays);
    }

    const std::string problem = describeTimeWindowProblem(filter);
    if (!problem.empty()) {
        ImGui::TextColored(kErrorColor, "%s -- time filter not applied", problem.c_str());
    }
    ImGui::PopID();
    return changed;
}

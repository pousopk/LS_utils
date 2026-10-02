#include "widgets/date_picker.hpp"

#include <imgui.h>

#include <map>

#include <string>

namespace {

constexpr const char* kMonthNames[12] = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
};
constexpr const char* kWeekdayHeaders[7] = {"Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"};
constexpr float kDayCellWidth = 26.0f;
// The theme's muted accent (ui_theme.cpp kAccentMuted).
constexpr ImVec4 kMarkedDayColor(0.902f, 0.451f, 0.086f, 0.35f);

} // namespace

bool DatePickerButton(
    const char* label, std::optional<CalendarDate>& value, const char* emptyText,
    const std::map<CalendarDate, int>* markedDays) {
    bool changed = false;
    ImGui::PushID(label);

    // The month currently shown in the popup, kept in ImGui's per-window
    // state storage (keyed under this picker's ID) so it survives across
    // frames while the popup is open.
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const ImGuiID viewYearId = ImGui::GetID("viewYear");
    const ImGuiID viewMonthId = ImGui::GetID("viewMonth");

    const std::string buttonText = value ? formatCalendarDate(*value) : std::string(emptyText);
    if (ImGui::Button((buttonText + "##open").c_str())) {
        const CalendarDate start = value ? *value : todayLocal();
        storage->SetInt(viewYearId, start.year);
        storage->SetInt(viewMonthId, start.month);
        ImGui::OpenPopup("calendar");
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(label);

    if (ImGui::BeginPopup("calendar")) {
        CalendarDate view{storage->GetInt(viewYearId, 1970), storage->GetInt(viewMonthId, 1), 1};

        if (ImGui::SmallButton("<<")) {
            view = addMonths(view, -12);
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("##prevMonth", ImGuiDir_Left)) {
            view = addMonths(view, -1);
        }
        ImGui::SameLine();
        ImGui::Text("%s %d", kMonthNames[view.month - 1], view.year);
        ImGui::SameLine();
        if (ImGui::ArrowButton("##nextMonth", ImGuiDir_Right)) {
            view = addMonths(view, 1);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(">>")) {
            view = addMonths(view, 12);
        }
        storage->SetInt(viewYearId, view.year);
        storage->SetInt(viewMonthId, view.month);

        const CalendarDate today = todayLocal();
        if (ImGui::BeginTable("days", 7, ImGuiTableFlags_SizingFixedFit)) {
            for (const char* header : kWeekdayHeaders) {
                ImGui::TableSetupColumn(header, ImGuiTableColumnFlags_WidthFixed, kDayCellWidth);
            }
            ImGui::TableHeadersRow();

            const int leadingBlanks = weekdayMondayFirst(view);
            const int dayCount = daysInMonth(view.year, view.month);
            ImGui::TableNextRow();
            for (int i = 0; i < leadingBlanks; ++i) {
                ImGui::TableNextColumn();
            }
            for (int day = 1; day <= dayCount; ++day) {
                ImGui::TableNextColumn();
                const CalendarDate cell{view.year, view.month, day};
                const bool isSelected = value && *value == cell;
                const bool isToday = cell == today;
                int markedCount = 0;
                if (markedDays != nullptr) {
                    const auto it = markedDays->find(cell);
                    if (it != markedDays->end()) {
                        markedCount = it->second;
                    }
                }
                if (markedCount > 0) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(kMarkedDayColor));
                }
                if (isToday) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.75f, 1.0f, 1.0f));
                }
                ImGui::PushID(day);
                if (ImGui::Selectable(std::to_string(day).c_str(), isSelected)) {
                    value = cell;
                    changed = true;
                    ImGui::CloseCurrentPopup();
                }
                if (markedCount > 0 && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%d picture(s)", markedCount);
                }
                ImGui::PopID();
                if (isToday) {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }
        if (markedDays != nullptr) {
            ImGui::TextDisabled(markedDays->empty() ? "No pictures loaded" : "Tinted days have pictures");
        }

        ImGui::Separator();
        if (ImGui::Button("Today")) {
            value = today;
            changed = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            if (value) {
                value.reset();
                changed = true;
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();
    return changed;
}

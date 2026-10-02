#include "widgets/label_studio_window.hpp"

#include "manager/label_studio_profiles_state.hpp"
#include "manager/labeling_state.hpp"
#include "widgets/ml_app_ui.hpp"
#include "widgets/dataset_browser_window.hpp"
#include "widgets/date_picker.hpp"
#include "widgets/labeling_window.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <string>

void drawSharedTaskRangeNote(const SharedLabelStudioProjectData& sharedData) {
    const std::string note = describeTaskImportDateRange(sharedData.appliedImportDateRange);
    if (note.empty()) {
        return;
    }
    ImGui::TextDisabled("%s", note.c_str());
    if (sharedData.droppedOutOfRangeTasks > 0) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
            "Label Studio ignored the date filter: downloading every task and filtering locally (%d skipped so far).",
            sharedData.droppedOutOfRangeTasks);
    }
}

void drawLabelStudioSessionSummary(
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow) {
    if (session.status == LabelStudioSessionStatus::Connected && session.activeProjectId > 0) {
        ImGui::Text("Label Studio: %s (id %d)", session.activeProjectTitle.c_str(), session.activeProjectId);
        ImGui::SameLine();
        if (ImGui::Button("Change##LabelStudioSession")) {
            onOpenLabelStudioWindow();
        }
    } else {
        ImGui::TextDisabled("Label Studio: not connected");
        ImGui::SameLine();
        if (ImGui::Button("Open Label Studio##LabelStudioSession")) {
            onOpenLabelStudioWindow();
        }
    }
}

namespace {

// The import-date range pickers. They edit pendingImportDateRange;
// "Apply range" (or picking a project below) makes it the range every
// load uses. Placed above the project list so a range can be chosen
// before the automatic load on project select starts.
void drawImportDateRangeControls(SharedLabelStudioProjectData& sharedData, const LabelStudioSessionState& session) {
    ImGui::SeparatorText("Import date range");
    TaskImportDateRange& pending = sharedData.pendingImportDateRange;
    DatePickerButton("Imported from", pending.from);
    ImGui::SameLine();
    DatePickerButton("Imported to", pending.to);

    const bool valid = isValidTaskImportDateRange(pending);
    const bool hasChanges = pending != sharedData.appliedImportDateRange;
    ImGui::BeginDisabled(!valid || !hasChanges);
    if (ImGui::Button("Apply range")) {
        if (setSharedTaskImportDateRange(sharedData, pending)) {
            // No-op until a project is selected and its config has loaded.
            refreshSharedLabelStudioProjectData(sharedData, session);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (!valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "\"Imported from\" is after \"Imported to\".");
    } else if (hasChanges) {
        ImGui::TextDisabled("(not applied yet)");
    } else {
        const std::string note = describeTaskImportDateRange(sharedData.appliedImportDateRange);
        ImGui::TextDisabled("%s", note.empty() ? "Whole project (no date limit)" : note.c_str());
    }
}

constexpr ImVec4 kErrorColor(1.0f, 0.4f, 0.4f, 1.0f);
constexpr ImVec4 kWarningColor(1.0f, 0.7f, 0.2f, 1.0f);

// "Save as..." modal: name field, and an Overwrite button (with a warning)
// in place of Save when the name is taken. Closes after any attempt; a
// failure shows under the profile row.
void drawSaveProfileAsPopup(LabelStudioProfilesState& profiles, const LabelStudioSessionState& session) {
    if (!ImGui::BeginPopupModal("Save profile as", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }
    if (ImGui::IsWindowAppearing()) {
        ImGui::SetKeyboardFocusHere();
    }
    ImGui::InputText("Name", &profiles.saveAsName);
    const std::string name = trimLabelStudioProfileName(profiles.saveAsName);
    const bool exists = findLabelStudioProfile(profiles.store, name) != nullptr;
    if (exists) {
        ImGui::TextColored(kWarningColor, "A profile named \"%s\" exists; saving overwrites it.", name.c_str());
    }
    ImGui::BeginDisabled(name.empty());
    if (ImGui::Button(exists ? "Overwrite" : "Save")) {
        saveLabelStudioProfileAs(profiles, name, session);
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void drawDeleteProfilePopup(LabelStudioProfilesState& profiles) {
    if (!ImGui::BeginPopupModal("Delete profile", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }
    ImGui::Text("Delete profile \"%s\"?", profiles.selectedName.c_str());
    ImGui::TextDisabled("The Base URL and API Token fields are kept.");
    if (ImGui::Button("Delete")) {
        deleteSelectedLabelStudioProfile(profiles);
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// Profile picker above the URL/token fields: choosing a profile fills them
// in (it doesn't connect); Save / Save as... / Delete write the profiles
// file right away. Load/save errors and warnings show underneath.
void drawProfileRow(LabelStudioProfilesState& profiles, LabelStudioSessionState& session) {
    const char* preview = profiles.selectedName.empty() ? "(none)" : profiles.selectedName.c_str();
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::BeginCombo("Profile", preview)) {
        if (ImGui::Selectable("(none)", profiles.selectedName.empty())) {
            selectLabelStudioProfile(profiles, "", session);
        }
        for (size_t i = 0; i < profiles.store.profiles.size(); ++i) {
            const LabelStudioProfile& profile = profiles.store.profiles[i];
            ImGui::PushID(static_cast<int>(i));   // names may contain "##"
            if (ImGui::Selectable(profile.name.c_str(), profile.name == profiles.selectedName)) {
                selectLabelStudioProfile(profiles, profile.name, session);
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    const bool editable = canEditLabelStudioProfiles(profiles);
    const bool hasFields = !session.baseUrl.empty() && !session.apiToken.empty();
    const bool hasSelection = !profiles.selectedName.empty();

    ImGui::SameLine();
    ImGui::BeginDisabled(
        !editable || !hasSelection || !hasFields || !selectedLabelStudioProfileDiffers(profiles, session));
    if (ImGui::Button("Save##Profile")) {
        saveSelectedLabelStudioProfile(profiles, session);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!editable || !hasFields);
    if (ImGui::Button("Save as...##Profile")) {
        profiles.saveAsName = profiles.selectedName;
        ImGui::OpenPopup("Save profile as");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!editable || !hasSelection);
    if (ImGui::Button("Delete##Profile")) {
        ImGui::OpenPopup("Delete profile");
    }
    ImGui::EndDisabled();

    drawSaveProfileAsPopup(profiles, session);
    drawDeleteProfilePopup(profiles);

    if (!profiles.error.empty()) {
        ImGui::TextColored(kErrorColor, "%s", profiles.error.c_str());
    }
    if (!profiles.warning.empty()) {
        ImGui::TextColored(kWarningColor, "%s", profiles.warning.c_str());
    }
}

// Content for the Connection tab: saved profiles, base URL/token, Connect,
// import date range, project list.
void drawConnectionTabContent(
    LabelStudioSessionState& session, LabelStudioProfilesState& profiles, SharedLabelStudioProjectData& sharedData) {
    drawProfileRow(profiles, session);
    ImGui::InputText("Base URL", &session.baseUrl);
    ImGui::InputText("API Token", &session.apiToken, ImGuiInputTextFlags_Password);

    const bool canConnect = !session.baseUrl.empty() && !session.apiToken.empty();
    ImGui::BeginDisabled(!canConnect);
    if (ImGui::Button("Connect")) {
        connectLabelStudioSession(session);
    }
    ImGui::EndDisabled();

    ImGui::Separator();

    switch (session.status) {
        case LabelStudioSessionStatus::Disconnected:
            ImGui::TextDisabled("Not connected.");
            break;
        case LabelStudioSessionStatus::Connecting:
            ImGui::TextDisabled("Connecting...");
            break;
        case LabelStudioSessionStatus::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", session.error.c_str());
            break;
        case LabelStudioSessionStatus::Connected: {
            drawImportDateRangeControls(sharedData, session);
            ImGui::Separator();
            ImGui::Text("Connected. %zu project(s):", session.projects.size());
            ImGui::BeginChild("LabelStudioProjectList", ImVec2(0, 200.0f), true);
            for (const auto& project : session.projects) {
                const bool isActive = project.id == session.activeProjectId;
                const std::string label = project.title + " (id " + std::to_string(project.id) + ")";
                if (ImGui::Selectable(label.c_str(), isActive)) {
                    // A range picked but not yet applied still counts for the
                    // load this selection triggers.
                    // A different project reloads on its own (updateSharedLabelStudioProjectData
                    // sees the key change); re-clicking the active one has to reload explicitly.
                    if (isValidTaskImportDateRange(sharedData.pendingImportDateRange)
                        && setSharedTaskImportDateRange(sharedData, sharedData.pendingImportDateRange) && isActive) {
                        refreshSharedLabelStudioProjectData(sharedData, session);
                    }
                    session.activeProjectId = project.id;
                    session.activeProjectTitle = project.title;
                }
            }
            ImGui::EndChild();
            break;
        }
    }
}

} // namespace

void drawLabelStudioTab(MlAppUi& ui, ImGuiTabItemFlags flags) {
    bool windowOpen = ui.showLabelStudioWindow;
    if (!windowOpen) {
        return;
    }

    const bool tabVisible = ImGui::BeginTabItem("Label Studio", &windowOpen, flags);

    // Closing the whole tab while the Labeling tab has unsaved edits is
    // the same hazard the standalone Labeling window used to guard against
    // -- veto the close and let its own unsaved-changes prompt (rendered
    // inside drawLabelingTabContent) decide what happens next.
    if (!windowOpen && anyEditorDirty(ui.labelingState)) {
        windowOpen = true;
        ui.labelingState.unsavedPromptOpen = true;
        ui.labelingState.unsavedPromptAction = LabelingUnsavedPromptAction::CloseWindow;
        // The prompt only renders inside the Labeling tab, so bring it forward.
        ui.openLabelStudioTab(LabelStudioTab::Labeling);
    }
    ui.showLabelStudioWindow = windowOpen;
    if (!tabVisible) {
        return;
    }

    auto onOpenConnectionTab = [&ui] { ui.openLabelStudioTab(LabelStudioTab::Connection); };

    if (ImGui::BeginTabBar("LabelStudioTabs")) {
        ImGuiTabItemFlags connectionFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::Connection) {
            connectionFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Connection", nullptr, connectionFlags)) {
            drawConnectionTabContent(ui.labelStudioSession, ui.labelStudioProfiles, ui.labelStudioProjectData);
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags labelingFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::Labeling) {
            labelingFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Labeling", nullptr, labelingFlags)) {
            drawLabelingTabContent(
                &ui.showLabelStudioWindow, ui.labelingState, ui.labelStudioSession, ui.labelStudioProjectData,
                onOpenConnectionTab);
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags datasetBrowserFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::DatasetBrowser) {
            datasetBrowserFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Dataset Browser", nullptr, datasetBrowserFlags)) {
            drawDatasetBrowserTabContent(
                ui.datasetBrowserState, ui.labelStudioSession, ui.labelStudioProjectData, onOpenConnectionTab);
            ImGui::EndTabItem();
        }

        ui.pendingLabelStudioTab.reset();
        ImGui::EndTabBar();
    }

    ImGui::EndTabItem();
}

#include "widgets/label_studio_window.hpp"

#include "manager/labeling_state.hpp"
#include "widgets/AppUi.hpp"
#include "widgets/dataset_browser_window.hpp"
#include "widgets/label_assistant_window.hpp"
#include "widgets/labeling_window.hpp"
#include "widgets/timestamp_search_window.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <string>

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

// Content for the Connection tab: base URL/token, Connect, project list.
void drawConnectionTabContent(LabelStudioSessionState& session) {
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
            ImGui::Text("Connected. %zu project(s):", session.projects.size());
            ImGui::BeginChild("LabelStudioProjectList", ImVec2(0, 200.0f), true);
            for (const auto& project : session.projects) {
                const bool isActive = project.id == session.activeProjectId;
                const std::string label = project.title + " (id " + std::to_string(project.id) + ")";
                if (ImGui::Selectable(label.c_str(), isActive)) {
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

void drawLabelStudioWindow(AppUi& ui) {
    bool windowOpen = ui.showLabelStudioWindow;
    if (!windowOpen) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(1200.0f, 900.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Label Studio", &windowOpen)) {
        ImGui::End();
        return;
    }

    // Closing the whole window while the Labeling tab has unsaved edits is
    // the same hazard the standalone Labeling window used to guard against
    // -- veto the close and let its own unsaved-changes prompt (rendered
    // inside drawLabelingTabContent) decide what happens next.
    if (!windowOpen && anyEditorDirty(ui.labelingState)) {
        windowOpen = true;
        ui.labelingState.unsavedPromptOpen = true;
        ui.labelingState.unsavedPromptAction = LabelingUnsavedPromptAction::CloseWindow;
    }
    ui.showLabelStudioWindow = windowOpen;

    auto onLabelTask = [&ui](int taskId) { ui.openLabelingForTask(taskId); };
    auto onOpenConnectionTab = [&ui] { ui.openLabelStudioTab(LabelStudioTab::Connection); };

    if (ImGui::BeginTabBar("LabelStudioTabs")) {
        ImGuiTabItemFlags connectionFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::Connection) {
            connectionFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Connection", nullptr, connectionFlags)) {
            drawConnectionTabContent(ui.labelStudioSession);
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

        ImGuiTabItemFlags labelAssistantFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::LabelAssistant) {
            labelAssistantFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Label Assistant", nullptr, labelAssistantFlags)) {
            drawLabelAssistantTabContent(
                ui.labelAssistantState, ui.labelStudioSession, ui.labelStudioProjectData, onLabelTask,
                onOpenConnectionTab);
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags timestampSearchFlags = ImGuiTabItemFlags_None;
        if (ui.pendingLabelStudioTab == LabelStudioTab::TimestampSearch) {
            timestampSearchFlags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Find by Timestamp", nullptr, timestampSearchFlags)) {
            drawTimestampSearchTabContent(
                ui.timestampSearchState, ui.labelStudioSession, ui.labelStudioProjectData, onLabelTask,
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

    ImGui::End();
}

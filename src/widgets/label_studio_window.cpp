#include "widgets/label_studio_window.hpp"

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

void drawLabelStudioWindow(bool* show, LabelStudioSessionState& session) {
    if (!*show) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(420.0f, 360.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Label Studio", show)) {
        ImGui::End();
        return;
    }

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

    ImGui::End();
}

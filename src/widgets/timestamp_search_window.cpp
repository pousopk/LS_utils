#include "widgets/timestamp_search_window.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_studio_client.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <ctime>

namespace {

void drawConnectionFields(TimestampSearchState& state) {
    ImGui::InputText("Label Studio URL", &state.labelStudioBaseUrl);
    ImGui::InputInt("Project ID", &state.labelStudioProjectId);
    ImGui::InputText("API Token", &state.labelStudioApiToken, ImGuiInputTextFlags_Password);
    if (!state.labelStudioAutoFetchStatus.empty()) {
        ImGui::TextDisabled("%s", state.labelStudioAutoFetchStatus.c_str());
    }
}

void drawEntryList(TimestampSearchState& state) {
    ImGui::InputInt("Tolerance (+/- minutes)", &state.toleranceMinutes);
    if (state.toleranceMinutes < 0) {
        state.toleranceMinutes = 0;
    }

    ImGui::InputTextWithHint("##NewTimestampEntry", "YYYY/MM/DD HH:MM:SS (local time)", &state.newEntryText);
    ImGui::SameLine();
    if (ImGui::Button("Add") && !state.newEntryText.empty()) {
        state.entries.push_back(TimestampSearchEntry{state.newEntryText});
        state.newEntryText.clear();
    }

    ImGui::BeginChild("TimestampSearchEntryList", ImVec2(0, 140.0f), true);
    int removeIndex = -1;
    for (int i = 0; i < static_cast<int>(state.entries.size()); ++i) {
        ImGui::PushID(i);
        const bool valid = parseTypedLocalTimestamp(state.entries[i].rawText).has_value();
        if (!valid) {
            ImGui::TextColored(
                ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s (invalid format)", state.entries[i].rawText.c_str());
        } else {
            ImGui::TextUnformatted(state.entries[i].rawText.c_str());
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Remove")) {
            removeIndex = i;
        }
        ImGui::PopID();
    }
    if (removeIndex >= 0) {
        state.entries.erase(state.entries.begin() + removeIndex);
    }
    ImGui::EndChild();
}

void drawSearchBar(TimestampSearchState& state) {
    ImGui::Separator();
    if (state.runState == TimestampSearchRunState::Running) {
        if (!state.lastProgress.phaseLabel.empty()) {
            ImGui::Text(
                "%s: %d / %d", state.lastProgress.phaseLabel.c_str(), state.lastProgress.completed,
                state.lastProgress.total);
        } else {
            ImGui::Text("%d / %d", state.lastProgress.completed, state.lastProgress.total);
        }
        const float fraction = state.lastProgress.total > 0
            ? static_cast<float>(state.lastProgress.completed) / static_cast<float>(state.lastProgress.total)
            : 0.0f;
        ImGui::ProgressBar(fraction);
        if (ImGui::Button("Cancel")) {
            state.worker.requestCancel();
        }
        return;
    }

    bool anyValidEntry = false;
    for (const auto& entry : state.entries) {
        if (parseTypedLocalTimestamp(entry.rawText).has_value()) {
            anyValidEntry = true;
            break;
        }
    }
    const bool canSearch = anyValidEntry && !state.labelStudioBaseUrl.empty() && state.labelStudioProjectId > 0
        && !state.labelStudioApiToken.empty();
    ImGui::BeginDisabled(!canSearch);
    if (ImGui::Button("Search")) {
        startTimestampSearch(state);
    }
    ImGui::EndDisabled();
    if (!canSearch) {
        ImGui::TextDisabled("Fill in the Label Studio connection and add at least one valid timestamp to search.");
    }
}

void drawResults(TimestampSearchState& state) {
    if (!state.resultError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.resultError.c_str());
        return;
    }

    for (const auto& group : state.resultGroups) {
        ImGui::Separator();
        ImGui::Text(
            "%s +/- %d min -- %d candidate(s)", group.entry.rawText.c_str(), state.toleranceMinutes,
            static_cast<int>(group.candidates.size()));

        if (group.candidates.empty()) {
            ImGui::TextDisabled("No candidates in this window.");
            continue;
        }

        for (const auto& candidateView : group.candidates) {
            ImGui::PushID(candidateView.candidate.taskId);
            if (candidateView.texture != 0) {
                const ImVec2 size =
                    fitImageToRegion(candidateView.textureWidth, candidateView.textureHeight, 160.0f, 160.0f);
                ImGui::Image((void*)(intptr_t)candidateView.texture, size);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    const ImVec2 largeSize =
                        fitImageToRegion(candidateView.textureWidth, candidateView.textureHeight, 480.0f, 480.0f);
                    ImGui::Image((void*)(intptr_t)candidateView.texture, largeSize);
                    ImGui::EndTooltip();
                }
            } else {
                ImGui::TextDisabled("(image unavailable)");
            }
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::Text("Task #%d", candidateView.candidate.taskId);
            char timeBuf[32];
            std::tm tm{};
            const std::time_t createdAt = candidateView.candidate.createdAt;
            gmtime_r(&createdAt, &tm);
            std::strftime(timeBuf, sizeof(timeBuf), "%Y/%m/%d %H:%M:%S", &tm);
            ImGui::Text("Created: %s UTC", timeBuf);
            ImGui::Text("Delta: %+lld sec", candidateView.candidate.deltaSeconds);
            ImGui::EndGroup();
            ImGui::PopID();
        }
    }
}

} // namespace

void drawTimestampSearchWindow(bool* show, TimestampSearchState& state) {
    if (!*show) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(700.0f, 800.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Find by Timestamp", show)) {
        ImGui::End();
        return;
    }

    drawConnectionFields(state);
    ImGui::Separator();
    drawEntryList(state);
    drawSearchBar(state);

    if (state.runState == TimestampSearchRunState::Complete) {
        drawResults(state);
    }

    ImGui::End();
}

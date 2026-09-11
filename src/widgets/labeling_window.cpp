#include "widgets/labeling_window.hpp"

#include "manager/app_runtime.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <opencv2/imgcodecs.hpp>

#include <filesystem>

namespace {

void ensureScratchFolder(LabelingState& state) {
    if (!state.scratchFolderPath.empty()) {
        return;
    }
    const std::string path = (std::filesystem::temp_directory_path() / "vision_app_labeling_download").string();
    std::filesystem::create_directories(path);
    state.scratchFolderPath = path;
}

void drawConnectionFields(LabelingState& state) {
    ImGui::InputText("Label Studio URL", &state.labelStudioBaseUrl);
    ImGui::InputInt("Project ID", &state.labelStudioProjectId);
    ImGui::InputText("API Token", &state.labelStudioApiToken, ImGuiInputTextFlags_Password);
    if (!state.configStatus.empty()) {
        ImGui::TextDisabled("%s", state.configStatus.c_str());
    }
}

void drawTaskListPanel(LabelingState& state) {
    ImGui::BeginChild("LabelingTaskList", ImVec2(220.0f, 0), true);

    const bool canList = !state.labelStudioBaseUrl.empty() && state.labelStudioProjectId > 0
        && !state.labelStudioApiToken.empty() && !state.projectConfig.dataImageKey.empty();
    ImGui::BeginDisabled(!canList || state.taskListLoading);
    if (ImGui::Button("Refresh task list", ImVec2(-1, 0))) {
        state.taskListLoading = true;
        LabelingJobRequest request;
        request.kind = LabelingJobKind::FetchTaskList;
        request.taskListJob.baseUrl = state.labelStudioBaseUrl;
        request.taskListJob.projectId = state.labelStudioProjectId;
        request.taskListJob.apiToken = state.labelStudioApiToken;
        request.taskListJob.dataImageKey = state.projectConfig.dataImageKey;
        state.worker.start(std::move(request));
    }
    ImGui::EndDisabled();

    if (!state.taskListError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.taskListError.c_str());
    }

    ImGui::Separator();
    for (const auto& task : state.taskList) {
        ImGui::PushID(task.taskId);
        const bool selected = state.selectedTaskId == task.taskId;
        const std::string label = "#" + std::to_string(task.taskId) + (task.hasAnnotation ? "  [labeled]" : "");
        if (ImGui::Selectable(label.c_str(), selected)) {
            requestSelectLabelingTask(state, task.taskId);
        }
        ImGui::PopID();
    }

    ImGui::EndChild();
}

void drawImageCanvas(LabelingState& state) {
    ImGui::BeginChild("LabelingCanvas", ImVec2(0, 0), true);
    if (state.taskLoadState == LabelingTaskLoadState::Loading) {
        ImGui::TextDisabled("Loading...");
    } else if (state.taskLoadState == LabelingTaskLoadState::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.taskLoadError.c_str());
    } else if (state.imageTexture != 0) {
        const ImVec2 size = fitImageToRegion(
            state.imageWidth, state.imageHeight, ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
        ImGui::Image((void*)(intptr_t)state.imageTexture, size);
        // Box-drawing/choice-picker overlays are added in Tasks 9-10.
    } else {
        ImGui::TextDisabled("Select a task to label.");
    }
    ImGui::EndChild();
}

} // namespace

void drawLabelingWindow(bool* show, LabelingState& state) {
    if (!*show) {
        return;
    }
    ensureScratchFolder(state);
    updateLabelingState(state);

    if (!state.pendingLocalImagePath.empty() && state.pendingLocalImagePath != state.loadedLocalImagePath) {
        const cv::Mat image = cv::imread(state.pendingLocalImagePath);
        if (!image.empty()) {
            uploadFrameToTexture(state.imageTexture, image, state.imageWidth, state.imageHeight);
        }
        state.loadedLocalImagePath = state.pendingLocalImagePath;
    }

    if (state.focusTaskId >= 0) {
        const int taskId = state.focusTaskId;
        state.focusTaskId = -1;
        requestSelectLabelingTask(state, taskId);
    }

    ImGui::SetNextWindowSize(ImVec2(1000.0f, 700.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Labeling", show)) {
        ImGui::End();
        return;
    }

    drawConnectionFields(state);
    ImGui::Separator();

    ImGui::BeginChild("LabelingBody", ImVec2(0, 0), false);
    drawTaskListPanel(state);
    ImGui::SameLine();
    drawImageCanvas(state);
    ImGui::EndChild();

    ImGui::End();
}

#include "widgets/labeling_window.hpp"

#include "manager/app_runtime.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
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

struct BoxDragState {
    bool active = false;
    bool creatingNew = false;
    int startX = 0;
    int startY = 0;
    int offsetX = 0;
    int offsetY = 0;
};

void drawBoxOverlay(const BoxLabelEditorState& editor, int imageWidth, int imageHeight) {
    if (imageWidth <= 0 || imageHeight <= 0) {
        return;
    }
    const ImVec2 imageMin = ImGui::GetItemRectMin();
    const ImVec2 imageMax = ImGui::GetItemRectMax();
    const float imageW = imageMax.x - imageMin.x;
    const float imageH = imageMax.y - imageMin.y;
    if (imageW <= 1.0f || imageH <= 1.0f) {
        return;
    }
    const float sx = imageW / static_cast<float>(imageWidth);
    const float sy = imageH / static_cast<float>(imageHeight);

    for (int i = 0; i < static_cast<int>(editor.boxes.size()); ++i) {
        const auto& box = editor.boxes[i].box;
        const ImVec2 r0(imageMin.x + sx * box.x, imageMin.y + sy * box.y);
        const ImVec2 r1(imageMin.x + sx * (box.x + box.width), imageMin.y + sy * (box.y + box.height));
        const ImU32 color = (i == editor.selectedBoxIndex) ? IM_COL32(255, 210, 60, 255) : IM_COL32(80, 200, 255, 255);
        ImGui::GetWindowDrawList()->AddRect(r0, r1, color, 0.0f, 0, 2.0f);
        ImGui::GetWindowDrawList()->AddText(ImVec2(r0.x, r0.y - 14.0f), color, editor.boxes[i].className.c_str());
    }
}

void handleBoxDrag(BoxLabelEditorState& editor, int imageWidth, int imageHeight) {
    if (imageWidth <= 0 || imageHeight <= 0) {
        return;
    }
    const ImVec2 imageMin = ImGui::GetItemRectMin();
    const ImVec2 imageMax = ImGui::GetItemRectMax();
    const float imageW = imageMax.x - imageMin.x;
    const float imageH = imageMax.y - imageMin.y;
    if (imageW <= 1.0f || imageH <= 1.0f) {
        return;
    }

    auto mapMouseToImage = [&](const ImVec2& mouse, int& outX, int& outY) {
        const float u = std::clamp((mouse.x - imageMin.x) / imageW, 0.0f, 1.0f);
        const float v = std::clamp((mouse.y - imageMin.y) / imageH, 0.0f, 1.0f);
        outX = std::clamp(static_cast<int>(u * static_cast<float>(imageWidth)), 0, imageWidth - 1);
        outY = std::clamp(static_cast<int>(v * static_cast<float>(imageHeight)), 0, imageHeight - 1);
    };

    static BoxDragState drag;
    const bool hovered = ImGui::IsItemHovered();

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        int startX = 0;
        int startY = 0;
        mapMouseToImage(ImGui::GetMousePos(), startX, startY);

        int hitIndex = -1;
        for (int i = 0; i < static_cast<int>(editor.boxes.size()); ++i) {
            if (editor.boxes[i].box.contains(cv::Point(startX, startY))) {
                hitIndex = i;
                break;
            }
        }

        if (hitIndex >= 0) {
            editor.selectedBoxIndex = hitIndex;
            drag.active = true;
            drag.creatingNew = false;
            drag.offsetX = startX - editor.boxes[hitIndex].box.x;
            drag.offsetY = startY - editor.boxes[hitIndex].box.y;
        } else if (!editor.availableLabels.empty()) {
            DraftDetectionBox newBox;
            newBox.box = cv::Rect(startX, startY, 1, 1);
            newBox.className = editor.availableLabels.front();
            editor.boxes.push_back(newBox);
            editor.selectedBoxIndex = static_cast<int>(editor.boxes.size()) - 1;
            editor.dirty = true;
            drag.active = true;
            drag.creatingNew = true;
            drag.startX = startX;
            drag.startY = startY;
        }
    }

    if (drag.active && editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size())) {
        int currentX = 0;
        int currentY = 0;
        mapMouseToImage(ImGui::GetMousePos(), currentX, currentY);
        cv::Rect& box = editor.boxes[editor.selectedBoxIndex].box;

        if (drag.creatingNew) {
            box.x = std::min(drag.startX, currentX);
            box.y = std::min(drag.startY, currentY);
            box.width = std::abs(currentX - drag.startX) + 1;
            box.height = std::abs(currentY - drag.startY) + 1;
        } else {
            box.x = std::clamp(currentX - drag.offsetX, 0, imageWidth - box.width);
            box.y = std::clamp(currentY - drag.offsetY, 0, imageHeight - box.height);
        }
        editor.dirty = true;

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            drag.active = false;
        }
    }
}

void drawBoxEditorPanel(BoxLabelEditorState& editor) {
    ImGui::Text("Boxes (%d)", static_cast<int>(editor.boxes.size()));
    if (editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size())) {
        auto& selected = editor.boxes[editor.selectedBoxIndex];
        if (ImGui::BeginCombo("Class", selected.className.c_str())) {
            for (const auto& label : editor.availableLabels) {
                const bool isSelected = label == selected.className;
                if (ImGui::Selectable(label.c_str(), isSelected)) {
                    selected.className = label;
                    editor.dirty = true;
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Delete selected box")) {
            editor.boxes.erase(editor.boxes.begin() + editor.selectedBoxIndex);
            editor.selectedBoxIndex = -1;
            editor.dirty = true;
        }
    } else {
        ImGui::TextDisabled("Click-drag on the image to draw a box; click an existing box to select it.");
    }
}

void drawChoiceEditorPanel(ChoiceLabelEditorState& editor) {
    ImGui::Text("Classification");
    for (const auto& label : editor.availableLabels) {
        const bool selected = editor.selectedLabel.has_value() && *editor.selectedLabel == label;
        if (ImGui::RadioButton(label.c_str(), selected)) {
            editor.selectedLabel = label;
            editor.dirty = true;
        }
    }
    if (editor.selectedLabel.has_value() && ImGui::SmallButton("Clear selection")) {
        editor.selectedLabel.reset();
        editor.dirty = true;
    }
}

void drawImageCanvas(LabelingState& state, float width) {
    ImGui::BeginChild("LabelingCanvas", ImVec2(width, 0), true);
    if (state.taskLoadState == LabelingTaskLoadState::Loading) {
        ImGui::TextDisabled("Loading...");
    } else if (state.taskLoadState == LabelingTaskLoadState::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.taskLoadError.c_str());
    } else if (state.imageTexture != 0) {
        const ImVec2 size = fitImageToRegion(
            state.imageWidth, state.imageHeight, ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
        ImGui::Image((void*)(intptr_t)state.imageTexture, size);
        if (state.boxEditor) {
            drawBoxOverlay(*state.boxEditor, state.imageWidth, state.imageHeight);
            handleBoxDrag(*state.boxEditor, state.imageWidth, state.imageHeight);
        }
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
            if (state.imageTexture == 0) {
                glGenTextures(1, &state.imageTexture);
            }
            uploadFrameToTexture(state.imageTexture, image, state.imageWidth, state.imageHeight);
        }
        state.loadedLocalImagePath = state.pendingLocalImagePath;
    }

    if (state.focusTaskId >= 0) {
        const int taskId = state.focusTaskId;
        state.focusTaskId = -1;
        requestSelectLabelingTask(state, taskId);
    }

    if (!state.submitInProgress && !state.unsavedPromptOpen
        && state.unsavedPromptAction == LabelingUnsavedPromptAction::CloseWindow && state.submitStatus == "Saved.") {
        *show = false;
        state.unsavedPromptAction = LabelingUnsavedPromptAction::None;
    }

    ImGui::SetNextWindowSize(ImVec2(1000.0f, 700.0f), ImGuiCond_FirstUseEver);
    bool windowOpen = *show;
    if (!ImGui::Begin("Labeling", &windowOpen)) {
        ImGui::End();
        return;
    }
    if (!windowOpen && anyEditorDirty(state)) {
        *show = true; // veto the close; the prompt below decides what happens next
        state.unsavedPromptOpen = true;
        state.unsavedPromptAction = LabelingUnsavedPromptAction::CloseWindow;
    } else if (!windowOpen) {
        *show = false;
    }

    drawConnectionFields(state);
    ImGui::Separator();

    ImGui::BeginChild("LabelingBody", ImVec2(0, 0), false);
    drawTaskListPanel(state);
    ImGui::SameLine();
    const bool hasEditorPanel = state.boxEditor || state.choiceEditor;
    const float editorPanelWidth = 240.0f;
    const float canvasWidth =
        hasEditorPanel ? ImGui::GetContentRegionAvail().x - editorPanelWidth - ImGui::GetStyle().ItemSpacing.x : 0.0f;
    drawImageCanvas(state, canvasWidth);
    if (hasEditorPanel) {
        ImGui::SameLine();
        ImGui::BeginChild("LabelingEditorPanel", ImVec2(editorPanelWidth, 0), true);
        if (state.boxEditor) {
            drawBoxEditorPanel(*state.boxEditor);
        }
        if (state.boxEditor && state.choiceEditor) {
            ImGui::Separator();
        }
        if (state.choiceEditor) {
            drawChoiceEditorPanel(*state.choiceEditor);
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::Separator();
    const bool canSubmit = state.selectedTaskId >= 0 && anyEditorDirty(state) && !state.submitInProgress;
    ImGui::BeginDisabled(!canSubmit);
    if (ImGui::Button("Submit")) {
        beginSubmitLabelingAnnotation(state);
    }
    ImGui::EndDisabled();
    if (!state.submitStatus.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", state.submitStatus.c_str());
    }

    if (state.unsavedPromptOpen) {
        ImGui::OpenPopup("Unsaved changes");
    }
    if (ImGui::BeginPopupModal("Unsaved changes", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("This task has unsaved edits.");
        if (ImGui::Button("Save")) {
            confirmSaveAndSwitchTask(state);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard")) {
            // confirmDiscardAndSwitchTask clears unsavedPromptAction, so capture
            // whether this was a pending window-close before calling it.
            const bool wasClosingWindow = state.unsavedPromptAction == LabelingUnsavedPromptAction::CloseWindow;
            confirmDiscardAndSwitchTask(state);
            if (wasClosingWindow) {
                *show = false;
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            state.unsavedPromptOpen = false;
            state.unsavedPromptAction = LabelingUnsavedPromptAction::None;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::End();
}

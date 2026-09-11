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

constexpr float kHandleScreenRadius = 7.0f;   // hit-test + draw radius, in screen pixels

// Corner handle order: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
constexpr int kHandleCount = 4;

ImVec2 handleScreenPos(int handleIndex, const ImVec2& r0, const ImVec2& r1) {
    switch (handleIndex) {
        case 0: return ImVec2(r0.x, r0.y);
        case 1: return ImVec2(r1.x, r0.y);
        case 2: return ImVec2(r0.x, r1.y);
        default: return ImVec2(r1.x, r1.y);
    }
}

// The box corner that stays fixed while `handleIndex` is dragged (the
// corner diagonally opposite it).
cv::Point handleAnchorPoint(int handleIndex, const cv::Rect& box) {
    switch (handleIndex) {
        case 0: return cv::Point(box.x + box.width, box.y + box.height);
        case 1: return cv::Point(box.x, box.y + box.height);
        case 2: return cv::Point(box.x + box.width, box.y);
        default: return cv::Point(box.x, box.y);
    }
}

ImU32 toImU32(const LabelColor& color) {
    return IM_COL32(color.r, color.g, color.b, 255);
}

struct BoxDragState {
    bool active = false;
    bool creatingNew = false;
    int resizeHandle = -1;   // -1 = not resizing; else 0-3, see handleScreenPos/handleAnchorPoint
    int startX = 0;
    int startY = 0;
    int offsetX = 0;
    int offsetY = 0;
    int anchorX = 0;
    int anchorY = 0;
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
        const bool isSelected = i == editor.selectedBoxIndex;
        const ImU32 color = toImU32(colorForClassName(editor.boxes[i].className));
        ImGui::GetWindowDrawList()->AddRect(r0, r1, color, 0.0f, 0, isSelected ? 3.0f : 2.0f);
        ImGui::GetWindowDrawList()->AddText(ImVec2(r0.x, r0.y - 14.0f), color, editor.boxes[i].className.c_str());

        if (isSelected) {
            for (int h = 0; h < kHandleCount; ++h) {
                const ImVec2 p = handleScreenPos(h, r0, r1);
                ImGui::GetWindowDrawList()->AddCircleFilled(p, kHandleScreenRadius, IM_COL32(255, 255, 255, 255));
                ImGui::GetWindowDrawList()->AddCircle(p, kHandleScreenRadius, IM_COL32(30, 30, 30, 255), 0, 2.0f);
            }
        }
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
    const float sx = imageW / static_cast<float>(imageWidth);
    const float sy = imageH / static_cast<float>(imageHeight);

    auto mapMouseToImage = [&](const ImVec2& mouse, int& outX, int& outY) {
        const float u = std::clamp((mouse.x - imageMin.x) / imageW, 0.0f, 1.0f);
        const float v = std::clamp((mouse.y - imageMin.y) / imageH, 0.0f, 1.0f);
        outX = std::clamp(static_cast<int>(u * static_cast<float>(imageWidth)), 0, imageWidth - 1);
        outY = std::clamp(static_cast<int>(v * static_cast<float>(imageHeight)), 0, imageHeight - 1);
    };

    static BoxDragState drag;
    const bool hovered = ImGui::IsItemHovered();

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImVec2 mouse = ImGui::GetMousePos();
        int startX = 0;
        int startY = 0;
        mapMouseToImage(mouse, startX, startY);

        int handleHit = -1;
        if (editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size())) {
            const auto& box = editor.boxes[editor.selectedBoxIndex].box;
            const ImVec2 r0(imageMin.x + sx * box.x, imageMin.y + sy * box.y);
            const ImVec2 r1(imageMin.x + sx * (box.x + box.width), imageMin.y + sy * (box.y + box.height));
            for (int h = 0; h < kHandleCount; ++h) {
                const ImVec2 p = handleScreenPos(h, r0, r1);
                const float dx = mouse.x - p.x;
                const float dy = mouse.y - p.y;
                if ((dx * dx + dy * dy) <= (kHandleScreenRadius * kHandleScreenRadius)) {
                    handleHit = h;
                    break;
                }
            }
        }

        if (handleHit >= 0) {
            const cv::Point anchor = handleAnchorPoint(handleHit, editor.boxes[editor.selectedBoxIndex].box);
            drag.active = true;
            drag.creatingNew = false;
            drag.resizeHandle = handleHit;
            drag.anchorX = anchor.x;
            drag.anchorY = anchor.y;
        } else {
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
                drag.resizeHandle = -1;
                drag.offsetX = startX - editor.boxes[hitIndex].box.x;
                drag.offsetY = startY - editor.boxes[hitIndex].box.y;
            } else if (!editor.availableLabels.empty()) {
                DraftDetectionBox newBox;
                newBox.box = cv::Rect(startX, startY, 1, 1);
                newBox.className =
                    !editor.pendingNewBoxLabel.empty() ? editor.pendingNewBoxLabel : editor.availableLabels.front();
                editor.boxes.push_back(newBox);
                editor.selectedBoxIndex = static_cast<int>(editor.boxes.size()) - 1;
                editor.dirty = true;
                drag.active = true;
                drag.creatingNew = true;
                drag.resizeHandle = -1;
                drag.startX = startX;
                drag.startY = startY;
            }
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
        } else if (drag.resizeHandle >= 0) {
            box.x = std::min(drag.anchorX, currentX);
            box.y = std::min(drag.anchorY, currentY);
            box.width = std::abs(currentX - drag.anchorX) + 1;
            box.height = std::abs(currentY - drag.anchorY) + 1;
        } else {
            box.x = std::clamp(currentX - drag.offsetX, 0, imageWidth - box.width);
            box.y = std::clamp(currentY - drag.offsetY, 0, imageHeight - box.height);
        }
        editor.dirty = true;

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            drag.active = false;
            drag.resizeHandle = -1;
        }
    }
}

void handleBoxEditorKeyboardShortcuts(BoxLabelEditorState& editor) {
    if (ImGui::IsAnyItemActive()) {
        return; // don't hijack number keys while typing in a text field
    }

    for (int i = 0; i < static_cast<int>(editor.availableLabels.size()) && i < 9; ++i) {
        if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(ImGuiKey_1 + i))) {
            const std::string& label = editor.availableLabels[i];
            if (editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size())) {
                editor.boxes[editor.selectedBoxIndex].className = label;
                editor.dirty = true;
            } else {
                editor.pendingNewBoxLabel = label;
            }
        }
    }

    if ((ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))
        && editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size())) {
        editor.boxes.erase(editor.boxes.begin() + editor.selectedBoxIndex);
        editor.selectedBoxIndex = -1;
        editor.dirty = true;
    }
}

// Draws one button per label, tinted by colorForClassName and numbered
// 1..N (matching the number-key shortcuts). Returns the label clicked
// this frame, or an empty string if none was. Shared by the box editor
// and the mask editor.
std::string drawLabelPickerButtons(const std::vector<std::string>& availableLabels) {
    std::string clicked;
    for (int i = 0; i < static_cast<int>(availableLabels.size()); ++i) {
        const std::string& label = availableLabels[i];
        const LabelColor color = colorForClassName(label);
        const std::string buttonText = std::to_string(i + 1) + ": " + label;

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, 0.65f));
        ImGui::PushStyleColor(
            ImGuiCol_ButtonHovered, ImVec4(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, 0.85f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, 1.0f));
        if (ImGui::Button(buttonText.c_str(), ImVec2(-1, 0))) {
            clicked = label;
        }
        ImGui::PopStyleColor(3);
    }
    return clicked;
}

void drawBoxEditorPanel(BoxLabelEditorState& editor) {
    ImGui::Text("Boxes (%d)", static_cast<int>(editor.boxes.size()));

    const bool hasSelection =
        editor.selectedBoxIndex >= 0 && editor.selectedBoxIndex < static_cast<int>(editor.boxes.size());
    ImGui::TextDisabled(hasSelection ? "Pick a label to reassign the selected box:" : "Pick a label for the next box:");

    const std::string clicked = drawLabelPickerButtons(editor.availableLabels);
    if (!clicked.empty()) {
        if (hasSelection) {
            editor.boxes[editor.selectedBoxIndex].className = clicked;
            editor.dirty = true;
        } else {
            editor.pendingNewBoxLabel = clicked;
        }
    }

    if (hasSelection) {
        if (ImGui::Button("Delete selected box")) {
            editor.boxes.erase(editor.boxes.begin() + editor.selectedBoxIndex);
            editor.selectedBoxIndex = -1;
            editor.dirty = true;
        }
    } else {
        ImGui::TextDisabled("Click-drag on the image to draw a box; click an existing box to select it.");
    }

    handleBoxEditorKeyboardShortcuts(editor);
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
        const ImVec2 imagePos = ImGui::GetCursorScreenPos();
        ImGui::Image((void*)(intptr_t)state.imageTexture, size);
        if (state.boxEditor) {
            // A plain Image() item doesn't capture the mouse the way an
            // active widget does, so a click-drag on it can fall through
            // to the window's own drag/focus handling instead of our own
            // drag logic below. Overlaying an invisible button (same
            // position/size as the image) makes ImGui treat the drag as
            // captured by this widget for its whole duration -- the
            // standard fix for "dragging on my custom canvas moves the
            // window instead."
            ImGui::SetCursorScreenPos(imagePos);
            ImGui::InvisibleButton("LabelingCanvasHitRegion", size);
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

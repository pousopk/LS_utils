#include "widgets/labeling_window.hpp"

#include "ui_common/gl_texture.hpp"
#include "ui_common/image_fit.hpp"
#include "widgets/label_studio_window.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
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

void drawTaskListPanel(LabelingState& state, const LabelStudioSessionState& session, SharedLabelStudioProjectData& sharedData) {
    ImGui::BeginChild("LabelingTaskList", ImVec2(220.0f, 0), true);

    const bool canList = !session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty()
        && !sharedData.projectConfig.dataImageKey.empty();
    ImGui::BeginDisabled(!canList || sharedData.loading);
    if (ImGui::Button("Refresh task list", ImVec2(-1, 0))) {
        refreshSharedLabelStudioProjectData(sharedData, session);
    }
    ImGui::EndDisabled();

    if (sharedData.loading) {
        ImGui::TextDisabled("%s", describeSharedTaskListLoadProgress(sharedData).c_str());
    }

    if (!sharedData.error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", sharedData.error.c_str());
    }
    drawSharedTaskRangeNote(sharedData);

    ImGui::Separator();
    for (const auto& task : state.taskList) {
        ImGui::PushID(task.taskId);
        const bool selected = state.selectedTaskId == task.taskId;
        const std::string label = "#" + std::to_string(task.taskId) + (task.hasAnnotation ? "  [labeled]" : "");
        if (ImGui::Selectable(label.c_str(), selected)) {
            requestSelectLabelingTask(state, session, task.taskId);
        }
        ImGui::PopID();
    }

    ImGui::EndChild();
}

constexpr float kHandleScreenRadius = 7.0f;   // hit-test + draw radius, in screen pixels

// Corner handle order: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
constexpr int kHandleCount = 4;

// Maps a handle index (0=TL,1=TR,2=BL,3=BR) to its position in
// rotatedBoxCorners' perimeter-walk order (TL,TR,BR,BL) -- the two orders
// agree everywhere except indices 2 and 3.
constexpr int kHandleToCornerIndex[kHandleCount] = {0, 1, 3, 2};

constexpr float kRotateHandleScreenOffset = 24.0f;   // screen pixels above the (rotated) top edge's midpoint
constexpr int kRotateHandleIndex = 4;                // distinct from the 4 resize handles (0-3)

// Screen-space position of the rotate handle: the midpoint of the top edge
// (screenCorners[0]..screenCorners[1], i.e. TL..TR) offset outward along
// that edge's own perpendicular, so the handle stays visually "attached"
// to the box as it rotates rather than always pointing straight up.
ImVec2 rotateHandleScreenPos(const ImVec2 screenCorners[4]) {
    const ImVec2 mid((screenCorners[0].x + screenCorners[1].x) * 0.5f, (screenCorners[0].y + screenCorners[1].y) * 0.5f);
    const ImVec2 edge(screenCorners[1].x - screenCorners[0].x, screenCorners[1].y - screenCorners[0].y);
    const float edgeLen = std::sqrt(edge.x * edge.x + edge.y * edge.y);
    if (edgeLen < 1.0f) {
        return ImVec2(mid.x, mid.y - kRotateHandleScreenOffset);
    }
    // Perpendicular to the edge, pointing away from the box (rotate -90 deg
    // in screen space: (ex,ey) -> (ey,-ex) points "up" for an unrotated,
    // left-to-right top edge -- verified visually in manual testing; flip
    // to (-ey, ex) if it ever points into the box instead.
    const ImVec2 perp(edge.y / edgeLen, -edge.x / edgeLen);
    return ImVec2(mid.x + perp.x * kRotateHandleScreenOffset, mid.y + perp.y * kRotateHandleScreenOffset);
}

ImU32 toImU32(const LabelColor& color) {
    return IM_COL32(color.r, color.g, color.b, 255);
}

struct BoxDragState {
    bool active = false;
    bool creatingNew = false;
    bool rotating = false;
    int resizeHandle = -1;   // -1 = not resizing; else 0-3, see kHandleToCornerIndex/rotatedHandleAnchorPoint
    int startX = 0;
    int startY = 0;
    int offsetX = 0;
    int offsetY = 0;
    float anchorImageX = 0.0f;         // resize anchor corner, image space (may be non-integer once rotated)
    float anchorImageY = 0.0f;
    float dragRotationDegrees = 0.0f;  // box's rotation, captured at resize-drag-start (fixed for the resize)
    cv::Rect rotateStartBox;           // box as of rotate-drag-start (fixed reference, not updated mid-drag)
    float rotateStartRotationDegrees = 0.0f;
    float rotateStartAngleRadians = 0.0f;   // atan2(mouse - center) at rotate-drag-start
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
        const float rotationDegrees = editor.boxes[i].rotationDegrees;
        const bool isSelected = i == editor.selectedBoxIndex;
        const ImU32 color = toImU32(colorForClassName(editor.boxes[i].className));
        const auto corners = rotatedBoxCorners(box, rotationDegrees);
        ImVec2 screenCorners[4];
        for (int c = 0; c < 4; ++c) {
            screenCorners[c] = ImVec2(imageMin.x + sx * corners[c].x, imageMin.y + sy * corners[c].y);
        }

        if (rotationDegrees == 0.0f) {
            ImGui::GetWindowDrawList()->AddRect(
                screenCorners[0], screenCorners[2], color, 0.0f, 0, isSelected ? 3.0f : 2.0f);
        } else {
            ImGui::GetWindowDrawList()->AddPolyline(
                screenCorners, 4, color, ImDrawFlags_Closed, isSelected ? 3.0f : 2.0f);
        }
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(screenCorners[0].x, screenCorners[0].y - 14.0f), color, editor.boxes[i].className.c_str());

        if (isSelected) {
            for (int h = 0; h < kHandleCount; ++h) {
                const ImVec2& handlePos = screenCorners[kHandleToCornerIndex[h]];
                ImGui::GetWindowDrawList()->AddCircleFilled(handlePos, kHandleScreenRadius, IM_COL32(255, 255, 255, 255));
                ImGui::GetWindowDrawList()->AddCircle(handlePos, kHandleScreenRadius, IM_COL32(30, 30, 30, 255), 0, 2.0f);
            }
            const ImVec2 rotateHandlePos = rotateHandleScreenPos(screenCorners);
            const ImVec2 topMid((screenCorners[0].x + screenCorners[1].x) * 0.5f, (screenCorners[0].y + screenCorners[1].y) * 0.5f);
            ImGui::GetWindowDrawList()->AddLine(topMid, rotateHandlePos, IM_COL32(255, 255, 255, 200), 1.5f);
            ImGui::GetWindowDrawList()->AddCircleFilled(rotateHandlePos, kHandleScreenRadius, IM_COL32(120, 200, 255, 255));
            ImGui::GetWindowDrawList()->AddCircle(rotateHandlePos, kHandleScreenRadius, IM_COL32(30, 30, 30, 255), 0, 2.0f);
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
            const auto& selectedBox = editor.boxes[editor.selectedBoxIndex];
            const auto corners = rotatedBoxCorners(selectedBox.box, selectedBox.rotationDegrees);
            ImVec2 screenCorners[4];
            for (int c = 0; c < 4; ++c) {
                screenCorners[c] = ImVec2(imageMin.x + sx * corners[c].x, imageMin.y + sy * corners[c].y);
            }
            for (int h = 0; h < kHandleCount; ++h) {
                const ImVec2& corner = screenCorners[kHandleToCornerIndex[h]];
                const float dx = mouse.x - corner.x;
                const float dy = mouse.y - corner.y;
                if ((dx * dx + dy * dy) <= (kHandleScreenRadius * kHandleScreenRadius)) {
                    handleHit = h;
                    break;
                }
            }
            if (handleHit < 0) {
                const ImVec2 rotateHandlePos = rotateHandleScreenPos(screenCorners);
                const float dx = mouse.x - rotateHandlePos.x;
                const float dy = mouse.y - rotateHandlePos.y;
                if ((dx * dx + dy * dy) <= (kHandleScreenRadius * kHandleScreenRadius)) {
                    handleHit = kRotateHandleIndex;
                }
            }
        }

        if (handleHit == kRotateHandleIndex) {
            const auto& selectedBox = editor.boxes[editor.selectedBoxIndex];
            const cv::Point2f localCenter(selectedBox.box.width / 2.0f, selectedBox.box.height / 2.0f);
            const cv::Point2f pivot(static_cast<float>(selectedBox.box.x), static_cast<float>(selectedBox.box.y));
            const cv::Point2f center = rotatePointClockwise(pivot + localCenter, pivot, selectedBox.rotationDegrees);
            drag.active = true;
            drag.creatingNew = false;
            drag.resizeHandle = -1;
            drag.rotating = true;
            drag.rotateStartBox = selectedBox.box;
            drag.rotateStartRotationDegrees = selectedBox.rotationDegrees;
            drag.rotateStartAngleRadians = std::atan2(
                static_cast<float>(startY) - center.y, static_cast<float>(startX) - center.x);
        } else if (handleHit >= 0) {
            const auto& selectedBox = editor.boxes[editor.selectedBoxIndex];
            const cv::Point2f anchor = rotatedHandleAnchorPoint(handleHit, selectedBox.box, selectedBox.rotationDegrees);
            drag.active = true;
            drag.creatingNew = false;
            drag.rotating = false;
            drag.resizeHandle = handleHit;
            drag.anchorImageX = anchor.x;
            drag.anchorImageY = anchor.y;
            drag.dragRotationDegrees = selectedBox.rotationDegrees;
        } else {
            int hitIndex = -1;
            for (int i = 0; i < static_cast<int>(editor.boxes.size()); ++i) {
                if (rotatedBoxContainsPoint(
                        editor.boxes[i].box, editor.boxes[i].rotationDegrees,
                        cv::Point2f(static_cast<float>(startX), static_cast<float>(startY)))) {
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
        } else if (drag.rotating) {
            const cv::Point2f localCenter(drag.rotateStartBox.width / 2.0f, drag.rotateStartBox.height / 2.0f);
            const cv::Point2f pivot(
                static_cast<float>(drag.rotateStartBox.x), static_cast<float>(drag.rotateStartBox.y));
            const cv::Point2f center =
                rotatePointClockwise(pivot + localCenter, pivot, drag.rotateStartRotationDegrees);
            const float currentAngle =
                std::atan2(static_cast<float>(currentY) - center.y, static_cast<float>(currentX) - center.x);
            const RotatedBoxAngleDrag rotated = rotateBoxAroundCenter(
                drag.rotateStartBox, drag.rotateStartRotationDegrees, drag.rotateStartAngleRadians, currentAngle);
            box = rotated.box;
            editor.boxes[editor.selectedBoxIndex].rotationDegrees = rotated.rotationDegrees;
        } else if (drag.resizeHandle >= 0) {
            box = resizeRotatedBox(
                drag.dragRotationDegrees, cv::Point2f(drag.anchorImageX, drag.anchorImageY),
                cv::Point2f(static_cast<float>(currentX), static_cast<float>(currentY)));
        } else {
            box.x = std::clamp(currentX - drag.offsetX, 0, imageWidth - box.width);
            box.y = std::clamp(currentY - drag.offsetY, 0, imageHeight - box.height);
        }
        editor.dirty = true;

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            drag.active = false;
            drag.resizeHandle = -1;
            drag.rotating = false;
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

struct MaskPaintDragState {
    bool active = false;
    int lastX = -1;
    int lastY = -1;
};

// Paints/erases into the currently selected mask region as the mouse
// drags over the canvas -- analogous to handleBoxDrag, but there's
// nothing to "select" on the image itself (mask regions are selected via
// the region list in the editor panel, see Task 7); this only paints.
// Sets `maskChanged` to true whenever a stroke actually modifies pixels,
// so the caller knows to recomposite and re-upload the display texture.
void handleMaskPaint(BrushLabelEditorState& editor, int imageWidth, int imageHeight, bool& maskChanged) {
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

    static MaskPaintDragState drag;
    const bool hovered = ImGui::IsItemHovered();
    const bool hasSelection =
        editor.selectedRegionIndex >= 0 && editor.selectedRegionIndex < static_cast<int>(editor.regions.size());

    if (hovered && ImGui::GetIO().MouseWheel != 0.0f) {
        editor.brushRadius = std::clamp(editor.brushRadius + ImGui::GetIO().MouseWheel * 2.0f, 2.0f, 60.0f);
    }

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hasSelection) {
        drag.active = true;
        drag.lastX = -1;
        drag.lastY = -1;
    }

    if (drag.active && hasSelection) {
        int currentX = 0;
        int currentY = 0;
        mapMouseToImage(ImGui::GetMousePos(), currentX, currentY);

        cv::Mat& mask = editor.regions[editor.selectedRegionIndex].mask;
        const cv::Scalar paintValue = editor.eraseMode ? cv::Scalar(0) : cv::Scalar(255);
        const int radius = std::max(1, static_cast<int>(editor.brushRadius));

        if (drag.lastX >= 0) {
            cv::line(mask, cv::Point(drag.lastX, drag.lastY), cv::Point(currentX, currentY), paintValue, radius * 2);
        }
        cv::circle(mask, cv::Point(currentX, currentY), radius, paintValue, -1);
        drag.lastX = currentX;
        drag.lastY = currentY;
        editor.dirty = true;
        maskChanged = true;

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            drag.active = false;
        }
    }
}

// Draws an outline circle at the current mouse position, sized to the
// brush radius, so the user can see how big a stroke will be before
// clicking -- shown whenever a mask region is selected (painting is only
// possible then) and the canvas is hovered. Green for paint, red for
// erase. Purely visual, no state change.
void drawMaskBrushCursor(const BrushLabelEditorState& editor, int imageWidth, int imageHeight) {
    const bool hasSelection =
        editor.selectedRegionIndex >= 0 && editor.selectedRegionIndex < static_cast<int>(editor.regions.size());
    if (imageWidth <= 0 || imageHeight <= 0 || !hasSelection || !ImGui::IsItemHovered()) {
        return;
    }

    const ImVec2 imageMin = ImGui::GetItemRectMin();
    const ImVec2 imageMax = ImGui::GetItemRectMax();
    const float imageW = imageMax.x - imageMin.x;
    if (imageW <= 1.0f) {
        return;
    }
    const float sx = imageW / static_cast<float>(imageWidth);

    const ImVec2 mouse = ImGui::GetMousePos();
    const float screenRadius = editor.brushRadius * sx;
    const ImU32 color = editor.eraseMode ? IM_COL32(255, 80, 80, 220) : IM_COL32(80, 255, 120, 220);
    ImGui::GetWindowDrawList()->AddCircle(mouse, screenRadius, color, 0, 2.0f);
}

void drawMaskEditorPanel(BrushLabelEditorState& editor, int imageWidth, int imageHeight, bool& maskChanged) {
    ImGui::Text("Masks (%d)", static_cast<int>(editor.regions.size()));

    for (int i = 0; i < static_cast<int>(editor.regions.size()); ++i) {
        ImGui::PushID(i);
        const bool selected = i == editor.selectedRegionIndex;
        const std::string label = std::to_string(i + 1) + ": " + editor.regions[i].className;
        if (ImGui::Selectable(label.c_str(), selected)) {
            editor.selectedRegionIndex = i;
        }
        ImGui::PopID();
    }

    const bool hasSelection =
        editor.selectedRegionIndex >= 0 && editor.selectedRegionIndex < static_cast<int>(editor.regions.size());
    ImGui::TextDisabled(hasSelection ? "Pick a label to reassign the selected mask:" : "Pick a label for the next mask:");

    const std::string clicked = drawLabelPickerButtons(editor.availableLabels);
    if (!clicked.empty()) {
        if (hasSelection) {
            editor.regions[editor.selectedRegionIndex].className = clicked;
            editor.dirty = true;
        } else {
            editor.pendingNewMaskLabel = clicked;
        }
    }

    if (ImGui::Button("New mask") && imageWidth > 0 && imageHeight > 0) {
        DraftBrushRegion region;
        region.className = !editor.pendingNewMaskLabel.empty()
            ? editor.pendingNewMaskLabel
            : (!editor.availableLabels.empty() ? editor.availableLabels.front() : std::string());
        region.mask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
        editor.regions.push_back(std::move(region));
        editor.selectedRegionIndex = static_cast<int>(editor.regions.size()) - 1;
        editor.dirty = true;
        maskChanged = true;
    }

    if (hasSelection) {
        ImGui::SameLine();
        if (ImGui::Button("Delete selected mask")) {
            editor.regions.erase(editor.regions.begin() + editor.selectedRegionIndex);
            editor.selectedRegionIndex = -1;
            editor.dirty = true;
            maskChanged = true;
        }
    }

    ImGui::Checkbox("Erase", &editor.eraseMode);
    ImGui::SliderFloat("Brush size", &editor.brushRadius, 2.0f, 60.0f);

    if (!hasSelection) {
        ImGui::TextDisabled("Select or create a mask, then paint on the image.");
    }
}

void drawImageCanvas(LabelingState& state, float width, bool& maskChanged) {
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
        if (state.boxEditor || state.maskEditor) {
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
            if (state.boxEditor) {
                drawBoxOverlay(*state.boxEditor, state.imageWidth, state.imageHeight);
                handleBoxDrag(*state.boxEditor, state.imageWidth, state.imageHeight);
            }
            if (state.maskEditor) {
                handleMaskPaint(*state.maskEditor, state.imageWidth, state.imageHeight, maskChanged);
                drawMaskBrushCursor(*state.maskEditor, state.imageWidth, state.imageHeight);
            }
        }
    } else {
        ImGui::TextDisabled("Select a task to label.");
    }
    ImGui::EndChild();
}

// Ctrl+Enter to submit and Ctrl+Left/Right to move between tasks, matching
// Label Studio's own annotation:submit/image:prev/image:next shortcuts.
// Gated on !IsAnyItemActive() so these don't fire while typing in the
// connection fields; the caller additionally gates this on the Labeling
// window having focus, so these don't hijack the shortcuts globally.
void handleLabelingWindowKeyboardShortcuts(LabelingState& state, const LabelStudioSessionState& session) {
    if (ImGui::IsAnyItemActive()) {
        return;
    }
    if (!ImGui::GetIO().KeyCtrl) {
        return;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        const bool canSubmit = state.selectedTaskId >= 0 && anyEditorDirty(state) && !state.submitInProgress;
        if (canSubmit) {
            beginSubmitLabelingAnnotation(state, session);
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        if (const auto nextId = nextLabelingTaskId(state, 1)) {
            requestSelectLabelingTask(state, session, *nextId);
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        if (const auto prevId = nextLabelingTaskId(state, -1)) {
            requestSelectLabelingTask(state, session, *prevId);
        }
    }
}

} // namespace

void drawLabelingTabContent(
    bool* mergedWindowOpen, LabelingState& state, const LabelStudioSessionState& session,
    SharedLabelStudioProjectData& sharedData, const std::function<void()>& onOpenLabelStudioWindow) {
    ensureScratchFolder(state);

    if (!state.pendingLocalImagePath.empty() && state.pendingLocalImagePath != state.loadedLocalImagePath) {
        cv::Mat image = cv::imread(state.pendingLocalImagePath);
        if (!image.empty()) {
            if (image.channels() == 1) {
                cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
            }
            state.baseImage = image;
            if (state.imageTexture == 0) {
                glGenTextures(1, &state.imageTexture);
            }
            const cv::Mat display =
                state.maskEditor ? compositeMaskOverlay(state.baseImage, state.maskEditor->regions) : state.baseImage;
            uploadFrameToTexture(state.imageTexture, display, state.imageWidth, state.imageHeight);
        }
        state.loadedLocalImagePath = state.pendingLocalImagePath;
    }

    if (state.focusTaskId >= 0) {
        const int taskId = state.focusTaskId;
        state.focusTaskId = -1;
        requestSelectLabelingTask(state, session, taskId);
    }

    if (!state.submitInProgress && !state.unsavedPromptOpen
        && state.unsavedPromptAction == LabelingUnsavedPromptAction::CloseWindow && state.submitStatus == "Saved.") {
        *mergedWindowOpen = false;
        state.unsavedPromptAction = LabelingUnsavedPromptAction::None;
    }

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        handleLabelingWindowKeyboardShortcuts(state, session);
    }

    drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);
    if (!state.configStatus.empty()) {
        ImGui::TextDisabled("%s", state.configStatus.c_str());
    }
    ImGui::Separator();

    ImGui::BeginChild("LabelingBody", ImVec2(0, 0), false);
    drawTaskListPanel(state, session, sharedData);
    ImGui::SameLine();
    const bool hasEditorPanel = state.boxEditor || state.choiceEditor || state.maskEditor;
    const float editorPanelWidth = 240.0f;
    const float canvasWidth =
        hasEditorPanel ? ImGui::GetContentRegionAvail().x - editorPanelWidth - ImGui::GetStyle().ItemSpacing.x : 0.0f;
    bool maskChanged = false;
    drawImageCanvas(state, canvasWidth, maskChanged);
    if (hasEditorPanel) {
        ImGui::SameLine();
        ImGui::BeginChild("LabelingEditorPanel", ImVec2(editorPanelWidth, 0), true);
        if (state.boxEditor) {
            drawBoxEditorPanel(*state.boxEditor);
        }
        if (state.boxEditor && (state.choiceEditor || state.maskEditor)) {
            ImGui::Separator();
        }
        if (state.choiceEditor) {
            drawChoiceEditorPanel(*state.choiceEditor);
        }
        if (state.choiceEditor && state.maskEditor) {
            ImGui::Separator();
        }
        if (state.maskEditor) {
            drawMaskEditorPanel(*state.maskEditor, state.imageWidth, state.imageHeight, maskChanged);
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    if (maskChanged && !state.baseImage.empty()) {
        const cv::Mat display =
            state.maskEditor ? compositeMaskOverlay(state.baseImage, state.maskEditor->regions) : state.baseImage;
        uploadFrameToTexture(state.imageTexture, display, state.imageWidth, state.imageHeight);
    }

    ImGui::Separator();
    const bool canSubmit = state.selectedTaskId >= 0 && anyEditorDirty(state) && !state.submitInProgress;
    ImGui::BeginDisabled(!canSubmit);
    if (ImGui::Button("Submit")) {
        beginSubmitLabelingAnnotation(state, session);
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
            confirmSaveAndSwitchTask(state, session);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard")) {
            // confirmDiscardAndSwitchTask clears unsavedPromptAction, so capture
            // whether this was a pending window-close before calling it.
            const bool wasClosingWindow = state.unsavedPromptAction == LabelingUnsavedPromptAction::CloseWindow;
            confirmDiscardAndSwitchTask(state, session);
            if (wasClosingWindow) {
                *mergedWindowOpen = false;
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
}

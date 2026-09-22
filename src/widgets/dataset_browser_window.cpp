#include "widgets/dataset_browser_window.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_color.hpp"
#include "manager/rotated_box_geometry.hpp"
#include "widgets/file_browser_utils.hpp"
#include "widgets/label_studio_window.hpp"
#include "widgets/tooltip_helpers.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>

namespace {

constexpr float kThumbnailCellSize = 150.0f;
constexpr float kThumbnailCellSpacing = 8.0f;
constexpr int kGridLookaheadRows = 2;

ImU32 toImU32(const LabelColor& color) {
    return IM_COL32(color.r, color.g, color.b, 255);
}

// Draws `boxes` (in `textureWidth`x`textureHeight` pixel space -- see
// DatasetThumbnailEntry) as an overlay on top of the image last drawn
// via ImGui::Image/ImageButton, scaling to that image's actual on-screen
// rect (ImGui::GetItemRectMin/Max). Mirrors labeling_window.cpp's
// drawBoxOverlay exactly (same scale-then-AddRect/AddPolyline/AddText
// shape, same colorForClassName/toImU32 coloring), so annotation and
// prediction boxes look and behave like every other box this app draws
// -- just not interactively editable here.
void drawBoxesOverlay(const std::vector<DraftDetectionBox>& boxes, int textureWidth, int textureHeight) {
    if (boxes.empty() || textureWidth <= 0 || textureHeight <= 0) {
        return;
    }
    const ImVec2 imageMin = ImGui::GetItemRectMin();
    const ImVec2 imageMax = ImGui::GetItemRectMax();
    const float imageW = imageMax.x - imageMin.x;
    const float imageH = imageMax.y - imageMin.y;
    if (imageW <= 1.0f || imageH <= 1.0f) {
        return;
    }
    const float sx = imageW / static_cast<float>(textureWidth);
    const float sy = imageH / static_cast<float>(textureHeight);

    for (const auto& box : boxes) {
        const ImU32 color = toImU32(colorForClassName(box.className));
        const auto corners = rotatedBoxCorners(box.box, box.rotationDegrees);
        ImVec2 screenCorners[4];
        for (int c = 0; c < 4; ++c) {
            screenCorners[c] = ImVec2(imageMin.x + sx * corners[c].x, imageMin.y + sy * corners[c].y);
        }

        if (box.rotationDegrees == 0.0f) {
            ImGui::GetWindowDrawList()->AddRect(screenCorners[0], screenCorners[2], color, 0.0f, 0, 2.0f);
        } else {
            ImGui::GetWindowDrawList()->AddPolyline(screenCorners, 4, color, ImDrawFlags_Closed, 2.0f);
        }
        if (!box.className.empty()) {
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(screenCorners[0].x, screenCorners[0].y - 14.0f), color, box.className.c_str());
        }
    }
}

void drawFilterControls(DatasetBrowserState& state) {
    bool filterChanged = false;

    static const char* kPresenceLabels[] = {"Any", "Has", "Lacks"};

    int annotationIndex = static_cast<int>(state.filter.annotationFilter);
    if (ImGui::Combo("Annotation", &annotationIndex, kPresenceLabels, IM_ARRAYSIZE(kPresenceLabels))) {
        state.filter.annotationFilter = static_cast<DatasetPresenceFilter>(annotationIndex);
        filterChanged = true;
    }
    int predictionIndex = static_cast<int>(state.filter.predictionFilter);
    if (ImGui::Combo("Prediction", &predictionIndex, kPresenceLabels, IM_ARRAYSIZE(kPresenceLabels))) {
        state.filter.predictionFilter = static_cast<DatasetPresenceFilter>(predictionIndex);
        filterChanged = true;
    }

    if (ImGui::InputText("Class name", &state.filter.classNameFilter)) {
        filterChanged = true;
    }

    static const char* kConfidenceLabels[] = {"None", "< threshold", "> threshold"};
    int confidenceIndex = static_cast<int>(state.filter.confidenceFilterMode);
    if (ImGui::Combo("Confidence filter", &confidenceIndex, kConfidenceLabels, IM_ARRAYSIZE(kConfidenceLabels))) {
        state.filter.confidenceFilterMode = static_cast<DatasetConfidenceFilterMode>(confidenceIndex);
        filterChanged = true;
    }
    ImGui::BeginDisabled(state.filter.confidenceFilterMode == DatasetConfidenceFilterMode::None);
    if (ImGui::SliderFloat("Threshold", &state.filter.confidenceThreshold, 0.0f, 1.0f, "%.2f")) {
        filterChanged = true;
    }
    ImGui::EndDisabled();

    if (filterChanged) {
        reapplyDatasetBrowserFilter(state);
    }

    // Display-only toggles -- don't affect which tasks match, only
    // whether the overlay draws that box list.
    ImGui::Checkbox("Show annotations", &state.showAnnotationBoxes);
    ImGui::SameLine();
    ImGui::Checkbox("Show predictions", &state.showPredictionBoxes);
}

void drawGrid(DatasetBrowserState& state) {
    const float availableWidth = ImGui::GetContentRegionAvail().x;
    const int columns = std::max(1, static_cast<int>(availableWidth / (kThumbnailCellSize + kThumbnailCellSpacing)));
    const int itemCount = static_cast<int>(state.matchingTaskIds.size());
    const int totalRows = columns > 0 ? (itemCount + columns - 1) / columns : 0;

    ImGui::BeginChild("DatasetBrowserGrid", ImVec2(0, 0), true);

    std::vector<int> visibleAndLookaheadIds;

    ImGuiListClipper clipper;
    clipper.Begin(totalRows, kThumbnailCellSize + kThumbnailCellSpacing);
    while (clipper.Step()) {
        const int lookaheadStart = std::max(0, clipper.DisplayStart - kGridLookaheadRows);
        const int lookaheadEnd = std::min(totalRows, clipper.DisplayEnd + kGridLookaheadRows);

        for (int row = lookaheadStart; row < lookaheadEnd; ++row) {
            for (int col = 0; col < columns; ++col) {
                const int index = row * columns + col;
                if (index >= itemCount) {
                    break;
                }
                visibleAndLookaheadIds.push_back(state.matchingTaskIds[index]);
            }
        }

        for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
            for (int col = 0; col < columns; ++col) {
                const int index = row * columns + col;
                if (index >= itemCount) {
                    break;
                }
                const int taskId = state.matchingTaskIds[index];
                if (col > 0) {
                    ImGui::SameLine();
                }

                ImGui::PushID(taskId);
                const auto it = state.thumbnailCache.entries.find(taskId);
                if (it == state.thumbnailCache.entries.end() || it->second.status == DatasetThumbnailStatus::Loading) {
                    ImGui::Button("...", ImVec2(kThumbnailCellSize, kThumbnailCellSize));
                } else if (it->second.status == DatasetThumbnailStatus::Failed) {
                    ImGui::Button("Failed", ImVec2(kThumbnailCellSize, kThumbnailCellSize));
                } else {
                    const ImVec2 fitted = fitImageToRegion(
                        it->second.textureWidth, it->second.textureHeight, kThumbnailCellSize, kThumbnailCellSize);
                    if (ImGui::ImageButton("thumb", (void*)(intptr_t)it->second.texture, fitted)) {
                        state.selectedTaskId = taskId;
                    }
                    if (state.showAnnotationBoxes) {
                        drawBoxesOverlay(it->second.annotationBoxes, it->second.textureWidth, it->second.textureHeight);
                    }
                    if (state.showPredictionBoxes) {
                        drawBoxesOverlay(it->second.predictionBoxes, it->second.textureWidth, it->second.textureHeight);
                    }
                    drawHoverEnlargedImage(it->second.texture, it->second.textureWidth, it->second.textureHeight, fitted);
                }
                ImGui::PopID();
            }
        }
    }
    clipper.End();

    for (const int taskId : visibleAndLookaheadIds) {
        state.thumbnailCache.touch(taskId);
        if (state.thumbnailCache.entries.find(taskId) == state.thumbnailCache.entries.end()) {
            const auto summaryIt = std::find_if(
                state.summaries.begin(), state.summaries.end(),
                [taskId](const DatasetTaskSummary& s) { return s.taskId == taskId; });
            if (summaryIt != state.summaries.end()) {
                const auto boxes = boxesToDrawForTask(state.rawTasksJson, state.rectangleLabelsFromName, taskId);
                state.thumbnailWorker.requestThumbnail(
                    DatasetThumbnailRequest{taskId, summaryIt->imagePath, boxes.annotationBoxes, boxes.predictionBoxes});
            }
        }
    }
    state.thumbnailWorker.setStillWanted(visibleAndLookaheadIds);
    state.thumbnailCache.evictIfNeeded(visibleAndLookaheadIds);

    ImGui::EndChild();
}

void drawSelectedTaskDetail(DatasetBrowserState& state) {
    ImGui::BeginChild("DatasetBrowserDetail", ImVec2(320.0f, 0), true);
    if (!state.selectedTaskId) {
        ImGui::TextDisabled("Click a thumbnail to preview it here.");
        ImGui::EndChild();
        return;
    }

    const int selectedId = *state.selectedTaskId;
    const auto summaryIt = std::find_if(
        state.summaries.begin(), state.summaries.end(),
        [selectedId](const DatasetTaskSummary& s) { return s.taskId == selectedId; });
    if (summaryIt == state.summaries.end()) {
        ImGui::TextDisabled("Task no longer in the current list.");
        ImGui::EndChild();
        return;
    }

    ImGui::Text("Task #%d", summaryIt->taskId);
    ImGui::Text("Annotation: %s", summaryIt->hasAnnotation ? "yes" : "no");
    ImGui::Text("Prediction: %s", summaryIt->hasPrediction ? "yes" : "no");
    if (summaryIt->minConfidence && summaryIt->maxConfidence) {
        ImGui::Text("Confidence: %.2f - %.2f", *summaryIt->minConfidence, *summaryIt->maxConfidence);
    }
    if (!summaryIt->classNames.empty()) {
        std::string classNamesText;
        for (const auto& className : summaryIt->classNames) {
            if (!classNamesText.empty()) {
                classNamesText += ", ";
            }
            classNamesText += className;
        }
        ImGui::TextWrapped("Classes: %s", classNamesText.c_str());
    }

    const auto textureIt = state.thumbnailCache.entries.find(selectedId);
    if (textureIt != state.thumbnailCache.entries.end() && textureIt->second.status == DatasetThumbnailStatus::Loaded) {
        const ImVec2 fitted =
            fitImageToRegion(textureIt->second.textureWidth, textureIt->second.textureHeight, 280.0f, 280.0f);
        ImGui::Image((void*)(intptr_t)textureIt->second.texture, fitted);
        if (state.showAnnotationBoxes) {
            drawBoxesOverlay(textureIt->second.annotationBoxes, textureIt->second.textureWidth, textureIt->second.textureHeight);
        }
        if (state.showPredictionBoxes) {
            drawBoxesOverlay(textureIt->second.predictionBoxes, textureIt->second.textureWidth, textureIt->second.textureHeight);
        }
    } else {
        ImGui::TextDisabled("Thumbnail not loaded.");
    }

    ImGui::EndChild();
}

void drawExportFolderPickerPopup(DatasetBrowserState& state) {
    if (state.exportFolderPickerOpen) {
        ImGui::OpenPopup("Pick Dataset Export Folder");
        state.exportFolderPickerOpen = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Pick Dataset Export Folder", nullptr)) {
        std::string selected;
        if (drawDirectoryBrowser(
                state.exportFolderPickerExplorerDir, &selected, "DatasetExportFolderPickerDirs",
                state.exportFolderPickerFilter)) {
            state.exportDestinationFolder = selected;
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void drawExportSection(DatasetBrowserState& state, const LabelStudioSessionState& session) {
    ImGui::Separator();
    ImGui::TextWrapped(
        "Export folder: %s", state.exportDestinationFolder.empty() ? "(none)" : state.exportDestinationFolder.c_str());
    if (ImGui::Button("Browse Folder...")) {
        state.exportFolderPickerExplorerDir = state.exportDestinationFolder;
        state.exportFolderPickerOpen = true;
    }

    if (state.exportState == DatasetExportState::Running) {
        ImGui::Text("%d / %d", state.lastExportProgress.completed, state.lastExportProgress.total);
        const float fraction = state.lastExportProgress.total > 0
            ? static_cast<float>(state.lastExportProgress.completed) / static_cast<float>(state.lastExportProgress.total)
            : 0.0f;
        ImGui::ProgressBar(fraction);
        if (ImGui::Button("Cancel")) {
            state.exportWorker.requestCancel();
        }
        return;
    }

    if (state.exportState == DatasetExportState::Cancelled) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Export cancelled.");
    }

    const bool canExport = !state.matchingTaskIds.empty() && !state.exportDestinationFolder.empty();
    ImGui::BeginDisabled(!canExport);
    if (ImGui::Button("Export")) {
        startDatasetBrowserExport(state, session);
    }
    ImGui::EndDisabled();
    if (!state.exportStatus.empty()) {
        ImGui::TextDisabled("%s", state.exportStatus.c_str());
    }
}

} // namespace

void drawDatasetBrowserTabContent(
    DatasetBrowserState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);

    const bool canBrowse = session.status == LabelStudioSessionStatus::Connected && session.activeProjectId > 0;
    ImGui::BeginDisabled(!canBrowse);
    if (ImGui::Button("Refresh")) {
        refreshDatasetBrowserTaskList(state, session);
    }
    ImGui::EndDisabled();

    if (!state.taskListError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.taskListError.c_str());
    }

    if (!state.taskListLoaded) {
        ImGui::TextDisabled("Click Refresh to load this project's tasks.");
        return;
    }

    drawFilterControls(state);
    ImGui::Text("%zu of %zu tasks match", state.matchingTaskIds.size(), state.summaries.size());

    drawExportSection(state, session);
    drawExportFolderPickerPopup(state);

    ImGui::Separator();
    ImGui::BeginChild("DatasetBrowserBody", ImVec2(0, 0), false);
    ImGui::BeginChild("DatasetBrowserGridPane", ImVec2(-330.0f, 0), false);
    drawGrid(state);
    ImGui::EndChild();
    ImGui::SameLine();
    drawSelectedTaskDetail(state);
    ImGui::EndChild();
}

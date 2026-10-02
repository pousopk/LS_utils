#include "widgets/label_assistant_window.hpp"

#include "ui_common/image_fit.hpp"
#include "manager/label_studio_client.hpp"
#include "ui_common/file_browser_utils.hpp"
#include "widgets/filter_widgets.hpp"
#include "widgets/label_studio_window.hpp"
#include "widgets/model_slot_config_widget.hpp"
#include "ui_common/tooltip_helpers.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <filesystem>
#include <vector>

namespace {

void drawTaskModeToggle(LabelAssistantState& state) {
    auto switchTo = [&state](ComparisonTaskMode mode) {
        if (state.taskMode == mode) {
            return;
        }
        state.taskMode = mode;
        state.runState = LabelAssistantRunState::NotStarted;
        state.result = LabelAssistantResult{};
        state.selectedImageFilename.reset();
    };

    if (ImGui::RadioButton("Detection", state.taskMode == ComparisonTaskMode::Detection)) {
        switchTo(ComparisonTaskMode::Detection);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Classification", state.taskMode == ComparisonTaskMode::Classification)) {
        switchTo(ComparisonTaskMode::Classification);
    }
}

void drawSourceModeToggle(LabelAssistantState& state) {
    auto switchTo = [&state](LabelAssistantSourceMode mode) {
        if (state.sourceMode == mode) {
            return;
        }
        state.sourceMode = mode;
        state.runState = LabelAssistantRunState::NotStarted;
        state.result = LabelAssistantResult{};
        state.selectedImageFilename.reset();
        state.imageFolderPath.clear();
    };

    ImGui::TextUnformatted("Source");
    if (ImGui::RadioButton("Local Folder", state.sourceMode == LabelAssistantSourceMode::LocalFolder)) {
        switchTo(LabelAssistantSourceMode::LocalFolder);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(
            "Label Studio Project", state.sourceMode == LabelAssistantSourceMode::LabelStudioProject)) {
        switchTo(LabelAssistantSourceMode::LabelStudioProject);
    }
}

void drawModelConfig(LabelAssistantState& state) {
    float* confThreshold =
        (state.taskMode == ComparisonTaskMode::Detection) ? &state.modelConfig.confThreshold : nullptr;
    float* nmsThreshold =
        (state.taskMode == ComparisonTaskMode::Detection) ? &state.modelConfig.nmsThreshold : nullptr;

    const bool loadClicked = drawModelSlotConfigFields(
        state.modelConfig.onnxPath, state.modelConfig.classNamesPath, state.modelConfig.inputWidth,
        state.modelConfig.inputHeight, confThreshold, nmsThreshold, &state.modelConfig.isObbDetectionModel,
        state.modelConfig.autoDetectStatus, state.modelConfig.loadError,
        [&state]() {
            state.filePickerTarget = LabelAssistantFilePickerTarget::OnnxModel;
            state.filePickerOpen = true;
        },
        [&state]() {
            state.filePickerTarget = LabelAssistantFilePickerTarget::ClassNamesFile;
            state.filePickerOpen = true;
        });

    if (!state.modelConfig.engineStatus.empty()) {
        ImGui::TextDisabled("%s", state.modelConfig.engineStatus.c_str());
    }

    if (loadClicked) {
        loadLabelAssistantModel(state);
    }
}

void drawFolderPicker(LabelAssistantState& state) {
    ImGui::TextWrapped(
        "Image Folder: %s", state.imageFolderPath.empty() ? "(none)" : state.imageFolderPath.c_str());
    if (ImGui::Button("Browse Folder...")) {
        state.folderPickerExplorerDir = state.imageFolderPath;
        state.folderPickerOpen = true;
    }
}

void drawRunBar(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData) {
    ImGui::Separator();
    if (state.runState == LabelAssistantRunState::Running) {
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

    if (state.runState == LabelAssistantRunState::Cancelled) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Run cancelled.");
    }

    const bool modelLoaded = state.taskMode == ComparisonTaskMode::Detection
        ? state.modelConfig.detectionModel != nullptr
        : state.modelConfig.classificationModel != nullptr;
    const bool sourceReady = state.sourceMode == LabelAssistantSourceMode::LocalFolder
        ? !state.imageFolderPath.empty()
        : !session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty() && sharedData.loaded;
    const bool canRun = modelLoaded && sourceReady;
    if (state.sourceMode == LabelAssistantSourceMode::LabelStudioProject) {
        drawSharedTaskRangeNote(sharedData);
    }
    ImGui::BeginDisabled(!canRun);
    if (ImGui::Button("Run")) {
        startLabelAssistantRun(state, session, sharedData);
    }
    ImGui::EndDisabled();
    if (!canRun) {
        ImGui::TextDisabled(
            state.sourceMode == LabelAssistantSourceMode::LocalFolder
                ? "Load a model matching the selected mode and pick an image folder to run."
                : "Load a model matching the selected mode and wait for the task list to load to run.");
    }
}

void drawResultsSummary(const LabelAssistantState& state) {
    const int drafted = state.taskMode == ComparisonTaskMode::Classification
        ? static_cast<int>(state.result.classificationDrafts.size())
        : static_cast<int>(state.result.detectionDrafts.size());
    ImGui::Text("Images processed: %d   Drafted: %d", state.result.imagesProcessed, drafted);
}

void drawImageList(
    LabelAssistantState& state, const LabelStudioSessionState& session, const LabelTaskCallback& onLabelTask) {
    ImGui::BeginChild("LabelAssistantImageList", ImVec2(320.0f, 380.0f), true);
    drawTextSearch("##LabelAssistantImageFilter", "Search filename...", state.imageSearch);
    drawConfidenceSort("Sort", state.sortMode);
    drawConfidenceFilter("Confidence filter", state.confidenceFilter);

    const std::vector<LabelAssistantImageEntry> entries = buildLabelAssistantImageEntries(state);
    const std::vector<const LabelAssistantImageEntry*> filtered =
        filterLabelAssistantEntries(entries, state.imageSearch, state.confidenceFilter, state.sortMode);

    ImGui::Separator();
    ImGui::BeginChild("LabelAssistantImageListScroll", ImVec2(0, 0), false);
    for (const auto* entry : filtered) {
        const bool selected = state.selectedImageFilename && *state.selectedImageFilename == entry->filename;
        const std::string label = entry->filename + "  [" + entry->labelSummary + "]";
        if (ImGui::Selectable(label.c_str(), selected)) {
            state.selectedImageFilename = entry->filename;
        }
        const auto taskId = parseTaskIdFromFilename(entry->filename);
        if (taskId.has_value()) {
            ImGui::SameLine();
            if (ImGui::SmallButton(("Label##" + entry->filename).c_str())) {
                onLabelTask(*taskId);
            }
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();
}

void drawSelectedImageDetail(LabelAssistantState& state) {
    ImGui::SameLine();
    ImGui::BeginChild("LabelAssistantImageDetail", ImVec2(0, 380.0f), true);

    if (!state.selectedImageFilename) {
        ImGui::TextDisabled("Select an image to view details.");
        ImGui::EndChild();
        return;
    }

    if (state.previewTexture != 0 && state.previewTextureWidth > 0 && state.previewTextureHeight > 0) {
        const float maxWidth = ImGui::GetContentRegionAvail().x;
        const ImVec2 size = fitImageToRegion(state.previewTextureWidth, state.previewTextureHeight, maxWidth, 300.0f);
        ImGui::Image((void*)(intptr_t)state.previewTexture, size);
        drawHoverEnlargedImage(state.previewTexture, state.previewTextureWidth, state.previewTextureHeight, size);
    } else {
        ImGui::TextDisabled("No preview.");
    }

    if (state.taskMode == ComparisonTaskMode::Classification) {
        for (const auto& draft : state.result.classificationDrafts) {
            if (draft.imageFilename == *state.selectedImageFilename) {
                ImGui::Text("Predicted: %s (%.1f%%)", draft.predictedLabel.c_str(), draft.confidence * 100.0f);
                break;
            }
        }
    } else {
        for (const auto& draft : state.result.detectionDrafts) {
            if (draft.imageFilename == *state.selectedImageFilename) {
                if (draft.boxes.empty()) {
                    ImGui::TextDisabled("No detections.");
                }
                for (const auto& box : draft.boxes) {
                    ImGui::Text("%s (%.1f%%)", box.className.c_str(), box.confidence * 100.0f);
                }
                break;
            }
        }
    }

    ImGui::EndChild();
}

void drawExportSection(
    LabelAssistantState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    ImGui::Separator();
    const bool isLocalFolder = state.sourceMode == LabelAssistantSourceMode::LocalFolder;
    ImGui::TextUnformatted(isLocalFolder ? "Push to Label Studio" : "Attach Predictions to Label Studio");

    // In LabelStudioProject mode the shared session summary is already
    // shown above the Run bar -- no need to repeat it here.
    if (isLocalFolder) {
        drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);
    }

    ImGui::InputText(
        state.taskMode == ComparisonTaskMode::Classification ? "from_name (choices)"
                                                               : "from_name (rectanglelabels)",
        &state.labelFromName);
    ImGui::InputText("to_name", &state.imageToName);

    if (state.taskMode == ComparisonTaskMode::Detection) {
        ImGui::Checkbox("Include images with no detections", &state.includeZeroDetectionImages);
    }

    if (state.pushState == LabelAssistantPushState::Running) {
        ImGui::Text("%d / %d", state.lastPushProgress.completed, state.lastPushProgress.total);
        const float fraction = state.lastPushProgress.total > 0
            ? static_cast<float>(state.lastPushProgress.completed) / static_cast<float>(state.lastPushProgress.total)
            : 0.0f;
        ImGui::ProgressBar(fraction);
        if (ImGui::Button("Cancel")) {
            state.pushWorker.requestCancel();
        }
        return;
    }

    if (state.pushState == LabelAssistantPushState::Cancelled) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Push cancelled.");
    }

    const bool hasDrafts = state.taskMode == ComparisonTaskMode::Classification
        ? !state.result.classificationDrafts.empty()
        : !state.result.detectionDrafts.empty();
    const bool canPush = hasDrafts && !session.baseUrl.empty() && session.activeProjectId > 0
        && !session.apiToken.empty();
    ImGui::BeginDisabled(!canPush);
    if (ImGui::Button(isLocalFolder ? "Push to Label Studio" : "Attach Predictions to Label Studio")) {
        pushLabelAssistantDraftsToLabelStudio(state, session);
    }
    ImGui::EndDisabled();
    if (!state.exportStatus.empty()) {
        ImGui::TextDisabled("%s", state.exportStatus.c_str());
    }
}

void drawFolderPickerPopup(LabelAssistantState& state) {
    if (state.folderPickerOpen) {
        ImGui::OpenPopup("Pick Label Assistant Folder");
        state.folderPickerOpen = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Pick Label Assistant Folder", nullptr)) {
        std::string selected;
        if (drawDirectoryBrowser(
                state.folderPickerExplorerDir, &selected, "LabelAssistantFolderPickerDirs",
                state.folderPickerFilter)) {
            state.imageFolderPath = selected;
            state.runState = LabelAssistantRunState::NotStarted;
            state.result = LabelAssistantResult{};
            state.selectedImageFilename.reset();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void drawFilePickerPopup(LabelAssistantState& state) {
    if (state.filePickerOpen) {
        ImGui::OpenPopup("Pick Label Assistant File");
        state.filePickerOpen = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Pick Label Assistant File", nullptr)) {
        namespace fs = std::filesystem;

        const std::string dirBeforeBrowse = state.filePickerDir;
        drawDirectoryBrowser(state.filePickerDir, nullptr, "LabelAssistantFilePickerDirs", state.filePickerFilter);
        if (state.filePickerDir != dirBeforeBrowse) {
            state.filePickerSelectedFile.clear();
        }

        ImGui::BeginChild("LabelAssistantFilePickerFiles", ImVec2(0, 260.0f), true);
        for (const auto& file : listFiles(fs::path(state.filePickerDir))) {
            if (!fileNameMatchesFilter(file, state.filePickerFilter)) {
                continue;
            }
            const std::string filePath = file.string();
            const bool selected = (filePath == state.filePickerSelectedFile);
            if (ImGui::Selectable(file.filename().string().c_str(), selected)) {
                state.filePickerSelectedFile = filePath;
            }
        }
        ImGui::EndChild();

        if (!state.filePickerSelectedFile.empty()) {
            ImGui::TextWrapped("Selected: %s", state.filePickerSelectedFile.c_str());
        }

        if (ImGui::Button("Use Selected File") && !state.filePickerSelectedFile.empty()) {
            switch (state.filePickerTarget) {
                case LabelAssistantFilePickerTarget::OnnxModel:
                    state.modelConfig.onnxPath = state.filePickerSelectedFile;
                    applyLabelAssistantAutoDetect(state);
                    break;
                case LabelAssistantFilePickerTarget::ClassNamesFile:
                    state.modelConfig.classNamesPath = state.filePickerSelectedFile;
                    break;
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

} // namespace

void drawLabelAssistantTabContent(
    LabelAssistantState& state, const LabelStudioSessionState& session, const SharedLabelStudioProjectData& sharedData,
    const LabelTaskCallback& onLabelTask, const std::function<void()>& onOpenLabelStudioWindow) {
    drawTaskModeToggle(state);
    ImGui::Separator();

    drawModelConfig(state);
    ImGui::Separator();

    drawSourceModeToggle(state);
    if (state.sourceMode == LabelAssistantSourceMode::LocalFolder) {
        drawFolderPicker(state);
    } else {
        drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);
        if (!state.labelStudioAutoFetchStatus.empty()) {
            ImGui::TextDisabled("%s", state.labelStudioAutoFetchStatus.c_str());
        }
    }
    ImGui::Separator();

    drawRunBar(state, session, sharedData);

    if (state.runState == LabelAssistantRunState::Complete) {
        ImGui::Separator();
        if (!state.result.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.result.error.c_str());
        } else {
            drawResultsSummary(state);
            ImGui::Separator();
            drawImageList(state, session, onLabelTask);
            drawSelectedImageDetail(state);
            drawExportSection(state, session, onOpenLabelStudioWindow);
        }
    }

    drawFolderPickerPopup(state);
    drawFilePickerPopup(state);
}

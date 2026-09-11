#include "widgets/label_assistant_window.hpp"

#include "manager/app_runtime.hpp"
#include "manager/label_studio_client.hpp"
#include "widgets/file_browser_utils.hpp"
#include "widgets/model_slot_config_widget.hpp"

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

// Shared by the "Label Studio Project" source setup (needed before Run)
// and the export section (needed before Push, LocalFolder mode only).
void drawLabelStudioConnectionFields(LabelAssistantState& state) {
    ImGui::InputText("Label Studio URL", &state.labelStudioBaseUrl);
    ImGui::InputInt("Project ID", &state.labelStudioProjectId);
    ImGui::InputText("API Token", &state.labelStudioApiToken, ImGuiInputTextFlags_Password);
    if (!state.labelStudioAutoFetchStatus.empty()) {
        ImGui::TextDisabled("%s", state.labelStudioAutoFetchStatus.c_str());
    }
}

void drawModelConfig(LabelAssistantState& state) {
    float* confThreshold =
        (state.taskMode == ComparisonTaskMode::Detection) ? &state.modelConfig.confThreshold : nullptr;
    float* nmsThreshold =
        (state.taskMode == ComparisonTaskMode::Detection) ? &state.modelConfig.nmsThreshold : nullptr;

    const bool loadClicked = drawModelSlotConfigFields(
        state.modelConfig.onnxPath, state.modelConfig.classNamesPath, state.modelConfig.inputWidth,
        state.modelConfig.inputHeight, confThreshold, nmsThreshold, state.modelConfig.autoDetectStatus,
        state.modelConfig.loadError,
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

void drawRunBar(LabelAssistantState& state) {
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
        : !state.labelStudioBaseUrl.empty() && state.labelStudioProjectId > 0 && !state.labelStudioApiToken.empty();
    const bool canRun = modelLoaded && sourceReady;
    ImGui::BeginDisabled(!canRun);
    if (ImGui::Button("Run")) {
        startLabelAssistantRun(state);
    }
    ImGui::EndDisabled();
    if (!canRun) {
        ImGui::TextDisabled(
            state.sourceMode == LabelAssistantSourceMode::LocalFolder
                ? "Load a model matching the selected mode and pick an image folder to run."
                : "Load a model matching the selected mode and fill in the Label Studio connection to run.");
    }
}

void drawResultsSummary(const LabelAssistantState& state) {
    const int drafted = state.taskMode == ComparisonTaskMode::Classification
        ? static_cast<int>(state.result.classificationDrafts.size())
        : static_cast<int>(state.result.detectionDrafts.size());
    ImGui::Text("Images processed: %d   Drafted: %d", state.result.imagesProcessed, drafted);
}

// One row per drafted image, mode-agnostic so the list/sort/filter code
// below doesn't need to branch on taskMode. Detection's confidence is the
// mean across that image's boxes, matching batchEvalImageConfidence's
// existing convention (model_evaluation_state.hpp) for the same class of
// "one representative confidence per image" need.
struct LabelAssistantImageEntry {
    std::string filename;
    float confidence = 0.0f;
    std::string labelSummary;
};

std::vector<LabelAssistantImageEntry> buildImageEntries(const LabelAssistantState& state) {
    std::vector<LabelAssistantImageEntry> entries;
    if (state.taskMode == ComparisonTaskMode::Classification) {
        entries.reserve(state.result.classificationDrafts.size());
        for (const auto& draft : state.result.classificationDrafts) {
            entries.push_back(LabelAssistantImageEntry{draft.imageFilename, draft.confidence, draft.predictedLabel});
        }
    } else {
        entries.reserve(state.result.detectionDrafts.size());
        for (const auto& draft : state.result.detectionDrafts) {
            float sum = 0.0f;
            for (const auto& box : draft.boxes) {
                sum += box.confidence;
            }
            const float meanConfidence = draft.boxes.empty() ? 0.0f : sum / static_cast<float>(draft.boxes.size());
            entries.push_back(LabelAssistantImageEntry{
                draft.imageFilename, meanConfidence, std::to_string(draft.boxes.size()) + " box(es)"});
        }
    }
    return entries;
}

void drawImageList(LabelAssistantState& state, const LabelTaskCallback& onLabelTask) {
    ImGui::BeginChild("LabelAssistantImageList", ImVec2(320.0f, 380.0f), true);
    ImGui::InputTextWithHint("##LabelAssistantImageFilter", "Search filename...", &state.imageListFilter);

    static const char* kSortLabels[] = {"Filename", "Confidence (low first)", "Confidence (high first)"};
    int sortIndex = static_cast<int>(state.sortMode);
    if (ImGui::Combo("Sort", &sortIndex, kSortLabels, IM_ARRAYSIZE(kSortLabels))) {
        state.sortMode = static_cast<LabelAssistantSortMode>(sortIndex);
    }

    static const char* kConfidenceFilterLabels[] = {"None", "< threshold", "> threshold"};
    int confidenceFilterIndex = static_cast<int>(state.confidenceFilterMode);
    if (ImGui::Combo(
            "Confidence filter", &confidenceFilterIndex, kConfidenceFilterLabels,
            IM_ARRAYSIZE(kConfidenceFilterLabels))) {
        state.confidenceFilterMode = static_cast<LabelAssistantConfidenceFilterMode>(confidenceFilterIndex);
    }
    ImGui::BeginDisabled(state.confidenceFilterMode == LabelAssistantConfidenceFilterMode::None);
    ImGui::SliderFloat("Threshold", &state.confidenceFilterThreshold, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();

    const std::vector<LabelAssistantImageEntry> entries = buildImageEntries(state);
    std::vector<const LabelAssistantImageEntry*> filtered;
    for (const auto& entry : entries) {
        if (!fileNameMatchesFilter(std::filesystem::path(entry.filename), state.imageListFilter)) {
            continue;
        }
        if (state.confidenceFilterMode != LabelAssistantConfidenceFilterMode::None) {
            const bool passes = state.confidenceFilterMode == LabelAssistantConfidenceFilterMode::LessThan
                ? entry.confidence < state.confidenceFilterThreshold
                : entry.confidence > state.confidenceFilterThreshold;
            if (!passes) {
                continue;
            }
        }
        filtered.push_back(&entry);
    }

    if (state.sortMode == LabelAssistantSortMode::ConfidenceAscending) {
        std::sort(
            filtered.begin(), filtered.end(),
            [](const LabelAssistantImageEntry* a, const LabelAssistantImageEntry* b) {
                return a->confidence < b->confidence;
            });
    } else if (state.sortMode == LabelAssistantSortMode::ConfidenceDescending) {
        std::sort(
            filtered.begin(), filtered.end(),
            [](const LabelAssistantImageEntry* a, const LabelAssistantImageEntry* b) {
                return a->confidence > b->confidence;
            });
    }

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
                onLabelTask(state.labelStudioBaseUrl, state.labelStudioProjectId, state.labelStudioApiToken, *taskId);
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
        const ImVec2 size = fitImageToRegion(state.previewTextureWidth, state.previewTextureHeight, 500.0f, 300.0f);
        ImGui::Image((void*)(intptr_t)state.previewTexture, size);
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

void drawExportSection(LabelAssistantState& state) {
    ImGui::Separator();
    const bool isLocalFolder = state.sourceMode == LabelAssistantSourceMode::LocalFolder;
    ImGui::TextUnformatted(isLocalFolder ? "Push to Label Studio" : "Attach Predictions to Label Studio");

    // In LabelStudioProject mode the connection fields are already shown
    // (and required) above the Run bar -- no need to repeat them here.
    if (isLocalFolder) {
        drawLabelStudioConnectionFields(state);
    }

    ImGui::InputText(
        state.taskMode == ComparisonTaskMode::Classification ? "from_name (choices)"
                                                               : "from_name (rectanglelabels)",
        &state.labelFromName);
    ImGui::InputText("to_name", &state.imageToName);

    if (state.taskMode == ComparisonTaskMode::Detection) {
        ImGui::Checkbox("Include images with no detections", &state.includeZeroDetectionImages);
    }

    const bool hasDrafts = state.taskMode == ComparisonTaskMode::Classification
        ? !state.result.classificationDrafts.empty()
        : !state.result.detectionDrafts.empty();
    const bool canPush = hasDrafts && !state.labelStudioBaseUrl.empty() && state.labelStudioProjectId > 0
        && !state.labelStudioApiToken.empty();
    ImGui::BeginDisabled(!canPush);
    if (ImGui::Button(isLocalFolder ? "Push to Label Studio" : "Attach Predictions to Label Studio")) {
        pushLabelAssistantDraftsToLabelStudio(state);
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

void drawLabelAssistantWindow(bool* show, LabelAssistantState& state, const LabelTaskCallback& onLabelTask) {
    if (!*show) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(1100.0f, 900.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Label Assistant", show)) {
        ImGui::End();
        return;
    }

    drawTaskModeToggle(state);
    ImGui::Separator();

    drawModelConfig(state);
    ImGui::Separator();

    drawSourceModeToggle(state);
    if (state.sourceMode == LabelAssistantSourceMode::LocalFolder) {
        drawFolderPicker(state);
    } else {
        drawLabelStudioConnectionFields(state);
    }
    ImGui::Separator();

    drawRunBar(state);

    if (state.runState == LabelAssistantRunState::Complete) {
        ImGui::Separator();
        if (!state.result.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.result.error.c_str());
        } else {
            drawResultsSummary(state);
            ImGui::Separator();
            drawImageList(state, onLabelTask);
            drawSelectedImageDetail(state);
            drawExportSection(state);
        }
    }

    ImGui::End();

    drawFolderPickerPopup(state);
    drawFilePickerPopup(state);
}

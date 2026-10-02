#include "widgets/benchmark_tab.hpp"

#include "ui_common/path_picker.hpp"
#include "ui_common/image_fit.hpp"
#include "widgets/filter_widgets.hpp"
#include "widgets/label_studio_window.hpp"
#include "widgets/model_slot_config_widget.hpp"
#include "ui_common/tooltip_helpers.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

// Dark-ish red/green -- readable on this app's light theme (a bright/pastel
// red or green washes out against a white background).
const ImVec4 kGoodColor(0.05f, 0.45f, 0.05f, 1.0f);
const ImVec4 kBadColor(0.65f, 0.1f, 0.1f, 1.0f);

const char* slotLabel(const BenchmarkState& state, int slotIndex) {
    return slotIndex == 0 ? (state.compareTwoModels ? "Model A" : "Model") : "Model B";
}

void drawModelCountToggle(BenchmarkState& state) {
    if (ImGui::RadioButton("Single Model", !state.compareTwoModels)) {
        if (state.compareTwoModels) {
            state.compareTwoModels = false;
            resetBenchmarkResults(state);
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Compare Two Models", state.compareTwoModels)) {
        if (!state.compareTwoModels) {
            state.compareTwoModels = true;
            resetBenchmarkResults(state);
        }
    }
}

void drawSlotConfig(BenchmarkState& state, int slotIndex) {
    ModelSlotConfig& slot = state.slots[static_cast<size_t>(slotIndex)];
    ImGui::PushID(slotIndex);
    ImGui::TextUnformatted(slotLabel(state, slotIndex));

    const BenchmarkPickerTarget target = slotIndex == 0 ? BenchmarkPickerTarget::SlotAModel : BenchmarkPickerTarget::SlotBModel;
    drawModelSlotConfigFields(slot, [&state, &slot, target]() {
        state.filePickerTarget = target;
        openPathPicker(state.picker, PathPickerMode::File, "Pick ONNX Model", slot.onnxPath);
    });

    if (slotIndex == 1 && modelBTaskMismatch(state)) {
        ImGui::TextColored(
            kBadColor, "Model B is %s but Model A is %s -- both must match.", modelTaskName(state.slots[1].task),
            modelTaskName(state.slots[0].task));
    }

    ImGui::PopID();
}

// ---- Run body ----

void drawSourceModeToggle(BenchmarkState& state) {
    auto switchTo = [&state](BenchmarkSourceMode mode) {
        if (state.sourceMode == mode) {
            return;
        }
        state.sourceMode = mode;
        state.imageFolderPath.clear();
        resetBenchmarkResults(state);
    };

    ImGui::TextUnformatted("Source");
    if (ImGui::RadioButton("Local Folder##BatchSource", state.sourceMode == BenchmarkSourceMode::LocalFolder)) {
        switchTo(BenchmarkSourceMode::LocalFolder);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(
            "Label Studio Project##BatchSource", state.sourceMode == BenchmarkSourceMode::LabelStudioProject)) {
        switchTo(BenchmarkSourceMode::LabelStudioProject);
    }
}

void drawLocalFolderAndGroundTruthPickers(BenchmarkState& state) {
    ImGui::TextWrapped(
        "Image Folder: %s", state.imageFolderPath.empty() ? "(none)" : state.imageFolderPath.c_str());
    if (ImGui::Button("Browse Folder...")) {
        state.filePickerTarget = BenchmarkPickerTarget::ImageFolder;
        openPathPicker(state.picker, PathPickerMode::Folder, "Pick Image Folder", state.imageFolderPath);
    }

    ImGui::TextWrapped(
        "Ground Truth (optional): %s",
        state.groundTruthJsonPath.empty() ? "(none)" : state.groundTruthJsonPath.c_str());
    if (ImGui::Button("Browse Ground Truth...")) {
        state.filePickerTarget = BenchmarkPickerTarget::GroundTruthJson;
        openPathPicker(
            state.picker, PathPickerMode::File, "Pick Ground Truth JSON", state.groundTruthJsonPath);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Ground Truth")) {
        state.groundTruthJsonPath.clear();
        loadBenchmarkGroundTruth(state);
    }
    if (!state.groundTruthStatus.empty()) {
        ImGui::TextDisabled("%s", state.groundTruthStatus.c_str());
    }
}

void drawSampleCheckbox(BenchmarkState& state) {
    ImGui::Checkbox("Randomly sample", &state.sampleEnabled);
    ImGui::SameLine();
    ImGui::BeginDisabled(!state.sampleEnabled);
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("images##SampleSize", &state.sampleSize);
    ImGui::EndDisabled();
    if (state.sampleSize < 1) {
        state.sampleSize = 1;
    }
    if (state.sampleEnabled) {
        ImGui::TextDisabled(
            "Evaluates a random subset instead of the whole folder -- useful for a quick check on a huge "
            "dataset.");
    }
}

void drawRunBar(BenchmarkState& state, const LabelStudioSessionState& session) {
    ImGui::Separator();
    if (state.runState == BenchmarkRunState::Running) {
        const std::string label = !state.lastProgress.phaseLabel.empty()
            ? state.lastProgress.phaseLabel
            : slotLabel(state, state.lastProgress.currentSlot == 2 ? 1 : 0);
        ImGui::Text(
            "%s: %d / %d", label.c_str(), state.lastProgress.completed, state.lastProgress.total);
        const float fraction = state.lastProgress.total > 0
            ? static_cast<float>(state.lastProgress.completed) /
                static_cast<float>(state.lastProgress.total)
            : 0.0f;
        ImGui::ProgressBar(fraction);
        if (ImGui::Button("Cancel")) {
            state.worker.requestCancel();
        }
        return;
    }

    if (state.runState == BenchmarkRunState::Cancelled) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Run cancelled.");
    }

    const bool modelsLoaded = benchmarkSlotsReady(state);
    const bool sourceReady = state.sourceMode == BenchmarkSourceMode::LocalFolder
        ? !state.imageFolderPath.empty()
        : !session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty();
    const bool canRun = modelsLoaded && sourceReady;

    ImGui::BeginDisabled(!canRun);
    if (ImGui::Button("Run")) {
        startBenchmarkRun(state, session);
    }
    ImGui::EndDisabled();
    if (!canRun) {
        const char* modelsHint =
            state.compareTwoModels ? "Load two matching models and " : "Load the model and ";
        const char* sourceHint = state.sourceMode == BenchmarkSourceMode::LocalFolder
            ? "pick an image folder to run."
            : "fill in the Label Studio connection to run.";
        ImGui::TextDisabled("%s%s", modelsHint, sourceHint);
    }
}

void drawPerClassApTable(const BenchmarkState& state) {
    if (!ImGui::TreeNodeEx("Per-Class Metrics @0.5", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    std::vector<std::string> classNames;
    for (const auto& c : state.detectionMetricsA.perClass) {
        classNames.push_back(c.className);
    }
    if (state.compareTwoModels) {
        for (const auto& c : state.detectionMetricsB.perClass) {
            if (std::find(classNames.begin(), classNames.end(), c.className) == classNames.end()) {
                classNames.push_back(c.className);
            }
        }
    }

    const auto findClass = [](const DetectionMetrics& metrics, const std::string& className) -> const ClassAveragePrecision* {
        for (const auto& c : metrics.perClass) {
            if (c.className == className) {
                return &c;
            }
        }
        return nullptr;
    };
    const auto drawValue = [](std::optional<float> value) {
        ImGui::TableNextColumn();
        value ? ImGui::Text("%.3f", *value) : ImGui::TextDisabled("--");
    };
    const auto drawSlotColumns = [&](const ClassAveragePrecision* c) {
        drawValue(c ? std::optional<float>(c->averagePrecision) : std::nullopt);
        drawValue(c ? classPrecision(*c) : std::nullopt);
        drawValue(c ? classRecall(*c) : std::nullopt);
    };

    const int columns = state.compareTwoModels ? 7 : 4;
    if (ImGui::BeginTable("BatchEvalPerClassAP", columns, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Class");
        if (state.compareTwoModels) {
            ImGui::TableSetupColumn("AP (A)");
            ImGui::TableSetupColumn("P (A)");
            ImGui::TableSetupColumn("R (A)");
            ImGui::TableSetupColumn("AP (B)");
            ImGui::TableSetupColumn("P (B)");
            ImGui::TableSetupColumn("R (B)");
        } else {
            ImGui::TableSetupColumn("AP@0.5");
            ImGui::TableSetupColumn("P");
            ImGui::TableSetupColumn("R");
        }
        ImGui::TableHeadersRow();
        for (const auto& className : classNames) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(className.c_str());
            drawSlotColumns(findClass(state.detectionMetricsA, className));
            if (state.compareTwoModels) {
                drawSlotColumns(findClass(state.detectionMetricsB, className));
            }
        }
        ImGui::EndTable();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("P/R are final values at the model's confidence threshold (IoU 0.5).");
    }
    ImGui::TreePop();
}

// Drawn inside a fixed-size scrollable child (rather than a collapsible tree
// node) so Model A's and Model B's matrices can sit side by side and both be
// visible at once, however many classes either one has.
void drawConfusionMatrix(
    const char* label, const ClassificationMetrics& metrics, int slotIndex, BenchmarkImageFilters& filters) {
    ImGui::TextUnformatted(label);
    if (metrics.confusionMatrix.empty()) {
        ImGui::TextDisabled("No data.");
        return;
    }

    std::vector<std::string> labels;
    for (const auto& [trueLabel, predictions] : metrics.confusionMatrix) {
        if (std::find(labels.begin(), labels.end(), trueLabel) == labels.end()) {
            labels.push_back(trueLabel);
        }
        for (const auto& [predictedLabel, count] : predictions) {
            if (std::find(labels.begin(), labels.end(), predictedLabel) == labels.end()) {
                labels.push_back(predictedLabel);
            }
        }
    }
    std::sort(labels.begin(), labels.end());

    ImGui::PushID(label);
    ImGui::BeginChild("Scroll", ImVec2(0, 220.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 1.0f));
    const int columns = static_cast<int>(labels.size()) + 1;
    if (ImGui::BeginTable("Matrix", columns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("True \\ Pred");
        for (const auto& predictedLabel : labels) {
            ImGui::TableSetupColumn(predictedLabel.c_str());
        }
        ImGui::TableHeadersRow();
        for (size_t row = 0; row < labels.size(); ++row) {
            const std::string& trueLabel = labels[row];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(trueLabel.c_str());
            const auto trueRow = metrics.confusionMatrix.find(trueLabel);
            for (size_t col = 0; col < labels.size(); ++col) {
                const std::string& predictedLabel = labels[col];
                ImGui::TableNextColumn();
                int count = 0;
                if (trueRow != metrics.confusionMatrix.end()) {
                    const auto cell = trueRow->second.find(predictedLabel);
                    if (cell != trueRow->second.end()) {
                        count = cell->second;
                    }
                }
                if (count == 0) {
                    ImGui::TextDisabled("0");
                    continue;
                }
                const bool selected = filters.confusionCell && filters.confusionCell->slotIndex == slotIndex
                    && filters.confusionCell->trueLabel == trueLabel
                    && filters.confusionCell->predictedLabel == predictedLabel;
                ImGui::PushID(static_cast<int>(row * labels.size() + col));
                ImGui::PushStyleColor(ImGuiCol_Text, predictedLabel == trueLabel ? kGoodColor : kBadColor);
                if (ImGui::Selectable(std::to_string(count).c_str(), selected)) {
                    if (selected) {
                        filters.confusionCell.reset();
                    } else {
                        filters.confusionCell = BenchmarkConfusionCellFilter{trueLabel, predictedLabel, slotIndex};
                    }
                }
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Show images: %s predicted as %s", trueLabel.c_str(), predictedLabel.c_str());
                }
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::PopID();
}

// One row with a value column per model; when comparing two, the better
// value is highlighted.
void drawMetricRow(const BenchmarkState& state, const char* label, float valueA, float valueB, bool higherIsBetter) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);

    const float values[2] = {valueA, valueB};
    const int slotCount = state.compareTwoModels ? 2 : 1;
    for (int i = 0; i < slotCount; ++i) {
        const float other = values[1 - i];
        const bool wins = slotCount == 2 && (higherIsBetter ? values[i] > other : values[i] < other);
        ImGui::TableNextColumn();
        if (wins) {
            ImGui::TextColored(kGoodColor, "%.4f", values[i]);
        } else {
            ImGui::Text("%.4f", values[i]);
        }
    }
}

int countAnomalousImages(const BenchmarkResult& result) {
    int count = 0;
    for (const auto& image : result.images) {
        if (image.anomalyResult && image.anomalyResult->isAnomalous) {
            count++;
        }
    }
    return count;
}

void drawAggregateMetrics(BenchmarkState& state) {
    const int slotCount = state.compareTwoModels ? 2 : 1;
    if (!ImGui::BeginTable("BatchEvalMetrics", slotCount + 1, ImGuiTableFlags_Borders)) {
        return;
    }
    ImGui::TableSetupColumn("Metric");
    for (int i = 0; i < slotCount; ++i) {
        ImGui::TableSetupColumn(slotLabel(state, i));
    }
    ImGui::TableHeadersRow();

    const BenchmarkResult* results[2] = {&state.resultA, &state.resultB};
    if (state.taskMode == ModelTask::Anomaly) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Flagged anomalous");
        for (int i = 0; i < slotCount; ++i) {
            ImGui::TableNextColumn();
            ImGui::Text(
                "%d / %d", countAnomalousImages(*results[i]), static_cast<int>(results[i]->images.size()));
        }
    } else if (state.hasGroundTruth) {
        if (state.taskMode == ModelTask::Detection) {
            drawMetricRow(
                state, "mAP@0.5", state.detectionMetricsA.meanAveragePrecision,
                state.detectionMetricsB.meanAveragePrecision, true);
        } else {
            drawMetricRow(
                state, "Accuracy", state.classificationMetricsA.accuracy, state.classificationMetricsB.accuracy, true);
        }
    } else {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("No ground truth provided -- showing timing only.");
    }

    const TimingStats& timingA = state.resultA.timing;
    const TimingStats& timingB = state.resultB.timing;
    drawMetricRow(
        state, "Mean Inference (ms)", static_cast<float>(timingA.meanMs), static_cast<float>(timingB.meanMs), false);
    drawMetricRow(
        state, "Median Inference (ms)", static_cast<float>(timingA.medianMs), static_cast<float>(timingB.medianMs),
        false);
    drawMetricRow(
        state, "P95 Inference (ms)", static_cast<float>(timingA.p95Ms), static_cast<float>(timingB.p95Ms), false);

    ImGui::EndTable();

    if (!state.hasGroundTruth) {
        return;
    }
    if (state.taskMode == ModelTask::Detection) {
        drawPerClassApTable(state);
    } else if (state.taskMode == ModelTask::Classification) {
        if (state.compareTwoModels) {
            ImGui::Columns(2, "BatchEvalConfusionMatrices");
            drawConfusionMatrix("Model A Confusion Matrix", state.classificationMetricsA, 0, state.filters);
            ImGui::NextColumn();
            drawConfusionMatrix("Model B Confusion Matrix", state.classificationMetricsB, 1, state.filters);
            ImGui::Columns(1);
        } else {
            drawConfusionMatrix("Confusion Matrix", state.classificationMetricsA, 0, state.filters);
        }
    }
}

void drawImageList(BenchmarkState& state) {
    BenchmarkImageFilters& filters = state.filters;
    const ModelTask mode = state.taskMode;

    ImGui::BeginChild("BatchEvalImageList", ImVec2(280.0f, 640.0f), true);

    if (mode != ModelTask::Anomaly) {
        ImGui::BeginDisabled(!state.hasGroundTruth);
        if (mode == ModelTask::Detection) {
            static const char* kErrorLabels[] = {"Any", "Any error", "False positives", "Missed"};
            int errorIndex = static_cast<int>(filters.errorFilter);
            if (ImGui::Combo("Errors", &errorIndex, kErrorLabels, IM_ARRAYSIZE(kErrorLabels))) {
                filters.errorFilter = static_cast<BenchmarkErrorFilter>(errorIndex);
            }
        } else {
            static const char* kErrorLabels[] = {"Any", "Misclassified"};
            int errorIndex = filters.errorFilter == BenchmarkErrorFilter::Any ? 0 : 1;
            if (ImGui::Combo("Errors", &errorIndex, kErrorLabels, IM_ARRAYSIZE(kErrorLabels))) {
                filters.errorFilter = errorIndex == 0 ? BenchmarkErrorFilter::Any : BenchmarkErrorFilter::AnyError;
            }
        }
        ImGui::EndDisabled();

        const std::vector<std::string> classNames =
            collectBenchmarkClassNames(mode, state.resultA, state.resultB);
        if (!filters.cls.className.empty()
            && std::find(classNames.begin(), classNames.end(), filters.cls.className) == classNames.end()) {
            filters.cls.className.clear();
        }
        drawClassFilter("Class", filters.cls, classNames);

        if (state.compareTwoModels) {
            ImGui::Checkbox("Models disagree", &filters.modelsDisagreeOnly);
        }
    }

    drawTextSearch("##BatchEvalImageFilter", "Search filename...", state.imageSearch);
    drawConfidenceSort("Sort", state.imageSort);

    if (mode == ModelTask::Detection) {
        static const char* kBasisLabels[] = {"Mean", "Lowest box", "Highest box"};
        int basisIndex = static_cast<int>(filters.confidenceBasis);
        if (ImGui::Combo("Conf. basis", &basisIndex, kBasisLabels, IM_ARRAYSIZE(kBasisLabels))) {
            filters.confidenceBasis = static_cast<BenchmarkConfidenceBasis>(basisIndex);
        }
    }

    drawConfidenceFilter("Confidence filter", filters.confidence);

    if (mode == ModelTask::Detection) {
        drawPresenceFilter("Detections", filters.detections, "Any", "Has detections", "No detections");
    }

    // (filename, sort key) -- the key is computed once per image rather
    // than inside the sort comparator.
    std::vector<std::pair<std::string, float>> entries;
    for (const auto& image : state.resultA.images) {
        if (!state.imageSearch.passes(image.imageFilename)) {
            continue;
        }
        const BenchmarkImageResult* imageB = findBenchmarkImage(state.resultB, image.imageFilename);
        if (!benchmarkImagePassesFilters(mode, state.hasGroundTruth, &image, imageB, filters)) {
            continue;
        }
        entries.emplace_back(
            image.imageFilename, benchmarkImageSortConfidence(mode, &image, imageB, filters.confidenceBasis));
    }

    sortByConfidence(entries, state.imageSort, [](const auto& entry) { return std::optional<float>(entry.second); });

    if (filters.confusionCell) {
        const char* slotName = slotLabel(state, filters.confusionCell->slotIndex);
        ImGui::TextWrapped(
            "Cell: %s -> %s (%s)", filters.confusionCell->trueLabel.c_str(),
            filters.confusionCell->predictedLabel.c_str(), slotName);
        ImGui::SameLine();
        if (ImGui::SmallButton("x##ClearConfusionCell")) {
            filters.confusionCell.reset();
        }
    }

    ImGui::Separator();
    ImGui::BeginChild("BatchEvalImageListScroll", ImVec2(0, 0), false);
    for (const auto& [filename, sortKey] : entries) {
        const bool selected = state.selectedImageFilename && *state.selectedImageFilename == filename;
        if (ImGui::Selectable(filename.c_str(), selected)) {
            state.selectedImageFilename = filename;
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();
}

void drawGroundTruthLine(ModelTask mode, const BenchmarkImageResult* image) {
    if (image == nullptr || !image->hasGroundTruth) {
        return;
    }
    if (mode == ModelTask::Classification) {
        ImGui::Text("Ground truth: %s", image->groundTruthLabel.c_str());
    } else {
        std::string classes;
        for (const auto& box : image->groundTruthBoxes) {
            if (!classes.empty()) {
                classes += ", ";
            }
            classes += box.className;
        }
        ImGui::Text("Ground truth boxes: %s", classes.empty() ? "(none)" : classes.c_str());
    }
}

bool detectionMatchesGroundTruth(const Detection& detection, const std::vector<GroundTruthBox>& groundTruthBoxes) {
    for (const auto& box : groundTruthBoxes) {
        if (box.className == detection.className
            && computeRotatedIoU(detection.box, detection.rotationDegrees, box.box, box.rotationDegrees) >= 0.5f) {
            return true;
        }
    }
    return false;
}

void drawSlotPredictionText(ModelTask mode, const BenchmarkImageResult* image) {
    if (image == nullptr) {
        ImGui::TextDisabled("No result.");
        return;
    }
    if (mode == ModelTask::Classification) {
        if (image->predictions.empty()) {
            ImGui::TextDisabled("No prediction.");
            return;
        }
        const auto& top1 = image->predictions.front();
        if (!image->hasGroundTruth) {
            ImGui::Text("Top-1: %s (%.1f%%)", top1.className.c_str(), top1.probability * 100.0f);
        } else {
            const bool correct = top1.className == image->groundTruthLabel;
            ImGui::TextColored(
                correct ? kGoodColor : kBadColor, "Top-1: %s (%.1f%%)", top1.className.c_str(),
                top1.probability * 100.0f);
        }
    } else if (mode == ModelTask::Anomaly) {
        if (!image->anomalyResult) {
            ImGui::TextDisabled("No result.");
            return;
        }
        const bool anomalous = image->anomalyResult->isAnomalous;
        ImGui::TextColored(
            anomalous ? kBadColor : kGoodColor, "Score: %.2f (%s)", image->anomalyResult->score,
            anomalous ? "ANOMALOUS" : "normal");
    } else {
        if (image->detections.empty()) {
            ImGui::TextDisabled("No detections.");
            return;
        }
        for (const auto& detection : image->detections) {
            ImGui::Bullet();
            ImGui::SameLine();
            if (!image->hasGroundTruth) {
                ImGui::Text("%s: %.1f%%", detection.className.c_str(), detection.confidence * 100.0f);
            } else {
                const bool matched = detectionMatchesGroundTruth(detection, image->groundTruthBoxes);
                ImGui::TextColored(
                    matched ? kGoodColor : kBadColor, "%s: %.1f%%", detection.className.c_str(),
                    detection.confidence * 100.0f);
            }
        }
    }
}

void drawSelectedImageDetail(BenchmarkState& state) {
    ImGui::SameLine();
    ImGui::BeginChild("BatchEvalImageDetail", ImVec2(0, 640.0f), true);

    if (!state.selectedImageFilename) {
        ImGui::TextDisabled("Select an image to view details.");
        ImGui::EndChild();
        return;
    }

    const std::string& filename = *state.selectedImageFilename;
    const BenchmarkImageResult* imageA = findBenchmarkImage(state.resultA, filename);
    const BenchmarkImageResult* imageB = findBenchmarkImage(state.resultB, filename);
    drawGroundTruthLine(state.taskMode, imageA != nullptr && imageA->hasGroundTruth ? imageA : imageB);

    const int slotCount = state.compareTwoModels ? 2 : 1;
    const BenchmarkImageResult* images[2] = {imageA, imageB};
    if (slotCount == 2) {
        ImGui::Columns(2, "BatchEvalDetailImages");
    }
    for (int i = 0; i < slotCount; ++i) {
        const BenchmarkPreviewTexture& preview = state.previewTextures[static_cast<size_t>(i)];
        ImGui::TextUnformatted(slotLabel(state, i));
        if (preview.previewTexture != 0 && preview.previewTextureWidth > 0 && preview.previewTextureHeight > 0) {
            // Fill the available column width (single wide image, or half
            // the pane per model when comparing two) rather than a small
            // fixed size -- detection labels are unreadable when the whole
            // photo is crushed down to a thumbnail.
            const float maxWidth = ImGui::GetContentRegionAvail().x;
            const ImVec2 size =
                fitImageToRegion(preview.previewTextureWidth, preview.previewTextureHeight, maxWidth, 520.0f);
            ImGui::Image((void*)(intptr_t)preview.previewTexture, size);
            drawHoverEnlargedImage(preview.previewTexture, preview.previewTextureWidth, preview.previewTextureHeight, size);
        } else {
            ImGui::TextDisabled("No preview.");
        }
        drawSlotPredictionText(state.taskMode, images[i]);
        if (slotCount == 2) {
            ImGui::NextColumn();
        }
    }
    if (slotCount == 2) {
        ImGui::Columns(1);
    }

    ImGui::EndChild();
}

void drawRunBody(
    BenchmarkState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    drawSourceModeToggle(state);
    if (state.sourceMode == BenchmarkSourceMode::LocalFolder) {
        drawLocalFolderAndGroundTruthPickers(state);
    } else {
        drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);
        if (!state.labelStudioAutoFetchStatus.empty()) {
            ImGui::TextDisabled("%s", state.labelStudioAutoFetchStatus.c_str());
        }
    }
    drawSampleCheckbox(state);
    ImGui::Separator();

    drawRunBar(state, session);

    if (state.runState == BenchmarkRunState::Complete) {
        ImGui::Separator();
        if (state.resultA.totalFilesInFolder > state.resultA.imagesFound) {
            ImGui::TextDisabled(
                "Sampled %d of %d images in the folder.", state.resultA.imagesFound,
                state.resultA.totalFilesInFolder);
        }
        drawAggregateMetrics(state);
        ImGui::Separator();
        drawImageList(state);
        drawSelectedImageDetail(state);
    }
}

void drawPickerPopup(BenchmarkState& state) {
    const std::optional<std::filesystem::path> picked = drawPathPicker(state.picker, "ModelEvalPicker");
    if (!picked) {
        return;
    }
    const std::string path = picked->string();
    switch (state.filePickerTarget) {
        case BenchmarkPickerTarget::ImageFolder:
            state.imageFolderPath = path;
            resetBenchmarkResults(state);
            break;
        case BenchmarkPickerTarget::SlotAModel:
            state.slots[0].onnxPath = path;
            loadBenchmarkSlot(state, 0);
            break;
        case BenchmarkPickerTarget::SlotBModel:
            state.slots[1].onnxPath = path;
            loadBenchmarkSlot(state, 1);
            break;
        case BenchmarkPickerTarget::GroundTruthJson:
            state.groundTruthJsonPath = path;
            loadBenchmarkGroundTruth(state);
            break;
    }
}

} // namespace

void drawBenchmarkTabContent(
    BenchmarkState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    drawModelCountToggle(state);
    ImGui::Separator();

    if (state.compareTwoModels) {
        ImGui::Columns(2, "ModelEvalSlots");
        drawSlotConfig(state, 0);
        ImGui::NextColumn();
        drawSlotConfig(state, 1);
        ImGui::Columns(1);
    } else {
        drawSlotConfig(state, 0);
    }
    ImGui::Separator();

    drawRunBody(state, session, onOpenLabelStudioWindow);

    drawPickerPopup(state);
}

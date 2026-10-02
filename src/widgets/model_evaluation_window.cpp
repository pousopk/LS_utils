#include "widgets/model_evaluation_window.hpp"

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

void drawTaskModeToggle(ModelEvaluationState& state) {
    if (ImGui::RadioButton("Detection", state.taskMode == ComparisonTaskMode::Detection)) {
        if (state.taskMode != ComparisonTaskMode::Detection) {
            state.taskMode = ComparisonTaskMode::Detection;
            resetModelEvaluationResults(state);
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Classification", state.taskMode == ComparisonTaskMode::Classification)) {
        if (state.taskMode != ComparisonTaskMode::Classification) {
            state.taskMode = ComparisonTaskMode::Classification;
            resetModelEvaluationResults(state);
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Anomaly", state.taskMode == ComparisonTaskMode::Anomaly)) {
        if (state.taskMode != ComparisonTaskMode::Anomaly) {
            state.taskMode = ComparisonTaskMode::Anomaly;
            resetModelEvaluationResults(state);
        }
    }
}

void drawModelCountToggle(ModelEvaluationState& state) {
    if (ImGui::RadioButton("Single Model", !state.compareTwoModels)) {
        if (state.compareTwoModels) {
            state.compareTwoModels = false;
            resetModelEvaluationResults(state);
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Compare Two Models", state.compareTwoModels)) {
        if (!state.compareTwoModels) {
            state.compareTwoModels = true;
            resetModelEvaluationResults(state);
        }
    }
}

void drawSlotConfig(ModelEvaluationState& state, int slotIndex) {
    ModelSlotConfig& slot = state.slots[static_cast<size_t>(slotIndex)];
    ImGui::PushID(slotIndex);
    ImGui::TextUnformatted(slotIndex == 0 ? (state.compareTwoModels ? "Model A" : "Model") : "Model B");

    float* confThreshold = (state.taskMode == ComparisonTaskMode::Detection) ? &slot.confThreshold : nullptr;
    float* nmsThreshold = (state.taskMode == ComparisonTaskMode::Detection) ? &slot.nmsThreshold : nullptr;

    const FilePickerTarget modelTarget = slotIndex == 0 ? FilePickerTarget::SlotAModel : FilePickerTarget::SlotBModel;
    const FilePickerTarget classNamesTarget =
        slotIndex == 0 ? FilePickerTarget::SlotAClassNames : FilePickerTarget::SlotBClassNames;

    const bool loadClicked = drawModelSlotConfigFields(
        slot.onnxPath, slot.classNamesPath, slot.inputWidth, slot.inputHeight, confThreshold, nmsThreshold, nullptr,
        slot.autoDetectStatus, slot.loadError,
        [&state, &slot, modelTarget]() {
            state.filePickerTarget = modelTarget;
            openPathPicker(state.picker, PathPickerMode::File, "Pick ONNX Model", slot.onnxPath);
        },
        [&state, &slot, classNamesTarget]() {
            state.filePickerTarget = classNamesTarget;
            openPathPicker(state.picker, PathPickerMode::File, "Pick Class Names File", slot.classNamesPath);
        });

    if (!slot.engineStatus.empty()) {
        ImGui::TextDisabled("%s", slot.engineStatus.c_str());
    }

    if (state.taskMode == ComparisonTaskMode::Anomaly) {
        ImGui::SliderFloat("Anomaly Threshold", &slot.anomalyThreshold, 0.0f, 1.0f, "%.2f");
        ImGui::InputFloat("Score Min", &slot.anomalyScoreMin);
        ImGui::InputFloat("Score Max", &slot.anomalyScoreMax);
    }

    if (loadClicked) {
        loadModelSlot(slot, state.taskMode);
    }

    ImGui::PopID();
}

// ---- Batch body ----

void drawBatchSourceModeToggle(BatchRuntime& batch) {
    auto switchTo = [&batch](BatchEvalSourceMode mode) {
        if (batch.sourceMode == mode) {
            return;
        }
        batch.sourceMode = mode;
        batch.runState = BatchEvalRunState::NotStarted;
        batch.resultA = BatchEvaluationResult{};
        batch.resultB = BatchEvaluationResult{};
        batch.selectedImageFilename.reset();
        batch.imageFolderPath.clear();
    };

    ImGui::TextUnformatted("Source");
    if (ImGui::RadioButton("Local Folder##BatchSource", batch.sourceMode == BatchEvalSourceMode::LocalFolder)) {
        switchTo(BatchEvalSourceMode::LocalFolder);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(
            "Label Studio Project##BatchSource", batch.sourceMode == BatchEvalSourceMode::LabelStudioProject)) {
        switchTo(BatchEvalSourceMode::LabelStudioProject);
    }
}

void drawLocalFolderAndGroundTruthPickers(ModelEvaluationState& state) {
    ImGui::TextWrapped(
        "Image Folder: %s", state.batch.imageFolderPath.empty() ? "(none)" : state.batch.imageFolderPath.c_str());
    if (ImGui::Button("Browse Folder...")) {
        state.filePickerTarget = FilePickerTarget::ImageFolder;
        openPathPicker(state.picker, PathPickerMode::Folder, "Pick Image Folder", state.batch.imageFolderPath);
    }

    ImGui::TextWrapped(
        "Ground Truth (optional): %s",
        state.batch.groundTruthJsonPath.empty() ? "(none)" : state.batch.groundTruthJsonPath.c_str());
    if (ImGui::Button("Browse Ground Truth...")) {
        state.filePickerTarget = FilePickerTarget::GroundTruthJson;
        openPathPicker(
            state.picker, PathPickerMode::File, "Pick Ground Truth JSON", state.batch.groundTruthJsonPath);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Ground Truth")) {
        state.batch.groundTruthJsonPath.clear();
        loadBatchEvalGroundTruth(state.batch);
    }
    if (!state.batch.groundTruthStatus.empty()) {
        ImGui::TextDisabled("%s", state.batch.groundTruthStatus.c_str());
    }
}

void drawSampleCheckbox(BatchRuntime& batch) {
    ImGui::Checkbox("Randomly sample", &batch.sampleEnabled);
    ImGui::SameLine();
    ImGui::BeginDisabled(!batch.sampleEnabled);
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("images##SampleSize", &batch.sampleSize);
    ImGui::EndDisabled();
    if (batch.sampleSize < 1) {
        batch.sampleSize = 1;
    }
    if (batch.sampleEnabled) {
        ImGui::TextDisabled(
            "Evaluates a random subset instead of the whole folder -- useful for a quick check on a huge "
            "dataset.");
    }
}

void drawRunBar(ModelEvaluationState& state, const LabelStudioSessionState& session) {
    ImGui::Separator();
    if (state.batch.runState == BatchEvalRunState::Running) {
        const std::string label = !state.batch.lastProgress.phaseLabel.empty()
            ? state.batch.lastProgress.phaseLabel
            : (state.batch.lastProgress.currentSlot == 2 ? "Model B" : "Model A");
        ImGui::Text(
            "%s: %d / %d", label.c_str(), state.batch.lastProgress.completed, state.batch.lastProgress.total);
        const float fraction = state.batch.lastProgress.total > 0
            ? static_cast<float>(state.batch.lastProgress.completed) /
                static_cast<float>(state.batch.lastProgress.total)
            : 0.0f;
        ImGui::ProgressBar(fraction);
        if (ImGui::Button("Cancel")) {
            state.batch.worker.requestCancel();
        }
        return;
    }

    if (state.batch.runState == BatchEvalRunState::Cancelled) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Run cancelled.");
    }

    const auto slotLoaded = [&state](int index) {
        const ModelSlotConfig& slot = state.slots[static_cast<size_t>(index)];
        if (state.taskMode == ComparisonTaskMode::Detection) {
            return slot.detectionModel != nullptr;
        }
        if (state.taskMode == ComparisonTaskMode::Classification) {
            return slot.classificationModel != nullptr;
        }
        return slot.anomalyModel != nullptr;
    };
    const bool slotALoaded = slotLoaded(0);
    const bool slotBLoaded = slotLoaded(1);
    const bool modelsLoaded = slotALoaded && (!state.compareTwoModels || slotBLoaded);
    const bool sourceReady = state.batch.sourceMode == BatchEvalSourceMode::LocalFolder
        ? !state.batch.imageFolderPath.empty()
        : !session.baseUrl.empty() && session.activeProjectId > 0 && !session.apiToken.empty();
    const bool canRun = modelsLoaded && sourceReady;

    ImGui::BeginDisabled(!canRun);
    if (ImGui::Button("Run")) {
        startBatchEvaluationRun(state.taskMode, state.compareTwoModels, state.slots, state.batch, session);
    }
    ImGui::EndDisabled();
    if (!canRun) {
        const char* modelsHint =
            state.compareTwoModels ? "Load both models and " : "Load the model and ";
        const char* sourceHint = state.batch.sourceMode == BatchEvalSourceMode::LocalFolder
            ? "pick an image folder to run."
            : "fill in the Label Studio connection to run.";
        ImGui::TextDisabled("%s%s", modelsHint, sourceHint);
    }
}

void drawMetricRow(const char* label, float valueA, float valueB, bool higherIsBetter) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);

    const bool aWins = higherIsBetter ? (valueA > valueB) : (valueA < valueB);
    const bool bWins = higherIsBetter ? (valueB > valueA) : (valueB < valueA);

    ImGui::TableNextColumn();
    if (aWins) {
        ImGui::TextColored(kGoodColor, "%.4f", valueA);
    } else {
        ImGui::Text("%.4f", valueA);
    }

    ImGui::TableNextColumn();
    if (bWins) {
        ImGui::TextColored(kGoodColor, "%.4f", valueB);
    } else {
        ImGui::Text("%.4f", valueB);
    }
}

void drawPerClassApTable(const ModelEvaluationState& state) {
    if (!ImGui::TreeNodeEx("Per-Class Metrics @0.5", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    std::vector<std::string> classNames;
    for (const auto& c : state.batch.detectionMetricsA.perClass) {
        classNames.push_back(c.className);
    }
    if (state.compareTwoModels) {
        for (const auto& c : state.batch.detectionMetricsB.perClass) {
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
            drawSlotColumns(findClass(state.batch.detectionMetricsA, className));
            if (state.compareTwoModels) {
                drawSlotColumns(findClass(state.batch.detectionMetricsB, className));
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
    const char* label, const ClassificationMetrics& metrics, int slotIndex, BatchEvalImageFilters& filters) {
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
                        filters.confusionCell = BatchEvalConfusionCellFilter{trueLabel, predictedLabel, slotIndex};
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

void drawMetricRowSingle(const char* label, float value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text("%.4f", value);
}

int countAnomalousImages(const BatchEvaluationResult& result) {
    int count = 0;
    for (const auto& image : result.images) {
        if (image.anomalyResult && image.anomalyResult->isAnomalous) {
            count++;
        }
    }
    return count;
}

void drawAggregateMetricsSingle(ModelEvaluationState& state) {
    if (!ImGui::BeginTable("BatchEvalMetrics", 2, ImGuiTableFlags_Borders)) {
        return;
    }
    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Model");
    ImGui::TableHeadersRow();

    if (state.taskMode == ComparisonTaskMode::Anomaly) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Flagged anomalous");
        ImGui::TableNextColumn();
        ImGui::Text(
            "%d / %d", countAnomalousImages(state.batch.resultA),
            static_cast<int>(state.batch.resultA.images.size()));
    } else if (state.batch.hasGroundTruth) {
        if (state.taskMode == ComparisonTaskMode::Detection) {
            drawMetricRowSingle("mAP@0.5", state.batch.detectionMetricsA.meanAveragePrecision);
        } else {
            drawMetricRowSingle("Accuracy", state.batch.classificationMetricsA.accuracy);
        }
    } else {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("No ground truth provided -- showing timing only.");
    }

    drawMetricRowSingle("Mean Inference (ms)", static_cast<float>(state.batch.resultA.timing.meanMs));
    drawMetricRowSingle("Median Inference (ms)", static_cast<float>(state.batch.resultA.timing.medianMs));
    drawMetricRowSingle("P95 Inference (ms)", static_cast<float>(state.batch.resultA.timing.p95Ms));

    ImGui::EndTable();

    if (state.batch.hasGroundTruth) {
        if (state.taskMode == ComparisonTaskMode::Detection) {
            drawPerClassApTable(state);
        } else {
            drawConfusionMatrix("Confusion Matrix", state.batch.classificationMetricsA, 0, state.batch.filters);
        }
    }
}

void drawAggregateMetrics(ModelEvaluationState& state) {
    if (!state.compareTwoModels) {
        drawAggregateMetricsSingle(state);
        return;
    }

    if (!ImGui::BeginTable("BatchEvalMetrics", 3, ImGuiTableFlags_Borders)) {
        return;
    }
    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Model A");
    ImGui::TableSetupColumn("Model B");
    ImGui::TableHeadersRow();

    if (state.taskMode == ComparisonTaskMode::Anomaly) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Flagged anomalous");
        ImGui::TableNextColumn();
        ImGui::Text(
            "%d / %d", countAnomalousImages(state.batch.resultA),
            static_cast<int>(state.batch.resultA.images.size()));
        ImGui::TableNextColumn();
        ImGui::Text(
            "%d / %d", countAnomalousImages(state.batch.resultB),
            static_cast<int>(state.batch.resultB.images.size()));
    } else if (state.batch.hasGroundTruth) {
        if (state.taskMode == ComparisonTaskMode::Detection) {
            drawMetricRow(
                "mAP@0.5", state.batch.detectionMetricsA.meanAveragePrecision,
                state.batch.detectionMetricsB.meanAveragePrecision, true);
        } else {
            drawMetricRow(
                "Accuracy", state.batch.classificationMetricsA.accuracy, state.batch.classificationMetricsB.accuracy,
                true);
        }
    } else {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("No ground truth provided -- showing timing only.");
    }

    drawMetricRow(
        "Mean Inference (ms)", static_cast<float>(state.batch.resultA.timing.meanMs),
        static_cast<float>(state.batch.resultB.timing.meanMs), false);
    drawMetricRow(
        "Median Inference (ms)", static_cast<float>(state.batch.resultA.timing.medianMs),
        static_cast<float>(state.batch.resultB.timing.medianMs), false);
    drawMetricRow(
        "P95 Inference (ms)", static_cast<float>(state.batch.resultA.timing.p95Ms),
        static_cast<float>(state.batch.resultB.timing.p95Ms), false);

    ImGui::EndTable();

    if (state.batch.hasGroundTruth) {
        if (state.taskMode == ComparisonTaskMode::Detection) {
            drawPerClassApTable(state);
        } else {
            ImGui::Columns(2, "BatchEvalConfusionMatrices");
            drawConfusionMatrix("Model A Confusion Matrix", state.batch.classificationMetricsA, 0, state.batch.filters);
            ImGui::NextColumn();
            drawConfusionMatrix("Model B Confusion Matrix", state.batch.classificationMetricsB, 1, state.batch.filters);
            ImGui::Columns(1);
        }
    }
}

void drawImageList(ModelEvaluationState& state) {
    BatchEvalImageFilters& filters = state.batch.filters;
    const ComparisonTaskMode mode = state.taskMode;

    ImGui::BeginChild("BatchEvalImageList", ImVec2(280.0f, 640.0f), true);

    if (mode != ComparisonTaskMode::Anomaly) {
        ImGui::BeginDisabled(!state.batch.hasGroundTruth);
        if (mode == ComparisonTaskMode::Detection) {
            static const char* kErrorLabels[] = {"Any", "Any error", "False positives", "Missed"};
            int errorIndex = static_cast<int>(filters.errorFilter);
            if (ImGui::Combo("Errors", &errorIndex, kErrorLabels, IM_ARRAYSIZE(kErrorLabels))) {
                filters.errorFilter = static_cast<BatchEvalErrorFilter>(errorIndex);
            }
        } else {
            static const char* kErrorLabels[] = {"Any", "Misclassified"};
            int errorIndex = filters.errorFilter == BatchEvalErrorFilter::Any ? 0 : 1;
            if (ImGui::Combo("Errors", &errorIndex, kErrorLabels, IM_ARRAYSIZE(kErrorLabels))) {
                filters.errorFilter = errorIndex == 0 ? BatchEvalErrorFilter::Any : BatchEvalErrorFilter::AnyError;
            }
        }
        ImGui::EndDisabled();

        const std::vector<std::string> classNames =
            collectBatchEvalClassNames(mode, state.batch.resultA, state.batch.resultB);
        if (!filters.cls.className.empty()
            && std::find(classNames.begin(), classNames.end(), filters.cls.className) == classNames.end()) {
            filters.cls.className.clear();
        }
        drawClassFilter("Class", filters.cls, classNames);

        if (state.compareTwoModels) {
            ImGui::Checkbox("Models disagree", &filters.modelsDisagreeOnly);
        }
    }

    drawTextSearch("##BatchEvalImageFilter", "Search filename...", state.batch.imageSearch);
    drawConfidenceSort("Sort", state.batch.imageSort);

    if (mode == ComparisonTaskMode::Detection) {
        static const char* kBasisLabels[] = {"Mean", "Lowest box", "Highest box"};
        int basisIndex = static_cast<int>(filters.confidenceBasis);
        if (ImGui::Combo("Conf. basis", &basisIndex, kBasisLabels, IM_ARRAYSIZE(kBasisLabels))) {
            filters.confidenceBasis = static_cast<BatchEvalConfidenceBasis>(basisIndex);
        }
    }

    drawConfidenceFilter("Confidence filter", filters.confidence);

    if (mode == ComparisonTaskMode::Detection) {
        drawPresenceFilter("Detections", filters.detections, "Any", "Has detections", "No detections");
    }

    // (filename, sort key) -- the key is computed once per image rather
    // than inside the sort comparator.
    std::vector<std::pair<std::string, float>> entries;
    for (const auto& image : state.batch.resultA.images) {
        if (!state.batch.imageSearch.passes(image.imageFilename)) {
            continue;
        }
        const BatchImageResult* imageB = findBatchImage(state.batch.resultB, image.imageFilename);
        if (!batchEvalImagePassesFilters(mode, state.batch.hasGroundTruth, &image, imageB, filters)) {
            continue;
        }
        entries.emplace_back(
            image.imageFilename, batchEvalImageSortConfidence(mode, &image, imageB, filters.confidenceBasis));
    }

    sortByConfidence(entries, state.batch.imageSort, [](const auto& entry) { return std::optional<float>(entry.second); });

    if (filters.confusionCell) {
        const char* slotName =
            filters.confusionCell->slotIndex == 0 ? (state.compareTwoModels ? "Model A" : "Model") : "Model B";
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
        const bool selected = state.batch.selectedImageFilename && *state.batch.selectedImageFilename == filename;
        if (ImGui::Selectable(filename.c_str(), selected)) {
            state.batch.selectedImageFilename = filename;
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();
}

void drawGroundTruthLine(ComparisonTaskMode mode, const BatchImageResult* image) {
    if (image == nullptr || !image->hasGroundTruth) {
        return;
    }
    if (mode == ComparisonTaskMode::Classification) {
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

void drawSlotPredictionText(ComparisonTaskMode mode, const BatchImageResult* image) {
    if (image == nullptr) {
        ImGui::TextDisabled("No result.");
        return;
    }
    if (mode == ComparisonTaskMode::Classification) {
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
    } else if (mode == ComparisonTaskMode::Anomaly) {
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

void drawSelectedImageDetail(ModelEvaluationState& state) {
    ImGui::SameLine();
    ImGui::BeginChild("BatchEvalImageDetail", ImVec2(0, 640.0f), true);

    if (!state.batch.selectedImageFilename) {
        ImGui::TextDisabled("Select an image to view details.");
        ImGui::EndChild();
        return;
    }

    const std::string& filename = *state.batch.selectedImageFilename;
    const BatchImageResult* imageA = findBatchImage(state.batch.resultA, filename);
    const BatchImageResult* imageB = findBatchImage(state.batch.resultB, filename);
    drawGroundTruthLine(state.taskMode, imageA != nullptr && imageA->hasGroundTruth ? imageA : imageB);

    const int slotCount = state.compareTwoModels ? 2 : 1;
    const BatchImageResult* images[2] = {imageA, imageB};
    if (slotCount == 2) {
        ImGui::Columns(2, "BatchEvalDetailImages");
    }
    for (int i = 0; i < slotCount; ++i) {
        const BatchPreviewTexture& preview = state.batch.previewTextures[static_cast<size_t>(i)];
        ImGui::TextUnformatted(i == 0 ? (state.compareTwoModels ? "Model A" : "Model") : "Model B");
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

void drawBatchBody(
    ModelEvaluationState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    drawBatchSourceModeToggle(state.batch);
    if (state.batch.sourceMode == BatchEvalSourceMode::LocalFolder) {
        drawLocalFolderAndGroundTruthPickers(state);
    } else {
        drawLabelStudioSessionSummary(session, onOpenLabelStudioWindow);
        if (!state.batch.labelStudioAutoFetchStatus.empty()) {
            ImGui::TextDisabled("%s", state.batch.labelStudioAutoFetchStatus.c_str());
        }
    }
    drawSampleCheckbox(state.batch);
    ImGui::Separator();

    drawRunBar(state, session);

    if (state.batch.runState == BatchEvalRunState::Complete) {
        ImGui::Separator();
        if (state.batch.resultA.totalFilesInFolder > state.batch.resultA.imagesFound) {
            ImGui::TextDisabled(
                "Sampled %d of %d images in the folder.", state.batch.resultA.imagesFound,
                state.batch.resultA.totalFilesInFolder);
        }
        drawAggregateMetrics(state);
        ImGui::Separator();
        drawImageList(state);
        drawSelectedImageDetail(state);
    }
}

void drawPickerPopup(ModelEvaluationState& state) {
    const std::optional<std::filesystem::path> picked = drawPathPicker(state.picker, "ModelEvalPicker");
    if (!picked) {
        return;
    }
    const std::string path = picked->string();
    switch (state.filePickerTarget) {
        case FilePickerTarget::ImageFolder:
            state.batch.imageFolderPath = path;
            resetModelEvaluationResults(state);
            break;
        case FilePickerTarget::SlotAModel:
            state.slots[0].onnxPath = path;
            applyAutoDetectToModelSlot(state.slots[0], state.taskMode);
            break;
        case FilePickerTarget::SlotAClassNames:
            state.slots[0].classNamesPath = path;
            break;
        case FilePickerTarget::SlotBModel:
            state.slots[1].onnxPath = path;
            applyAutoDetectToModelSlot(state.slots[1], state.taskMode);
            break;
        case FilePickerTarget::SlotBClassNames:
            state.slots[1].classNamesPath = path;
            break;
        case FilePickerTarget::GroundTruthJson:
            state.batch.groundTruthJsonPath = path;
            loadBatchEvalGroundTruth(state.batch);
            break;
    }
}

} // namespace

void drawBenchmarkTabContent(
    ModelEvaluationState& state, const LabelStudioSessionState& session,
    const std::function<void()>& onOpenLabelStudioWindow) {
    drawTaskModeToggle(state);
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

    drawBatchBody(state, session, onOpenLabelStudioWindow);

    drawPickerPopup(state);
}

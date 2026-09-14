#include "widgets/model_evaluation_window.hpp"

#include "objects/media_source.hpp"
#include "widgets/file_browser_utils.hpp"
#include "widgets/label_studio_window.hpp"
#include "widgets/model_slot_config_widget.hpp"
#include "widgets/tooltip_helpers.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <filesystem>
#include <optional>

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

void drawSourceToggle(ModelEvaluationState& state) {
    ImGui::TextUnformatted("Source");
    if (ImGui::RadioButton("Live", state.source == EvaluationSourceMode::Live)) {
        state.source = EvaluationSourceMode::Live;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Batch", state.source == EvaluationSourceMode::Batch)) {
        state.source = EvaluationSourceMode::Batch;
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
        [&state, modelTarget]() {
            state.filePickerTarget = modelTarget;
            state.filePickerOpen = true;
        },
        [&state, classNamesTarget]() {
            state.filePickerTarget = classNamesTarget;
            state.filePickerOpen = true;
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

// ---- Live body ----

void drawLiveSourcePicker(ModelEvaluationState& state, std::vector<CameraSession>& sessions) {
    ImGui::TextUnformatted("Live Source");
    if (ImGui::RadioButton("Existing session", state.live.sourceMode == LiveSourceMode::ExistingSession)) {
        state.live.sourceMode = LiveSourceMode::ExistingSession;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Loaded file", state.live.sourceMode == LiveSourceMode::LoadedFile)) {
        state.live.sourceMode = LiveSourceMode::LoadedFile;
    }

    if (state.live.sourceMode == LiveSourceMode::ExistingSession) {
        const std::string preview = state.live.sessionId.empty() ? "Select a session..." : state.live.sessionId;
        if (ImGui::BeginCombo("Session", preview.c_str())) {
            for (const auto& session : sessions) {
                const bool selected = (session.id == state.live.sessionId);
                if (ImGui::Selectable(session.id.c_str(), selected)) {
                    state.live.sessionId = session.id;
                }
            }
            ImGui::EndCombo();
        }
    } else if (state.live.sourceMode == LiveSourceMode::LoadedFile) {
        ImGui::TextWrapped(
            "File: %s", state.live.loadedSourcePath.empty() ? "(none)" : state.live.loadedSourcePath.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Browse...##LiveSourceFile")) {
            state.filePickerTarget = FilePickerTarget::LiveSourceFile;
            state.filePickerOpen = true;
        }
    }
}

void drawLiveSlotCanvas(const ModelEvaluationState& state, int slotIndex) {
    const LiveRuntimeSlot& liveSlot = state.live.liveSlots[static_cast<size_t>(slotIndex)];
    const char* label = slotIndex == 0 ? (state.compareTwoModels ? "Model A" : "Model") : "Model B";
    ImGui::BeginGroup();

    if (state.taskMode == ComparisonTaskMode::Detection) {
        float averageConfidence = 0.0f;
        if (!liveSlot.latestDetections.empty()) {
            float sum = 0.0f;
            for (const auto& detection : liveSlot.latestDetections) {
                sum += detection.confidence;
            }
            averageConfidence = sum / static_cast<float>(liveSlot.latestDetections.size());
        }
        ImGui::Text(
            "%s -- %.1f ms, %d detections, avg conf %.2f", label, liveSlot.latestInferenceMs,
            static_cast<int>(liveSlot.latestDetections.size()), averageConfidence);
    } else if (state.taskMode == ComparisonTaskMode::Anomaly) {
        ImGui::Text(
            "%s -- %.1f ms, score %.2f (%s)", label, liveSlot.latestInferenceMs, liveSlot.latestAnomalyResult.score,
            liveSlot.latestAnomalyResult.isAnomalous ? "ANOMALOUS" : "normal");
    } else {
        ImGui::Text("%s -- %.1f ms", label, liveSlot.latestInferenceMs);
    }

    if (liveSlot.texture != 0 && liveSlot.textureWidth > 0 && liveSlot.textureHeight > 0) {
        const ImVec2 size = fitImageToRegion(liveSlot.textureWidth, liveSlot.textureHeight, 480.0f, 360.0f);
        ImGui::Image((void*)(intptr_t)liveSlot.texture, size);
    } else {
        ImGui::TextDisabled("No frame yet.");
    }

    if (state.taskMode == ComparisonTaskMode::Classification) {
        if (liveSlot.latestPredictions.empty()) {
            ImGui::TextDisabled("No prediction yet.");
        } else {
            const auto& top1 = liveSlot.latestPredictions.front();
            ImGui::Text("Top-1: %s (%.1f%%)", top1.className.c_str(), top1.probability * 100.0f);
            ImGui::TextUnformatted("Top-5:");
            const size_t topCount = std::min<size_t>(5, liveSlot.latestPredictions.size());
            for (size_t i = 0; i < topCount; ++i) {
                const auto& prediction = liveSlot.latestPredictions[i];
                ImGui::BulletText("%s: %.1f%%", prediction.className.c_str(), prediction.probability * 100.0f);
            }
        }
    }

    if (!liveSlot.runtimeError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "%s", liveSlot.runtimeError.c_str());
    }

    ImGui::EndGroup();
}

void drawLiveBody(ModelEvaluationState& state, std::vector<CameraSession>& sessions) {
    drawLiveSourcePicker(state, sessions);
    ImGui::Separator();

    if (state.compareTwoModels) {
        ImGui::Columns(2, "LiveCanvases");
        drawLiveSlotCanvas(state, 0);
        ImGui::NextColumn();
        drawLiveSlotCanvas(state, 1);
        ImGui::Columns(1);
    } else {
        drawLiveSlotCanvas(state, 0);
    }

    if (state.taskMode == ComparisonTaskMode::Detection && state.compareTwoModels) {
        ImGui::Separator();
        ImGui::Text(
            "Agreement: %d / %d matched (A has %d, B has %d)", state.live.latestAgreement.matchedPairs,
            std::max(state.live.latestAgreement.totalA, state.live.latestAgreement.totalB),
            state.live.latestAgreement.totalA, state.live.latestAgreement.totalB);
    }
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
        state.batch.folderPickerExplorerDir = state.batch.imageFolderPath;
        state.batch.folderPickerOpen = true;
    }

    ImGui::TextWrapped(
        "Ground Truth (optional): %s",
        state.batch.groundTruthJsonPath.empty() ? "(none)" : state.batch.groundTruthJsonPath.c_str());
    if (ImGui::Button("Browse Ground Truth...")) {
        state.filePickerTarget = FilePickerTarget::GroundTruthJson;
        state.filePickerOpen = true;
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
    if (!ImGui::TreeNodeEx("Per-Class AP@0.5", ImGuiTreeNodeFlags_DefaultOpen)) {
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

    const auto findAp = [](const DetectionMetrics& metrics, const std::string& className) -> std::optional<float> {
        for (const auto& c : metrics.perClass) {
            if (c.className == className) {
                return c.averagePrecision;
            }
        }
        return std::nullopt;
    };

    const int columns = state.compareTwoModels ? 3 : 2;
    if (ImGui::BeginTable("BatchEvalPerClassAP", columns, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Class");
        ImGui::TableSetupColumn(state.compareTwoModels ? "AP@0.5 (A)" : "AP@0.5");
        if (state.compareTwoModels) {
            ImGui::TableSetupColumn("AP@0.5 (B)");
        }
        ImGui::TableHeadersRow();
        for (const auto& className : classNames) {
            const std::optional<float> apA = findAp(state.batch.detectionMetricsA, className);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(className.c_str());
            ImGui::TableNextColumn();
            apA ? ImGui::Text("%.3f", *apA) : ImGui::TextDisabled("--");
            if (state.compareTwoModels) {
                const std::optional<float> apB = findAp(state.batch.detectionMetricsB, className);
                ImGui::TableNextColumn();
                apB ? ImGui::Text("%.3f", *apB) : ImGui::TextDisabled("--");
            }
        }
        ImGui::EndTable();
    }
    ImGui::TreePop();
}

// Drawn inside a fixed-size scrollable child (rather than a collapsible tree
// node) so Model A's and Model B's matrices can sit side by side and both be
// visible at once, however many classes either one has.
void drawConfusionMatrix(const char* label, const ClassificationMetrics& metrics) {
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
        for (const auto& trueLabel : labels) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(trueLabel.c_str());
            const auto trueRow = metrics.confusionMatrix.find(trueLabel);
            for (const auto& predictedLabel : labels) {
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
                } else if (predictedLabel == trueLabel) {
                    ImGui::TextColored(kGoodColor, "%d", count);
                } else {
                    ImGui::TextColored(kBadColor, "%d", count);
                }
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
            drawConfusionMatrix("Confusion Matrix", state.batch.classificationMetricsA);
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
            drawConfusionMatrix("Model A Confusion Matrix", state.batch.classificationMetricsA);
            ImGui::NextColumn();
            drawConfusionMatrix("Model B Confusion Matrix", state.batch.classificationMetricsB);
            ImGui::Columns(1);
        }
    }
}

void drawImageList(ModelEvaluationState& state) {
    ImGui::BeginChild("BatchEvalImageList", ImVec2(280.0f, 640.0f), true);
    ImGui::BeginDisabled(!state.batch.hasGroundTruth);
    ImGui::Checkbox("Mismatches only", &state.batch.mismatchesOnly);
    ImGui::EndDisabled();
    ImGui::InputTextWithHint("##BatchEvalImageFilter", "Search filename...", &state.batch.imageListFilter);

    static const char* kSortLabels[] = {"Filename", "Confidence (low first)", "Confidence (high first)"};
    int sortIndex = static_cast<int>(state.batch.imageSortMode);
    if (ImGui::Combo("Sort", &sortIndex, kSortLabels, IM_ARRAYSIZE(kSortLabels))) {
        state.batch.imageSortMode = static_cast<BatchEvalImageSortMode>(sortIndex);
    }

    static const char* kConfidenceFilterLabels[] = {"None", "< threshold", "> threshold"};
    int confidenceFilterIndex = static_cast<int>(state.batch.confidenceFilterMode);
    if (ImGui::Combo(
            "Confidence filter", &confidenceFilterIndex, kConfidenceFilterLabels,
            IM_ARRAYSIZE(kConfidenceFilterLabels))) {
        state.batch.confidenceFilterMode = static_cast<BatchEvalConfidenceFilterMode>(confidenceFilterIndex);
    }
    ImGui::BeginDisabled(state.batch.confidenceFilterMode == BatchEvalConfidenceFilterMode::None);
    ImGui::SliderFloat("Threshold", &state.batch.confidenceFilterThreshold, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();

    if (state.taskMode == ComparisonTaskMode::Detection) {
        static const char* kDetectionPresenceLabels[] = {"Any", "Has detections", "No detections"};
        int detectionPresenceIndex = static_cast<int>(state.batch.detectionPresenceFilter);
        if (ImGui::Combo(
                "Detections", &detectionPresenceIndex, kDetectionPresenceLabels,
                IM_ARRAYSIZE(kDetectionPresenceLabels))) {
            state.batch.detectionPresenceFilter =
                static_cast<BatchEvalDetectionPresenceFilter>(detectionPresenceIndex);
        }
    }

    std::vector<std::string> filenames;
    for (const auto& image : state.batch.resultA.images) {
        if (!fileNameMatchesFilter(std::filesystem::path(image.imageFilename), state.batch.imageListFilter)) {
            continue;
        }
        if (state.batch.mismatchesOnly && state.batch.hasGroundTruth &&
            !isBatchEvalImageMismatch(state.taskMode, state.batch, image.imageFilename)) {
            continue;
        }
        if (!batchEvalImagePassesConfidenceFilter(state.taskMode, state.batch, image.imageFilename)) {
            continue;
        }
        if (!batchEvalImagePassesDetectionPresenceFilter(state.taskMode, state.batch, image.imageFilename)) {
            continue;
        }
        filenames.push_back(image.imageFilename);
    }

    if (state.batch.imageSortMode == BatchEvalImageSortMode::ConfidenceAscending) {
        std::sort(filenames.begin(), filenames.end(), [&state](const std::string& a, const std::string& b) {
            return batchEvalImageSortConfidence(state.taskMode, state.batch, a) <
                batchEvalImageSortConfidence(state.taskMode, state.batch, b);
        });
    } else if (state.batch.imageSortMode == BatchEvalImageSortMode::ConfidenceDescending) {
        std::sort(filenames.begin(), filenames.end(), [&state](const std::string& a, const std::string& b) {
            return batchEvalImageSortConfidence(state.taskMode, state.batch, a) >
                batchEvalImageSortConfidence(state.taskMode, state.batch, b);
        });
    }

    ImGui::Separator();
    ImGui::BeginChild("BatchEvalImageListScroll", ImVec2(0, 0), false);
    for (const auto& filename : filenames) {
        const bool selected = state.batch.selectedImageFilename && *state.batch.selectedImageFilename == filename;
        if (ImGui::Selectable(filename.c_str(), selected)) {
            state.batch.selectedImageFilename = filename;
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();
}

const BatchImageResult* findBatchImage(const BatchEvaluationResult& result, const std::string& filename) {
    for (const auto& image : result.images) {
        if (image.imageFilename == filename) {
            return &image;
        }
    }
    return nullptr;
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

void drawFolderPickerPopup(ModelEvaluationState& state) {
    if (state.batch.folderPickerOpen) {
        ImGui::OpenPopup("Pick Image Folder");
        state.batch.folderPickerOpen = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Pick Image Folder", nullptr)) {
        std::string selected;
        if (drawDirectoryBrowser(
                state.batch.folderPickerExplorerDir, &selected, "BatchEvalFolderPickerDirs",
                state.batch.folderPickerFilter)) {
            state.batch.imageFolderPath = selected;
            resetModelEvaluationResults(state);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void drawFilePickerPopup(ModelEvaluationState& state) {
    if (state.filePickerOpen) {
        ImGui::OpenPopup("Pick Model Evaluation File");
        state.filePickerOpen = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Pick Model Evaluation File", nullptr)) {
        namespace fs = std::filesystem;

        const std::string dirBeforeBrowse = state.filePickerDir;
        drawDirectoryBrowser(state.filePickerDir, nullptr, "ModelEvalFilePickerDirs", state.filePickerFilter);
        if (state.filePickerDir != dirBeforeBrowse) {
            state.filePickerSelectedFile.clear();
        }

        ImGui::BeginChild("ModelEvalFilePickerFiles", ImVec2(0, 260.0f), true);
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
                case FilePickerTarget::SlotAModel:
                    state.slots[0].onnxPath = state.filePickerSelectedFile;
                    applyAutoDetectToModelSlot(state.slots[0], state.taskMode);
                    break;
                case FilePickerTarget::SlotAClassNames:
                    state.slots[0].classNamesPath = state.filePickerSelectedFile;
                    break;
                case FilePickerTarget::SlotBModel:
                    state.slots[1].onnxPath = state.filePickerSelectedFile;
                    applyAutoDetectToModelSlot(state.slots[1], state.taskMode);
                    break;
                case FilePickerTarget::SlotBClassNames:
                    state.slots[1].classNamesPath = state.filePickerSelectedFile;
                    break;
                case FilePickerTarget::GroundTruthJson:
                    state.batch.groundTruthJsonPath = state.filePickerSelectedFile;
                    loadBatchEvalGroundTruth(state.batch);
                    break;
                case FilePickerTarget::LiveSourceFile:
                    state.live.loadedSourcePath = state.filePickerSelectedFile;
                    state.live.loadedSource = std::make_unique<MediaSource>(state.filePickerSelectedFile);
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

void drawModelEvaluationWindow(
    bool* show, ModelEvaluationState& state, std::vector<CameraSession>& sessions,
    const LabelStudioSessionState& session, const std::function<void()>& onOpenLabelStudioWindow) {
    if (!*show) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(1200.0f, 1000.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Model Evaluation", show)) {
        ImGui::End();
        return;
    }

    drawTaskModeToggle(state);
    drawModelCountToggle(state);
    ImGui::Separator();

    drawSourceToggle(state);
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

    if (state.source == EvaluationSourceMode::Live) {
        drawLiveBody(state, sessions);
    } else {
        drawBatchBody(state, session, onOpenLabelStudioWindow);
    }

    ImGui::End();

    drawFolderPickerPopup(state);
    drawFilePickerPopup(state);
}

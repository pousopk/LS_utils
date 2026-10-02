#include "manager/benchmark_state.hpp"

#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace {

// Fixed, hidden scratch location for Benchmark's LabelStudioProject
// source mode downloads -- cleared and recreated at the start of every
// run, and deliberately separate from Label Assistant's own scratch
// folder so the two windows never collide.
std::string benchmarkScratchFolder() {
    return (std::filesystem::temp_directory_path() / "vision_app_batch_eval_download").string();
}

cv::Mat annotateAnomalyHeatmap(const cv::Mat& frame, const AnomalyResult& result) {
    cv::Mat annotated = frame.clone();
    if (!result.heatmap.empty()) {
        cv::Mat normalized;
        cv::normalize(result.heatmap, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat colorized;
        cv::applyColorMap(normalized, colorized, cv::COLORMAP_JET);
        cv::addWeighted(annotated, 0.6, colorized, 0.4, 0.0, annotated);
    }

    const float scale = std::clamp(static_cast<float>(frame.cols) / 800.0f, 1.0f, 3.0f);
    const int textThickness = std::max(2, static_cast<int>(std::lround(scale * 2.0f)));
    const double fontScale = 0.85 * scale;
    const cv::Scalar color = result.isAnomalous ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

    char buffer[64];
    std::snprintf(
        buffer, sizeof(buffer), "Score: %.2f -- %s", result.score, result.isAnomalous ? "ANOMALOUS" : "normal");

    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(buffer, cv::FONT_HERSHEY_SIMPLEX, fontScale, textThickness, &baseline);
    cv::rectangle(
        annotated, cv::Point(4, 4), cv::Point(4 + textSize.width + 4, 4 + textSize.height + baseline + 4),
        cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(
        annotated, buffer, cv::Point(6, 6 + textSize.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, color,
        textThickness, cv::LINE_AA);
    return annotated;
}

void syncBenchmarkSelectedPreview(BenchmarkState& state) {
    if (state.selectedImageFilename == state.renderedPreviewFilename) {
        return;
    }
    state.renderedPreviewFilename = state.selectedImageFilename;

    if (!state.selectedImageFilename) {
        return;
    }
    const std::string& filename = *state.selectedImageFilename;
    const cv::Mat frame = cv::imread(state.imageFolderPath + "/" + filename);
    if (frame.empty()) {
        return;
    }

    const BenchmarkResult* results[2] = {&state.resultA, &state.resultB};
    for (int i = 0; i < 2; ++i) {
        BenchmarkPreviewTexture& preview = state.previewTextures[static_cast<size_t>(i)];
        const BenchmarkImageResult* image = findBenchmarkImage(*results[i], filename);
        if (image == nullptr) {
            continue;
        }
        if (preview.previewTexture == 0) {
            glGenTextures(1, &preview.previewTexture);
        }
        const ModelSlotConfig& slot = state.slots[static_cast<size_t>(i)];
        cv::Mat toUpload = frame;
        if (state.taskMode == ModelTask::Detection) {
            toUpload = annotateDetections(frame, image->detections);
        } else if (state.taskMode == ModelTask::Anomaly && slot.anomalyModel) {
            const AnomalyResult anomaly = slot.anomalyModel->infer(frame, slot.anomalyThreshold);
            toUpload = annotateAnomalyHeatmap(frame, anomaly);
        }
        uploadFrameToTexture(
            preview.previewTexture, toUpload, preview.previewTextureWidth, preview.previewTextureHeight);
    }
}

} // namespace

void loadBenchmarkGroundTruth(BenchmarkState& state) {
    state.hasGroundTruth = false;
    state.groundTruth = LabelStudioImportResult{};
    state.groundTruthStatus.clear();

    if (state.groundTruthJsonPath.empty()) {
        return;
    }

    state.groundTruth = loadLabelStudioExport(state.groundTruthJsonPath);
    if (!state.groundTruth.error.empty()) {
        state.groundTruthStatus = state.groundTruth.error;
        return;
    }

    state.hasGroundTruth = true;
    state.groundTruthStatus = std::to_string(state.groundTruth.images.size()) + " labeled images loaded";
    if (state.groundTruth.skippedCount > 0) {
        state.groundTruthStatus += " (" + std::to_string(state.groundTruth.skippedCount) + " entries skipped)";
    }
}

void syncBenchmarkLabelStudioAutoFetch(BenchmarkState& state, const LabelStudioSessionState& session) {
    if (session.baseUrl.empty() || session.activeProjectId <= 0 || session.apiToken.empty()) {
        return;
    }
    const std::string key = session.baseUrl + "|" + std::to_string(session.activeProjectId) + "|" + session.apiToken;
    if (key == state.lastAutoFetchKey) {
        return;
    }
    state.lastAutoFetchKey = key;

    // isDetection doesn't matter here -- dataImageKey extraction doesn't
    // depend on it, and from_name/to_name (which does) are unused by this
    // download-only flow.
    const LabelStudioLabelingConfig config =
        fetchLabelStudioLabelingConfig(session.baseUrl, session.activeProjectId, session.apiToken, true);

    if (config.error.empty()) {
        state.labelStudioDataImageKey = config.dataImageKey;
        state.labelStudioAutoFetchStatus = "Auto-filled from Label Studio project settings";
    } else {
        state.labelStudioAutoFetchStatus = "Labeling config: " + config.error;
    }
}

void startBenchmarkRun(BenchmarkState& state, const LabelStudioSessionState& session) {
    BenchmarkRunConfig config;
    config.mode = state.taskMode;
    config.source = state.sourceMode;

    if (state.sourceMode == BenchmarkSourceMode::LabelStudioProject) {
        const std::string scratchFolder = benchmarkScratchFolder();
        std::error_code ec;
        std::filesystem::remove_all(scratchFolder, ec);
        std::filesystem::create_directories(scratchFolder, ec);

        config.labelStudioBaseUrl = session.baseUrl;
        config.labelStudioProjectId = session.activeProjectId;
        config.labelStudioApiToken = session.apiToken;
        config.labelStudioDataImageKey = state.labelStudioDataImageKey;
        config.scratchFolderPath = scratchFolder;
        state.imageFolderPath = scratchFolder;
        // groundTruth/hasGroundTruth are left default -- the worker fills
        // them in itself during the download phase.
    } else {
        config.imageFolderPath = state.imageFolderPath;
        config.hasGroundTruth = state.hasGroundTruth;
        if (state.hasGroundTruth) {
            config.groundTruth = state.groundTruth;
        }
    }

    config.runSlotB = state.compareTwoModels;
    config.sampleSize = state.sampleEnabled ? state.sampleSize : 0;
    config.detectionModelA = state.slots[0].detectionModel;
    config.detectionModelB = state.slots[1].detectionModel;
    config.classificationModelA = state.slots[0].classificationModel;
    config.classificationModelB = state.slots[1].classificationModel;
    config.anomalyModelA = state.slots[0].anomalyModel;
    config.anomalyModelB = state.slots[1].anomalyModel;
    config.anomalyThresholdA = state.slots[0].anomalyThreshold;
    config.anomalyThresholdB = state.slots[1].anomalyThreshold;
    config.confThresholdA = state.slots[0].confThreshold;
    config.nmsThresholdA = state.slots[0].nmsThreshold;
    config.confThresholdB = state.slots[1].confThreshold;
    config.nmsThresholdB = state.slots[1].nmsThreshold;

    resetBenchmarkResults(state);
    state.worker.start(std::move(config));
    state.runState = BenchmarkRunState::Running;
}

void updateBenchmarkState(BenchmarkState& state, const LabelStudioSessionState& session) {
    if (state.runState == BenchmarkRunState::Running) {
        state.lastProgress = state.worker.progress();

        BenchmarkRunResult runResult;
        if (state.worker.tryTakeResult(runResult)) {
            if (runResult.cancelled) {
                state.runState = BenchmarkRunState::Cancelled;
            } else {
                state.resultA = std::move(runResult.slotA);
                state.resultB = std::move(runResult.slotB);
                // Recompute from the actual run results rather than trusting
                // the pre-run flag: LabelStudioProject source mode never sets
                // state.hasGroundTruth itself (ground truth is fetched and
                // attached per-image inside the worker), so the aggregate
                // metrics/mismatch-filter gate below would otherwise stay
                // permanently false even when every image matched real
                // ground truth.
                state.hasGroundTruth =
                    state.resultA.imagesWithGroundTruth > 0 || state.resultB.imagesWithGroundTruth > 0;
                if (state.taskMode == ModelTask::Detection) {
                    state.detectionMetricsA = computeDetectionMetrics(toDetectionEvaluationItems(state.resultA));
                    state.detectionMetricsB = computeDetectionMetrics(toDetectionEvaluationItems(state.resultB));
                } else if (state.taskMode == ModelTask::Classification) {
                    state.classificationMetricsA =
                        computeClassificationMetrics(toClassificationEvaluationItems(state.resultA));
                    state.classificationMetricsB =
                        computeClassificationMetrics(toClassificationEvaluationItems(state.resultB));
                }
                // Anomaly mode: no ground-truth metrics, only the flagged count.
                state.runState = BenchmarkRunState::Complete;
            }
        }
    }

    syncBenchmarkLabelStudioAutoFetch(state, session);

    syncBenchmarkSelectedPreview(state);
}

void resetBenchmarkResults(BenchmarkState& state) {
    state.runState = BenchmarkRunState::NotStarted;
    state.resultA = BenchmarkResult{};
    state.resultB = BenchmarkResult{};
    state.detectionMetricsA = DetectionMetrics{};
    state.detectionMetricsB = DetectionMetrics{};
    state.classificationMetricsA = ClassificationMetrics{};
    state.classificationMetricsB = ClassificationMetrics{};
    state.selectedImageFilename.reset();

    // Drop filter values that don't exist in the current mode/count (a
    // no-op unless taskMode/compareTwoModels just changed).
    BenchmarkImageFilters& filters = state.filters;
    filters.confusionCell.reset();
    if (!state.compareTwoModels) {
        filters.modelsDisagreeOnly = false;
    }
    if (state.taskMode != ModelTask::Detection
        && (filters.errorFilter == BenchmarkErrorFilter::FalsePositives
            || filters.errorFilter == BenchmarkErrorFilter::Missed)) {
        filters.errorFilter = BenchmarkErrorFilter::Any;
    }
}

void loadBenchmarkSlot(BenchmarkState& state, int slotIndex) {
    ModelSlotConfig& slot = state.slots[static_cast<size_t>(slotIndex)];
    loadModelSlot(slot);
    if (slotIndex == 0 && isModelSlotLoaded(slot) && slot.task != state.taskMode) {
        state.taskMode = slot.task;
        resetBenchmarkResults(state);
    }
}

bool modelBTaskMismatch(const BenchmarkState& state) {
    return isModelSlotLoaded(state.slots[0]) && isModelSlotLoaded(state.slots[1])
        && state.slots[1].task != state.slots[0].task;
}

bool benchmarkSlotsReady(const BenchmarkState& state) {
    if (!isModelSlotLoaded(state.slots[0])) {
        return false;
    }
    return !state.compareTwoModels || (isModelSlotLoaded(state.slots[1]) && !modelBTaskMismatch(state));
}

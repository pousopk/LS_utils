#include "manager/model_evaluation_state.hpp"

#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {

std::vector<std::string> loadClassNamesFromFile(const std::string& path) {
    std::vector<std::string> names;
    if (path.empty()) {
        return names;
    }
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

} // namespace

namespace {

// Same rotate-around-pivot construction as labeling_state.cpp's
// rotatedBoxCorners (phase B), reimplemented locally here rather than
// imported: this file is the model-inference/evaluation layer and
// shouldn't depend on labeling_state.hpp, which is specifically the
// Labeling window's editor state.
std::array<cv::Point, 4> rotatedDetectionBoxCorners(const cv::Rect& box, float rotationDegrees) {
    const double theta = static_cast<double>(rotationDegrees) * CV_PI / 180.0;
    const double cosT = std::cos(theta);
    const double sinT = std::sin(theta);
    const cv::Point2d pivot(box.x, box.y);
    const cv::Point2d localCorners[4] = {
        cv::Point2d(0.0, 0.0),
        cv::Point2d(box.width, 0.0),
        cv::Point2d(box.width, box.height),
        cv::Point2d(0.0, box.height),
    };
    std::array<cv::Point, 4> result;
    for (int i = 0; i < 4; ++i) {
        const double rx = localCorners[i].x * cosT - localCorners[i].y * sinT;
        const double ry = localCorners[i].x * sinT + localCorners[i].y * cosT;
        result[i] = cv::Point(
            static_cast<int>(std::round(pivot.x + rx)), static_cast<int>(std::round(pivot.y + ry)));
    }
    return result;
}

} // namespace

// Box/label size scales UP with the frame's own resolution relative to an
// 800px-wide baseline (never down -- a photo smaller than the baseline
// still gets full-size text, since shrinking further only makes small
// source photos worse), and every label is drawn on a solid background so
// it stays readable regardless of what color the photo behind it is
// (green-on-green or green-on-white was otherwise invisible no matter the
// font size).
cv::Mat annotateDetections(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat annotated = frame.clone();
    const float scale = std::clamp(static_cast<float>(frame.cols) / 800.0f, 1.0f, 3.0f);
    const int boxThickness = std::max(2, static_cast<int>(std::lround(scale * 2.0f)));
    const int textThickness = std::max(2, static_cast<int>(std::lround(scale * 2.0f)));
    const double fontScale = 0.85 * scale;
    for (const auto& detection : detections) {
        if (detection.rotationDegrees == 0.0f) {
            cv::rectangle(annotated, detection.box, cv::Scalar(0, 255, 0), boxThickness);
        } else {
            const auto corners = rotatedDetectionBoxCorners(detection.box, detection.rotationDegrees);
            const cv::Point* pts = corners.data();
            const int numPts = static_cast<int>(corners.size());
            cv::polylines(annotated, &pts, &numPts, 1, /*isClosed=*/true, cv::Scalar(0, 255, 0), boxThickness);
        }

        const std::string label =
            detection.className + " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        int baseline = 0;
        const cv::Size textSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, textThickness, &baseline);

        // Prefer just above the box; drop to just inside its top edge
        // instead if that would clip off the top of the frame.
        const int margin = 4;
        const int aboveTop = detection.box.y - margin - textSize.height - baseline;
        const int top = aboveTop >= 0 ? aboveTop : detection.box.y + margin;
        const cv::Point textOrigin(detection.box.x, top + textSize.height);

        cv::rectangle(
            annotated, cv::Point(textOrigin.x - 2, top - 2),
            cv::Point(textOrigin.x + textSize.width + 2, top + textSize.height + baseline + 2),
            cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(
            annotated, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), textThickness,
            cv::LINE_AA);
    }
    return annotated;
}

void loadModelSlot(ModelSlotConfig& slot, ComparisonTaskMode mode) {
    slot.loadError.clear();
    slot.engineStatus.clear();
    slot.detectionModel.reset();
    slot.classificationModel.reset();
    slot.anomalyModel.reset();

    if (slot.onnxPath.empty()) {
        return;
    }

    const std::vector<std::string> classNames = !slot.classNamesPath.empty()
        ? loadClassNamesFromFile(slot.classNamesPath)
        : slot.autoDetectedClassNames;

    if (mode == ComparisonTaskMode::Detection) {
        std::string error;
        auto model = std::make_shared<YoloModel>(
            slot.onnxPath, classNames, slot.inputWidth, slot.inputHeight, slot.hints, slot.isObbDetectionModel, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        slot.detectionModel = model;
        slot.engineStatus = model->isGpuActive() ? "Engine: GPU" : "Engine: CPU";
    } else if (mode == ComparisonTaskMode::Classification) {
        std::string error;
        auto model = std::make_shared<ClassificationModel>(
            slot.onnxPath, classNames, slot.inputWidth, slot.inputHeight, slot.hints, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        slot.classificationModel = model;
        slot.engineStatus = model->isGpuActive() ? "Engine: GPU" : "Engine: CPU";
    } else {
        std::string error;
        auto model = std::make_shared<AnomalyModel>(
            slot.onnxPath, slot.inputWidth, slot.inputHeight, slot.hints, slot.anomalyScoreMin, slot.anomalyScoreMax,
            error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        slot.anomalyModel = model;
        slot.engineStatus = model->isGpuActive() ? "Engine: GPU" : "Engine: CPU";
    }
}

void applyAutoDetectToModelSlot(ModelSlotConfig& slot, ComparisonTaskMode& taskMode) {
    const ModelAutoDetectResult result = autoDetectModel(slot.onnxPath);

    if (!result.shapeDetected) {
        slot.autoDetectStatus = "Could not read model metadata: " + result.error;
        return;
    }

    slot.inputWidth = result.inputWidth;
    slot.inputHeight = result.inputHeight;
    slot.autoDetectedClassNames = result.classNames;
    slot.hints = result.hints;

    std::string statusMessage =
        "Detected " + std::to_string(result.inputWidth) + "x" + std::to_string(result.inputHeight);
    if (result.suggestedMode == DetectedTaskMode::Detection) {
        statusMessage += ", suggested: Detection";
        taskMode = ComparisonTaskMode::Detection;
    } else if (result.suggestedMode == DetectedTaskMode::Classification) {
        statusMessage += ", suggested: Classification";
        taskMode = ComparisonTaskMode::Classification;
    } else if (result.suggestedMode == DetectedTaskMode::Anomaly) {
        statusMessage += ", suggested: Anomaly";
        taskMode = ComparisonTaskMode::Anomaly;
    }
    if (result.anomalyHints.present) {
        slot.anomalyScoreMin = result.anomalyHints.minimum;
        slot.anomalyScoreMax = result.anomalyHints.maximum;
        slot.anomalyThreshold = result.anomalyHints.threshold;
        statusMessage += ", anomaly score calibration found";
    }
    if (!result.classNames.empty()) {
        statusMessage += ", " + std::to_string(result.classNames.size()) + " class names found";
    }
    const OnnxPreprocessingHints defaultHints;
    if (result.hints.inputScale != defaultHints.inputScale) {
        statusMessage += ", raw pixel range";
    }
    if (result.hints.maintainAspectRatio) {
        statusMessage += ", letterbox padding";
    }
    slot.autoDetectStatus = statusMessage;
}

namespace {

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

void syncBatchEvalSelectedPreview(
    ComparisonTaskMode mode, const std::array<ModelSlotConfig, 2>& slots, const std::string& imageFolderPath,
    BatchRuntime& batch) {
    if (batch.selectedImageFilename == batch.renderedPreviewFilename) {
        return;
    }
    batch.renderedPreviewFilename = batch.selectedImageFilename;

    if (!batch.selectedImageFilename) {
        return;
    }
    const std::string& filename = *batch.selectedImageFilename;
    const cv::Mat frame = cv::imread(imageFolderPath + "/" + filename);
    if (frame.empty()) {
        return;
    }

    const BatchEvaluationResult* results[2] = {&batch.resultA, &batch.resultB};
    for (int i = 0; i < 2; ++i) {
        BatchPreviewTexture& preview = batch.previewTextures[static_cast<size_t>(i)];
        const BatchImageResult* image = findBatchImage(*results[i], filename);
        if (image == nullptr) {
            continue;
        }
        if (preview.previewTexture == 0) {
            glGenTextures(1, &preview.previewTexture);
        }
        const ModelSlotConfig& slot = slots[static_cast<size_t>(i)];
        cv::Mat toUpload = frame;
        if (mode == ComparisonTaskMode::Detection) {
            toUpload = annotateDetections(frame, image->detections);
        } else if (mode == ComparisonTaskMode::Anomaly && slot.anomalyModel) {
            const AnomalyResult anomaly = slot.anomalyModel->infer(frame, slot.anomalyThreshold);
            toUpload = annotateAnomalyHeatmap(frame, anomaly);
        }
        uploadFrameToTexture(
            preview.previewTexture, toUpload, preview.previewTextureWidth, preview.previewTextureHeight);
    }
}

} // namespace

void loadBatchEvalGroundTruth(BatchRuntime& batch) {
    batch.hasGroundTruth = false;
    batch.groundTruth = LabelStudioImportResult{};
    batch.groundTruthStatus.clear();

    if (batch.groundTruthJsonPath.empty()) {
        return;
    }

    batch.groundTruth = loadLabelStudioExport(batch.groundTruthJsonPath);
    if (!batch.groundTruth.error.empty()) {
        batch.groundTruthStatus = batch.groundTruth.error;
        return;
    }

    batch.hasGroundTruth = true;
    batch.groundTruthStatus = std::to_string(batch.groundTruth.images.size()) + " labeled images loaded";
    if (batch.groundTruth.skippedCount > 0) {
        batch.groundTruthStatus += " (" + std::to_string(batch.groundTruth.skippedCount) + " entries skipped)";
    }
}

void syncBatchEvalLabelStudioAutoFetch(BatchRuntime& batch, const LabelStudioSessionState& session) {
    if (session.baseUrl.empty() || session.activeProjectId <= 0 || session.apiToken.empty()) {
        return;
    }
    const std::string key = session.baseUrl + "|" + std::to_string(session.activeProjectId) + "|" + session.apiToken;
    if (key == batch.lastAutoFetchKey) {
        return;
    }
    batch.lastAutoFetchKey = key;

    // isDetection doesn't matter here -- dataImageKey extraction doesn't
    // depend on it, and from_name/to_name (which does) are unused by this
    // download-only flow.
    const LabelStudioLabelingConfig config =
        fetchLabelStudioLabelingConfig(session.baseUrl, session.activeProjectId, session.apiToken, true);

    if (config.error.empty()) {
        batch.labelStudioDataImageKey = config.dataImageKey;
        batch.labelStudioAutoFetchStatus = "Auto-filled from Label Studio project settings";
    } else {
        batch.labelStudioAutoFetchStatus = "Labeling config: " + config.error;
    }
}

namespace {

// Fixed, hidden scratch location for BatchRuntime's LabelStudioProject
// source mode downloads -- cleared and recreated at the start of every
// run, and deliberately separate from Label Assistant's own scratch
// folder so the two windows never collide.
std::string batchEvalScratchFolder() {
    return (std::filesystem::temp_directory_path() / "vision_app_batch_eval_download").string();
}

} // namespace

void startBatchEvaluationRun(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, BatchRuntime& batch,
    const LabelStudioSessionState& session) {
    BatchEvalRunConfig config;
    config.mode = mode;
    config.source = batch.sourceMode;

    if (batch.sourceMode == BatchEvalSourceMode::LabelStudioProject) {
        const std::string scratchFolder = batchEvalScratchFolder();
        std::error_code ec;
        std::filesystem::remove_all(scratchFolder, ec);
        std::filesystem::create_directories(scratchFolder, ec);

        config.labelStudioBaseUrl = session.baseUrl;
        config.labelStudioProjectId = session.activeProjectId;
        config.labelStudioApiToken = session.apiToken;
        config.labelStudioDataImageKey = batch.labelStudioDataImageKey;
        config.scratchFolderPath = scratchFolder;
        batch.imageFolderPath = scratchFolder;
        // groundTruth/hasGroundTruth are left default -- the worker fills
        // them in itself during the download phase.
    } else {
        config.imageFolderPath = batch.imageFolderPath;
        config.hasGroundTruth = batch.hasGroundTruth;
        if (batch.hasGroundTruth) {
            config.groundTruth = batch.groundTruth;
        }
    }

    config.runSlotB = compareTwoModels;
    config.sampleSize = batch.sampleEnabled ? batch.sampleSize : 0;
    config.detectionModelA = slots[0].detectionModel;
    config.detectionModelB = slots[1].detectionModel;
    config.classificationModelA = slots[0].classificationModel;
    config.classificationModelB = slots[1].classificationModel;
    config.anomalyModelA = slots[0].anomalyModel;
    config.anomalyModelB = slots[1].anomalyModel;
    config.anomalyThresholdA = slots[0].anomalyThreshold;
    config.anomalyThresholdB = slots[1].anomalyThreshold;
    config.confThresholdA = slots[0].confThreshold;
    config.nmsThresholdA = slots[0].nmsThreshold;
    config.confThresholdB = slots[1].confThreshold;
    config.nmsThresholdB = slots[1].nmsThreshold;

    batch.resultA = BatchEvaluationResult{};
    batch.resultB = BatchEvaluationResult{};
    batch.detectionMetricsA = DetectionMetrics{};
    batch.detectionMetricsB = DetectionMetrics{};
    batch.classificationMetricsA = ClassificationMetrics{};
    batch.classificationMetricsB = ClassificationMetrics{};
    batch.selectedImageFilename.reset();
    // The confusion matrices a cell filter pointed into are about to be replaced.
    batch.filters.confusionCell.reset();

    batch.worker.start(std::move(config));
    batch.runState = BatchEvalRunState::Running;
}

void updateBatchRuntime(
    ComparisonTaskMode mode, const std::array<ModelSlotConfig, 2>& slots, const std::string& imageFolderPath,
    BatchRuntime& batch, const LabelStudioSessionState& session) {
    if (batch.runState == BatchEvalRunState::Running) {
        batch.lastProgress = batch.worker.progress();

        BatchEvalRunResult runResult;
        if (batch.worker.tryTakeResult(runResult)) {
            if (runResult.cancelled) {
                batch.runState = BatchEvalRunState::Cancelled;
            } else {
                batch.resultA = std::move(runResult.slotA);
                batch.resultB = std::move(runResult.slotB);
                // Recompute from the actual run results rather than trusting
                // the pre-run flag: LabelStudioProject source mode never sets
                // batch.hasGroundTruth itself (ground truth is fetched and
                // attached per-image inside the worker), so the aggregate
                // metrics/mismatch-filter gate below would otherwise stay
                // permanently false even when every image matched real
                // ground truth.
                batch.hasGroundTruth =
                    batch.resultA.imagesWithGroundTruth > 0 || batch.resultB.imagesWithGroundTruth > 0;
                if (mode == ComparisonTaskMode::Detection) {
                    batch.detectionMetricsA = computeDetectionMetrics(toDetectionEvaluationItems(batch.resultA));
                    batch.detectionMetricsB = computeDetectionMetrics(toDetectionEvaluationItems(batch.resultB));
                } else if (mode == ComparisonTaskMode::Classification) {
                    batch.classificationMetricsA =
                        computeClassificationMetrics(toClassificationEvaluationItems(batch.resultA));
                    batch.classificationMetricsB =
                        computeClassificationMetrics(toClassificationEvaluationItems(batch.resultB));
                }
                // Anomaly mode: no metrics this sub-project (no ground truth yet).
                batch.runState = BatchEvalRunState::Complete;
            }
        }
    }

    syncBatchEvalLabelStudioAutoFetch(batch, session);

    syncBatchEvalSelectedPreview(mode, slots, imageFolderPath, batch);
}

void resetModelEvaluationResults(ModelEvaluationState& state) {
    state.batch.runState = BatchEvalRunState::NotStarted;
    state.batch.resultA = BatchEvaluationResult{};
    state.batch.resultB = BatchEvaluationResult{};
    state.batch.detectionMetricsA = DetectionMetrics{};
    state.batch.detectionMetricsB = DetectionMetrics{};
    state.batch.classificationMetricsA = ClassificationMetrics{};
    state.batch.classificationMetricsB = ClassificationMetrics{};
    state.batch.selectedImageFilename.reset();

    // Called on taskMode/compareTwoModels changes: drop filter values that
    // don't exist in the new mode/count.
    BatchEvalImageFilters& filters = state.batch.filters;
    filters.confusionCell.reset();
    if (!state.compareTwoModels) {
        filters.modelsDisagreeOnly = false;
    }
    if (state.taskMode != ComparisonTaskMode::Detection
        && (filters.errorFilter == BatchEvalErrorFilter::FalsePositives
            || filters.errorFilter == BatchEvalErrorFilter::Missed)) {
        filters.errorFilter = BatchEvalErrorFilter::Any;
    }
}

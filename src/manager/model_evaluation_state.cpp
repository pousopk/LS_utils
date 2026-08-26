#include "manager/model_evaluation_state.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
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

void loadModelSlot(ModelSlotConfig& slot, ComparisonTaskMode mode) {
    slot.loadError.clear();
    slot.engineStatus.clear();
    slot.detectionModel.reset();
    slot.classificationModel.reset();

    if (slot.onnxPath.empty()) {
        return;
    }

    const std::vector<std::string> classNames = !slot.classNamesPath.empty()
        ? loadClassNamesFromFile(slot.classNamesPath)
        : slot.autoDetectedClassNames;

    if (mode == ComparisonTaskMode::Detection) {
        std::string error;
        auto model = std::make_shared<YoloModel>(
            slot.onnxPath, classNames, slot.inputWidth, slot.inputHeight, slot.hints, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        slot.detectionModel = model;
        slot.engineStatus = model->isGpuActive() ? "Engine: GPU" : "Engine: CPU";
    } else {
        std::string error;
        auto model = std::make_shared<ClassificationModel>(
            slot.onnxPath, classNames, slot.inputWidth, slot.inputHeight, slot.hints, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        slot.classificationModel = model;
        slot.engineStatus = model->isGpuActive() ? "Engine: GPU" : "Engine: CPU";
    }
}

void applyAutoDetectToModelSlot(ModelSlotConfig& slot, ModelEvaluationState& state) {
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
        state.taskMode = ComparisonTaskMode::Detection;
    } else if (result.suggestedMode == DetectedTaskMode::Classification) {
        statusMessage += ", suggested: Classification";
        state.taskMode = ComparisonTaskMode::Classification;
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

cv::Mat currentLiveFrame(const LiveRuntime& live, std::vector<CameraSession>& sessions) {
    if (live.sourceMode == LiveSourceMode::ExistingSession) {
        CameraSession* session = findSession(sessions, live.sessionId);
        return session != nullptr ? session->latestRawFrame : cv::Mat();
    }
    if (live.sourceMode == LiveSourceMode::LoadedFile && live.loadedSource && live.loadedSource->isValid()) {
        return live.loadedSource->grabFrame();
    }
    return cv::Mat();
}

cv::Mat drawDetectionsOnFrame(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat annotated = frame.clone();
    for (const auto& detection : detections) {
        cv::rectangle(annotated, detection.box, cv::Scalar(0, 255, 0), 2);
        const std::string label =
            detection.className + " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        cv::putText(
            annotated, label, detection.box.tl() + cv::Point(0, -4),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    return annotated;
}

void updateLiveSlot(
    ComparisonTaskMode mode, const ModelSlotConfig& slot, LiveRuntimeSlot& liveSlot, const cv::Mat& frame) {
    if (liveSlot.texture == 0) {
        glGenTextures(1, &liveSlot.texture);
    }

    if (mode == ComparisonTaskMode::Detection) {
        if (liveSlot.workerModel != slot.detectionModel) {
            liveSlot.worker.reset();
            liveSlot.workerModel = slot.detectionModel;
            liveSlot.latestDetections.clear();
            liveSlot.runtimeError.clear();
            if (slot.detectionModel) {
                liveSlot.worker = std::make_unique<InferenceWorker>(slot.detectionModel);
            }
        }
        liveSlot.classificationWorker.reset();
        liveSlot.workerClassificationModel.reset();

        if (!liveSlot.worker) {
            return;
        }
        if (!frame.empty() && liveSlot.worker->isIdle()) {
            liveSlot.worker->submit(frame.clone(), slot.confThreshold, slot.nmsThreshold);
        }
        InferenceWorker::Result result;
        if (liveSlot.worker->tryTakeResult(result)) {
            liveSlot.runtimeError = result.error;
            liveSlot.latestDetections = std::move(result.detections);
            liveSlot.latestInferenceMs = result.inferenceMs;
            if (!frame.empty()) {
                const cv::Mat annotated = drawDetectionsOnFrame(frame, liveSlot.latestDetections);
                uploadFrameToTexture(liveSlot.texture, annotated, liveSlot.textureWidth, liveSlot.textureHeight);
            }
        }
    } else {
        if (liveSlot.workerClassificationModel != slot.classificationModel) {
            liveSlot.classificationWorker.reset();
            liveSlot.workerClassificationModel = slot.classificationModel;
            liveSlot.latestPredictions.clear();
            liveSlot.runtimeError.clear();
            if (slot.classificationModel) {
                liveSlot.classificationWorker =
                    std::make_unique<ClassificationInferenceWorker>(slot.classificationModel);
            }
        }
        liveSlot.worker.reset();
        liveSlot.workerModel.reset();

        if (!liveSlot.classificationWorker) {
            return;
        }
        if (!frame.empty() && liveSlot.classificationWorker->isIdle()) {
            liveSlot.classificationWorker->submit(frame.clone());
        }
        ClassificationInferenceWorker::Result result;
        if (liveSlot.classificationWorker->tryTakeResult(result)) {
            liveSlot.runtimeError = result.error;
            liveSlot.latestPredictions = std::move(result.predictions);
            liveSlot.latestInferenceMs = result.inferenceMs;
            if (!frame.empty()) {
                uploadFrameToTexture(liveSlot.texture, frame, liveSlot.textureWidth, liveSlot.textureHeight);
            }
        }
    }
}

} // namespace

void updateLiveRuntime(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, LiveRuntime& live,
    std::vector<CameraSession>& sessions) {
    const cv::Mat frame = currentLiveFrame(live, sessions);

    const int slotCount = compareTwoModels ? 2 : 1;
    for (int i = 0; i < slotCount; ++i) {
        updateLiveSlot(mode, slots[static_cast<size_t>(i)], live.liveSlots[static_cast<size_t>(i)], frame);
    }

    if (mode == ComparisonTaskMode::Detection && compareTwoModels) {
        live.latestAgreement = computeBoxAgreement(
            live.liveSlots[0].latestDetections, live.liveSlots[1].latestDetections, live.agreementIoUThreshold);
    }
}

namespace {

cv::Mat drawDetectionsOnImage(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat annotated = frame.clone();
    for (const auto& detection : detections) {
        cv::rectangle(annotated, detection.box, cv::Scalar(0, 255, 0), 2);
        const std::string label =
            detection.className + " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        cv::putText(
            annotated, label, detection.box.tl() + cv::Point(0, -4),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    return annotated;
}

const BatchImageResult* findBatchImage(const BatchEvaluationResult& result, const std::string& filename) {
    for (const auto& image : result.images) {
        if (image.imageFilename == filename) {
            return &image;
        }
    }
    return nullptr;
}

void syncBatchEvalSelectedPreview(ComparisonTaskMode mode, const std::string& imageFolderPath, BatchRuntime& batch) {
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
        const cv::Mat toUpload = (mode == ComparisonTaskMode::Detection)
            ? drawDetectionsOnImage(frame, image->detections)
            : frame;
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

void startBatchEvaluationRun(
    ComparisonTaskMode mode, bool compareTwoModels, const std::array<ModelSlotConfig, 2>& slots, BatchRuntime& batch) {
    BatchEvalRunConfig config;
    config.mode = mode;
    config.imageFolderPath = batch.imageFolderPath;
    config.hasGroundTruth = batch.hasGroundTruth;
    if (batch.hasGroundTruth) {
        config.groundTruth = batch.groundTruth;
    }
    config.runSlotB = compareTwoModels;
    config.detectionModelA = slots[0].detectionModel;
    config.detectionModelB = slots[1].detectionModel;
    config.classificationModelA = slots[0].classificationModel;
    config.classificationModelB = slots[1].classificationModel;
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

    batch.worker.start(std::move(config));
    batch.runState = BatchEvalRunState::Running;
}

void updateBatchRuntime(ComparisonTaskMode mode, const std::string& imageFolderPath, BatchRuntime& batch) {
    if (batch.runState == BatchEvalRunState::Running) {
        batch.lastProgress = batch.worker.progress();

        BatchEvalRunResult runResult;
        if (batch.worker.tryTakeResult(runResult)) {
            if (runResult.cancelled) {
                batch.runState = BatchEvalRunState::Cancelled;
            } else {
                batch.resultA = std::move(runResult.slotA);
                batch.resultB = std::move(runResult.slotB);
                if (mode == ComparisonTaskMode::Detection) {
                    batch.detectionMetricsA = computeDetectionMetrics(toDetectionEvaluationItems(batch.resultA));
                    batch.detectionMetricsB = computeDetectionMetrics(toDetectionEvaluationItems(batch.resultB));
                } else {
                    batch.classificationMetricsA =
                        computeClassificationMetrics(toClassificationEvaluationItems(batch.resultA));
                    batch.classificationMetricsB =
                        computeClassificationMetrics(toClassificationEvaluationItems(batch.resultB));
                }
                batch.runState = BatchEvalRunState::Complete;
            }
        }
    }

    syncBatchEvalSelectedPreview(mode, imageFolderPath, batch);
}

bool isBatchEvalImageMismatch(ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename) {
    const BatchImageResult* imageA = findBatchImage(batch.resultA, filename);
    const BatchImageResult* imageB = findBatchImage(batch.resultB, filename);
    if ((imageA == nullptr || !imageA->hasGroundTruth) && (imageB == nullptr || !imageB->hasGroundTruth)) {
        return false;
    }

    if (mode == ComparisonTaskMode::Classification) {
        const auto isMismatch = [](const BatchImageResult* image) {
            if (image == nullptr || !image->hasGroundTruth || image->predictions.empty()) {
                return false;
            }
            return image->predictions.front().className != image->groundTruthLabel;
        };
        return isMismatch(imageA) || isMismatch(imageB);
    }

    const auto hasUnmatchedBox = [](const BatchImageResult* image) {
        if (image == nullptr || !image->hasGroundTruth) {
            return false;
        }
        std::vector<bool> matched(image->groundTruthBoxes.size(), false);
        int falsePositives = 0;
        for (const auto& prediction : image->detections) {
            int bestIdx = -1;
            float bestIoU = 0.5f;
            for (size_t i = 0; i < image->groundTruthBoxes.size(); ++i) {
                if (matched[i] || image->groundTruthBoxes[i].className != prediction.className) {
                    continue;
                }
                const float iou = computeIoU(prediction.box, image->groundTruthBoxes[i].box);
                if (iou >= bestIoU) {
                    bestIoU = iou;
                    bestIdx = static_cast<int>(i);
                }
            }
            if (bestIdx >= 0) {
                matched[static_cast<size_t>(bestIdx)] = true;
            } else {
                falsePositives++;
            }
        }
        const bool hasFalseNegative = std::any_of(matched.begin(), matched.end(), [](bool m) { return !m; });
        return falsePositives > 0 || hasFalseNegative;
    };

    return hasUnmatchedBox(imageA) || hasUnmatchedBox(imageB);
}

std::optional<float> batchEvalImageConfidence(
    ComparisonTaskMode mode, const BatchEvaluationResult& result, const std::string& filename) {
    const BatchImageResult* image = findBatchImage(result, filename);
    if (image == nullptr) {
        return std::nullopt;
    }
    if (mode == ComparisonTaskMode::Classification) {
        if (image->predictions.empty()) {
            return std::nullopt;
        }
        return image->predictions.front().probability;
    }
    if (image->detections.empty()) {
        return std::nullopt;
    }
    float sum = 0.0f;
    for (const auto& detection : image->detections) {
        sum += detection.confidence;
    }
    return sum / static_cast<float>(image->detections.size());
}

bool batchEvalImagePassesConfidenceFilter(
    ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename) {
    if (batch.confidenceFilterMode == BatchEvalConfidenceFilterMode::None) {
        return true;
    }
    const std::optional<float> confA = batchEvalImageConfidence(mode, batch.resultA, filename);
    const std::optional<float> confB = batchEvalImageConfidence(mode, batch.resultB, filename);
    const auto passes = [&](float confidence) {
        return batch.confidenceFilterMode == BatchEvalConfidenceFilterMode::LessThan
            ? confidence < batch.confidenceFilterThreshold
            : confidence > batch.confidenceFilterThreshold;
    };
    return (confA.has_value() && passes(*confA)) || (confB.has_value() && passes(*confB));
}

float batchEvalImageSortConfidence(ComparisonTaskMode mode, const BatchRuntime& batch, const std::string& filename) {
    const std::optional<float> confA = batchEvalImageConfidence(mode, batch.resultA, filename);
    const std::optional<float> confB = batchEvalImageConfidence(mode, batch.resultB, filename);
    if (!confA.has_value() && !confB.has_value()) {
        return 0.0f;
    }
    if (!confA.has_value()) {
        return *confB;
    }
    if (!confB.has_value()) {
        return *confA;
    }
    return std::min(*confA, *confB);
}

void resetModelEvaluationResults(ModelEvaluationState& state) {
    for (auto& liveSlot : state.live.liveSlots) {
        liveSlot.worker.reset();
        liveSlot.workerModel.reset();
        liveSlot.classificationWorker.reset();
        liveSlot.workerClassificationModel.reset();
        liveSlot.latestDetections.clear();
        liveSlot.latestPredictions.clear();
        liveSlot.runtimeError.clear();
    }
    state.live.latestAgreement = BoxAgreement{};

    state.batch.runState = BatchEvalRunState::NotStarted;
    state.batch.resultA = BatchEvaluationResult{};
    state.batch.resultB = BatchEvaluationResult{};
    state.batch.detectionMetricsA = DetectionMetrics{};
    state.batch.detectionMetricsB = DetectionMetrics{};
    state.batch.classificationMetricsA = ClassificationMetrics{};
    state.batch.classificationMetricsB = ClassificationMetrics{};
    state.batch.selectedImageFilename.reset();
}

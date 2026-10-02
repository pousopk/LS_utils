#include "manager/model_slot.hpp"

#include "manager/model_metadata_detection.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

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

bool isModelSlotLoaded(const ModelSlotConfig& slot) {
    return slot.detectionModel || slot.classificationModel || slot.anomalyModel;
}

void loadModelSlot(ModelSlotConfig& slot) {
    slot.loadError.clear();
    slot.summary.clear();
    slot.detectionModel.reset();
    slot.classificationModel.reset();
    slot.anomalyModel.reset();

    if (slot.onnxPath.empty()) {
        return;
    }

    const ModelAutoDetectResult info = autoDetectModel(slot.onnxPath);
    if (!info.error.empty()) {
        slot.loadError = info.error;
        return;
    }

    std::string error;
    bool gpuActive = false;
    std::string summary;
    if (info.mode == DetectedTaskMode::Detection || info.mode == DetectedTaskMode::ObbDetection) {
        const bool isObb = info.mode == DetectedTaskMode::ObbDetection;
        auto model = std::make_shared<YoloModel>(
            slot.onnxPath, info.classNames, info.inputWidth, info.inputHeight, info.hints, isObb, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        gpuActive = model->isGpuActive();
        slot.task = ModelTask::Detection;
        slot.detectionModel = model;
        summary = isObb ? "Detection (OBB)" : "Detection";
    } else if (info.mode == DetectedTaskMode::Classification) {
        auto model = std::make_shared<ClassificationModel>(
            slot.onnxPath, info.classNames, info.inputWidth, info.inputHeight, info.hints, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        gpuActive = model->isGpuActive();
        slot.task = ModelTask::Classification;
        slot.classificationModel = model;
        summary = "Classification";
    } else {
        // Without vad_params calibration, scores are taken as already in [0, 1].
        const float scoreMin = info.anomalyHints.present ? info.anomalyHints.minimum : 0.0f;
        const float scoreMax = info.anomalyHints.present ? info.anomalyHints.maximum : 1.0f;
        auto model = std::make_shared<AnomalyModel>(
            slot.onnxPath, info.inputWidth, info.inputHeight, info.hints, scoreMin, scoreMax, error);
        if (!model->isValid()) {
            slot.loadError = error;
            return;
        }
        gpuActive = model->isGpuActive();
        slot.task = ModelTask::Anomaly;
        slot.anomalyModel = model;
        if (info.anomalyHints.present) {
            slot.anomalyThreshold = info.anomalyHints.threshold;
        }
        summary = info.anomalyHints.present ? "Anomaly" : "Anomaly (no score calibration)";
    }

    summary += " | " + std::to_string(info.inputWidth) + "x" + std::to_string(info.inputHeight);
    if (!info.classNames.empty()) {
        summary += " | " + std::to_string(info.classNames.size()) + " classes";
    }
    summary += gpuActive ? " | GPU" : " | CPU";
    slot.summary = summary;
}

#include "manager/yolo_inference.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>

LetterboxTransform computeLetterboxTransform(int origWidth, int origHeight, int modelInputSize) {
    const float scale = std::min(
        static_cast<float>(modelInputSize) / static_cast<float>(origWidth),
        static_cast<float>(modelInputSize) / static_cast<float>(origHeight));
    const float scaledWidth = origWidth * scale;
    const float scaledHeight = origHeight * scale;

    LetterboxTransform transform;
    transform.scale = scale;
    transform.padX = (modelInputSize - scaledWidth) / 2.0f;
    transform.padY = (modelInputSize - scaledHeight) / 2.0f;
    return transform;
}

cv::Mat letterboxResize(const cv::Mat& frame, int modelInputSize, const LetterboxTransform& transform) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), transform.scale, transform.scale);

    cv::Mat canvas(modelInputSize, modelInputSize, frame.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(
        static_cast<int>(transform.padX),
        static_cast<int>(transform.padY),
        resized.cols,
        resized.rows)));
    return canvas;
}

float computeIoU(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    const float intersectionArea = static_cast<float>(intersection.area());
    if (intersectionArea <= 0.0f) {
        return 0.0f;
    }
    const float unionArea = static_cast<float>(a.area() + b.area()) - intersectionArea;
    return unionArea > 0.0f ? intersectionArea / unionArea : 0.0f;
}

std::vector<Detection> decodeYoloOutput(
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    const LetterboxTransform& transform,
    int origWidth,
    int origHeight,
    float confThreshold,
    float nmsThreshold) {
    const int numClasses = output.cols - 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int row = 0; row < output.rows; ++row) {
        const float* data = output.ptr<float>(row);
        const float cx = data[0];
        const float cy = data[1];
        const float w = data[2];
        const float h = data[3];

        int bestClass = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            const float score = data[4 + c];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestClass < 0 || bestScore < confThreshold) {
            continue;
        }

        const float x0 = (cx - w / 2.0f - transform.padX) / transform.scale;
        const float y0 = (cy - h / 2.0f - transform.padY) / transform.scale;
        const float boxWidth = w / transform.scale;
        const float boxHeight = h / transform.scale;

        cv::Rect box(
            static_cast<int>(std::round(x0)),
            static_cast<int>(std::round(y0)),
            static_cast<int>(std::round(boxWidth)),
            static_cast<int>(std::round(boxHeight)));
        box &= cv::Rect(0, 0, origWidth, origHeight);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }

        boxes.push_back(box);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> keptIndices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, keptIndices);

    std::vector<Detection> detections;
    detections.reserve(keptIndices.size());
    for (int index : keptIndices) {
        Detection detection;
        detection.box = boxes[index];
        detection.classId = classIds[index];
        detection.confidence = scores[index];
        detection.className = (classIds[index] >= 0 && classIds[index] < static_cast<int>(classNames.size()))
            ? classNames[classIds[index]]
            : ("class_" + std::to_string(classIds[index]));
        detections.push_back(std::move(detection));
    }
    return detections;
}

BoxAgreement computeBoxAgreement(
    const std::vector<Detection>& a,
    const std::vector<Detection>& b,
    float iouThreshold) {
    BoxAgreement agreement;
    agreement.totalA = static_cast<int>(a.size());
    agreement.totalB = static_cast<int>(b.size());

    std::vector<bool> usedB(b.size(), false);
    for (const auto& detectionA : a) {
        int bestIndex = -1;
        float bestIoU = iouThreshold;
        for (size_t i = 0; i < b.size(); ++i) {
            if (usedB[i] || b[i].classId != detectionA.classId) {
                continue;
            }
            const float iou = computeIoU(detectionA.box, b[i].box);
            if (iou >= bestIoU) {
                bestIoU = iou;
                bestIndex = static_cast<int>(i);
            }
        }
        if (bestIndex >= 0) {
            usedB[bestIndex] = true;
            agreement.matchedPairs++;
        }
    }
    return agreement;
}

namespace {
std::vector<std::string> loadClassNames(const std::string& path) {
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

YoloModel::YoloModel(const std::string& onnxPath, const std::string& classNamesPath, std::string& errorOut) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
    } catch (const cv::Exception& e) {
        errorOut = std::string("Failed to load ONNX model: ") + e.what();
        return;
    }
    if (net_.empty()) {
        errorOut = "Failed to load ONNX model: empty network";
        return;
    }
    classNames_ = loadClassNames(classNamesPath);
    valid_ = true;
}

std::vector<Detection> YoloModel::infer(const cv::Mat& frame, float confThreshold, float nmsThreshold) {
    if (!valid_ || frame.empty()) {
        return {};
    }

    const LetterboxTransform transform = computeLetterboxTransform(frame.cols, frame.rows, kModelInputSize);
    const cv::Mat letterboxed = letterboxResize(frame, kModelInputSize, transform);

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed, 1.0 / 255.0, cv::Size(kModelInputSize, kModelInputSize), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat rawOutput = net_.forward();

    // Ultralytics v8/v11 export shape: [1, 4+numClasses, numBoxes]. Reshape
    // to [4+numClasses, numBoxes] and transpose to [numBoxes, 4+numClasses]
    // so each row is one candidate box, matching decodeYoloOutput's input.
    cv::Mat output(rawOutput.size[1], rawOutput.size[2], CV_32F, rawOutput.ptr<float>());
    cv::Mat transposed;
    cv::transpose(output, transposed);

    return decodeYoloOutput(transposed, classNames_, transform, frame.cols, frame.rows, confThreshold, nmsThreshold);
}

#include "manager/yolo_inference.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

LetterboxTransform computeLetterboxTransform(
    int origWidth, int origHeight, int targetWidth, int targetHeight, bool centerPadding) {
    const float scale = std::min(
        static_cast<float>(targetWidth) / static_cast<float>(origWidth),
        static_cast<float>(targetHeight) / static_cast<float>(origHeight));
    const float scaledWidth = origWidth * scale;
    const float scaledHeight = origHeight * scale;

    LetterboxTransform transform;
    transform.scale = scale;
    if (centerPadding) {
        transform.padX = (targetWidth - scaledWidth) / 2.0f;
        transform.padY = (targetHeight - scaledHeight) / 2.0f;
    } else {
        transform.padX = 0.0f;
        transform.padY = 0.0f;
    }
    return transform;
}

cv::Mat letterboxResize(
    const cv::Mat& frame, int targetWidth, int targetHeight, const LetterboxTransform& transform, float fillValue) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), transform.scale, transform.scale);

    cv::Mat canvas(targetHeight, targetWidth, frame.type(), cv::Scalar(fillValue, fillValue, fillValue));
    resized.copyTo(canvas(cv::Rect(
        static_cast<int>(transform.padX),
        static_cast<int>(transform.padY),
        resized.cols,
        resized.rows)));
    return canvas;
}

cv::Mat unletterboxMask(
    const cv::Mat& mask, const LetterboxTransform& transform, int origWidth, int origHeight) {
    const int scaledWidth = static_cast<int>(std::lround(origWidth * transform.scale));
    const int scaledHeight = static_cast<int>(std::lround(origHeight * transform.scale));
    const int padX = static_cast<int>(std::lround(transform.padX));
    const int padY = static_cast<int>(std::lround(transform.padY));

    const cv::Rect cropRect(padX, padY, scaledWidth, scaledHeight);
    const cv::Rect boundedCrop = cropRect & cv::Rect(0, 0, mask.cols, mask.rows);
    const cv::Mat cropped = mask(boundedCrop);

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(origWidth, origHeight), 0, 0, cv::INTER_LINEAR);
    return resized;
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

YoloModel::YoloModel(
    const std::string& onnxPath,
    const std::vector<std::string>& classNames,
    int inputWidth,
    int inputHeight,
    const OnnxPreprocessingHints& hints,
    std::string& errorOut)
    : classNames_(classNames), inputWidth_(inputWidth), inputHeight_(inputHeight), hints_(hints) {
    OrtSessionResult result = createOrtSession(onnxPath);
    if (!result.session) {
        errorOut = result.error;
        return;
    }
    session_ = std::move(result.session);
    gpuActive_ = result.gpuActive;

    Ort::AllocatorWithDefaultOptions allocator;
    inputName_ = session_->GetInputNameAllocated(0, allocator).get();
    outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

    valid_ = true;
}

std::vector<Detection> YoloModel::infer(const cv::Mat& frame, float confThreshold, float nmsThreshold) {
    if (!valid_ || frame.empty()) {
        return {};
    }

    const LetterboxTransform transform = computeLetterboxTransform(
        frame.cols, frame.rows, inputWidth_, inputHeight_, hints_.padCenter);
    const cv::Mat letterboxed = letterboxResize(frame, inputWidth_, inputHeight_, transform, hints_.padFill);

    const std::vector<float> inputData = hwcBgrToNchwFloat(letterboxed, hints_.inputScale);
    const std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputData.data()), inputData.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    cv::Mat rawOutput = ortValueToMat(outputs.front());

    // Ultralytics v8/v11 export shape: [1, 4+numClasses, numBoxes]. Reject
    // anything else up front -- a classification/regression head or an
    // end2end/NMS-baked export won't have this shape.
    if (rawOutput.dims != 3 || rawOutput.size[0] != 1) {
        CV_Error(cv::Error::StsError,
            "Unexpected model output shape (expected [1, 4+numClasses, numBoxes]) -- "
            "this model may not be a standard YOLO detector (e.g. a classification "
            "or regression head, or an NMS-baked/end2end export)");
    }

    // Reshape to [4+numClasses, numBoxes] and transpose to
    // [numBoxes, 4+numClasses] so each row is one candidate box, matching
    // decodeYoloOutput's input.
    cv::Mat output(rawOutput.size[1], rawOutput.size[2], CV_32F, rawOutput.ptr<float>());
    cv::Mat transposed;
    cv::transpose(output, transposed);

    return decodeYoloOutput(transposed, classNames_, transform, frame.cols, frame.rows, confThreshold, nmsThreshold);
}

#include "manager/yolo_inference.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

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

#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    int classId = -1;
    std::string className;
    float confidence = 0.0f;
};

// Maps between an `origWidth` x `origHeight` frame and the square
// `modelInputSize` x `modelInputSize` "letterboxed" image YOLO expects:
// the frame is scaled by `scale` to fit inside the square, then centered
// with `padX`/`padY` pixels of border on each side.
struct LetterboxTransform {
    float scale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

LetterboxTransform computeLetterboxTransform(int origWidth, int origHeight, int modelInputSize);

// Resizes+pads `frame` into a `modelInputSize` x `modelInputSize` square
// per `transform`, ready to feed to cv::dnn::blobFromImage.
cv::Mat letterboxResize(const cv::Mat& frame, int modelInputSize, const LetterboxTransform& transform);

float computeIoU(const cv::Rect& a, const cv::Rect& b);

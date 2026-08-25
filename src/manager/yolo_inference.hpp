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

// Decodes a raw Ultralytics YOLOv8/v11-style output tensor already reshaped
// to a `numBoxes` x (4 + numClasses) CV_32F matrix (each row is
// [cx, cy, w, h, classScore_0, ..., classScore_{numClasses-1}], in
// model-input-pixel space) into detections in original-frame pixel
// coordinates, keeping only the best class per row above `confThreshold`
// and applying NMS at `nmsThreshold`.
std::vector<Detection> decodeYoloOutput(
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    const LetterboxTransform& transform,
    int origWidth,
    int origHeight,
    float confThreshold,
    float nmsThreshold);

struct BoxAgreement {
    int matchedPairs = 0;
    int totalA = 0;
    int totalB = 0;
};

// Greedily matches same-class detections between `a` and `b` whose IoU is
// at or above `iouThreshold`, one-to-one (each detection matches at most
// once). Used to show how much two models agree on the same frame.
BoxAgreement computeBoxAgreement(
    const std::vector<Detection>& a,
    const std::vector<Detection>& b,
    float iouThreshold);

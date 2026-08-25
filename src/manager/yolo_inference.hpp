#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

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

LetterboxTransform computeLetterboxTransform(int origWidth, int origHeight, int targetWidth, int targetHeight);

// Resizes+pads `frame` into a `targetWidth` x `targetHeight` canvas per
// `transform`, ready to feed to cv::dnn::blobFromImage. `targetWidth`
// and `targetHeight` need not be equal -- non-square model inputs are
// supported.
cv::Mat letterboxResize(const cv::Mat& frame, int targetWidth, int targetHeight, const LetterboxTransform& transform);

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

class YoloModel {
public:
    // Loads the ONNX model. `classNames` is used as-is (empty means
    // label classes by index) -- resolving where names come from (a
    // browsed file, auto-detected metadata, ...) is the caller's job,
    // not this class's. `inputWidth`/`inputHeight` must match what the
    // model was exported with. On failure, isValid() is false and
    // errorOut holds a human-readable reason -- construction never
    // throws.
    YoloModel(
        const std::string& onnxPath,
        const std::vector<std::string>& classNames,
        int inputWidth,
        int inputHeight,
        std::string& errorOut);

    bool isValid() const { return valid_; }

    // Runs letterbox -> forward pass -> decode on `frame` using this
    // model's loaded network. Returns an empty vector if the model isn't
    // valid or `frame` is empty.
    std::vector<Detection> infer(const cv::Mat& frame, float confThreshold, float nmsThreshold);

private:
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    int inputWidth_ = 640;
    int inputHeight_ = 640;
    bool valid_ = false;
};

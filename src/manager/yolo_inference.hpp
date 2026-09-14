#pragma once

#include "manager/onnx_metadata.hpp"
#include "manager/onnx_runtime_env.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    int classId = -1;
    std::string className;
    float confidence = 0.0f;
    // Degrees, clockwise, around box's top-left corner (box.x, box.y)
    // pre-rotation -- same convention as DraftDetectionBox/GroundTruthBox.
    // 0.0 for an ordinary axis-aligned detection.
    float rotationDegrees = 0.0f;
};

struct ObbBox {
    cv::Rect box;              // the box's own unrotated x/y/width/height
    float rotationDegrees = 0.0f;
};

// Pure function: converts a center-pivot oriented box -- as YOLO's xywhr
// output describes it (cx,cy = center, w,h = box dimensions along its own
// axes, rotationDegrees = clockwise rotation of those axes relative to
// the image's x/y axes) -- into this app's top-left-pivot convention
// (box = the same rectangle's un-rotated x/y/width/height, rotationDegrees
// = clockwise rotation around box.x,box.y). Same rigid rectangle, just
// re-anchored to a different (but equivalent) corner: the pre-rotation
// top-left corner, relative to the center, is (-w/2,-h/2); rotating that
// offset by rotationDegrees and adding it to (cx,cy) gives the point that
// stays fixed on screen when the box is described as rotating around it
// instead of around its center.
ObbBox centerObbToTopLeftPivotBox(float cx, float cy, float w, float h, float rotationDegrees);

// Maps between an `origWidth` x `origHeight` frame and the square
// `modelInputSize` x `modelInputSize` "letterboxed" image YOLO expects:
// the frame is scaled by `scale` to fit inside the square, then centered
// with `padX`/`padY` pixels of border on each side.
struct LetterboxTransform {
    float scale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

// `centerPadding` true (default) centers the scaled image in the target
// canvas; false anchors it top-left (per a model's `padding` metadata
// declaring position "top_left") -- padX/padY are then both 0.
LetterboxTransform computeLetterboxTransform(
    int origWidth, int origHeight, int targetWidth, int targetHeight, bool centerPadding = true);

// Resizes+pads `frame` into a `targetWidth` x `targetHeight` canvas per
// `transform`, ready to feed to hwcBgrToNchwFloat(). `targetWidth`
// and `targetHeight` need not be equal -- non-square model inputs are
// supported. `fillValue` (0-255, replicated across channels) fills the
// padding border.
cv::Mat letterboxResize(
    const cv::Mat& frame, int targetWidth, int targetHeight, const LetterboxTransform& transform,
    float fillValue = 114.0f);

// Inverse of letterboxResize for a full raster (not point coordinates):
// crops `mask` down to the `transform`-described unpadded region, then
// resizes that crop to `origWidth` x `origHeight`. Companion to
// computeLetterboxTransform/letterboxResize, used to map a model's
// full-resolution output (e.g. an anomaly heatmap) back to original-
// frame coordinates.
cv::Mat unletterboxMask(
    const cv::Mat& mask, const LetterboxTransform& transform, int origWidth, int origHeight);

float computeIoU(const cv::Rect& a, const cv::Rect& b);

// Rotation-aware IoU: if both rotationDegreesA and rotationDegreesB are
// exactly 0.0, delegates to computeIoU (byte-identical fast path -- the
// overwhelmingly common case, and every existing caller/test of
// computeIoU is unaffected). Otherwise, computes each box's 4 corners
// via rotatedBoxCorners() and takes their exact polygon intersection via
// cv::intersectConvexConvex -- both boxes' corners come from the same
// perimeter-walk construction regardless of rotation, so their winding
// order is always mutually consistent, which is what
// intersectConvexConvex requires.
float computeRotatedIoU(
    const cv::Rect& boxA, float rotationDegreesA, const cv::Rect& boxB, float rotationDegreesB);

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

// Ultralytics OBB layout: [cx, cy, w, h, class_0..class_{nc-1}, angle_radians]
// -- one more column than decodeYoloOutput's [cx,cy,w,h,classes], the
// extra being the angle appended last. NMS runs on cv::RotatedRect via
// the cv::dnn::NMSBoxes overload for vector<RotatedRect> (rotation-aware,
// unlike decodeYoloOutput's axis-aligned cv::Rect overload) using each
// candidate's raw center/size/angle -- cheap and natural in that space,
// before any pivot conversion.
// Only the survivors are converted to the app's top-left-pivot Detection
// convention via centerObbToTopLeftPivotBox. A detection is rejected if
// its RotatedRect's axis-aligned bounding box doesn't intersect the frame
// at all -- the stored box itself is never clamped, since clamping a
// pre-rotation x/y/width/height directly would move the pivot and
// silently corrupt the rotation.
std::vector<Detection> decodeYoloObbOutput(
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
    // `hints` controls preprocessing (pixel scale, letterbox fill/anchor)
    // -- pass a default-constructed OnnxPreprocessingHints for the
    // standard Ultralytics convention, or the result of
    // extractPreprocessingHints() for a model that declares otherwise.
    YoloModel(
        const std::string& onnxPath,
        const std::vector<std::string>& classNames,
        int inputWidth,
        int inputHeight,
        const OnnxPreprocessingHints& hints,
        bool isObb,
        std::string& errorOut);

    bool isValid() const { return valid_; }

    // True if this model's session is using the CUDA execution provider
    // (only meaningful once isValid() is true).
    bool isGpuActive() const { return gpuActive_; }

    // Runs letterbox -> forward pass -> decode on `frame` using this
    // model's loaded network. Returns an empty vector if the model isn't
    // valid or `frame` is empty.
    std::vector<Detection> infer(const cv::Mat& frame, float confThreshold, float nmsThreshold);

private:
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    std::vector<std::string> classNames_;
    int inputWidth_ = 640;
    int inputHeight_ = 640;
    OnnxPreprocessingHints hints_;
    bool isObb_ = false;
    bool gpuActive_ = false;
    bool valid_ = false;
};

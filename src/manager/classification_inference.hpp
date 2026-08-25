#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

struct ClassPrediction {
    int classId = -1;
    std::string className;
    float probability = 0.0f;
};

// Sorts a raw classifier output (already-normalized probabilities, one per
// class) into ClassPrediction entries, descending by probability. Throws
// (via CV_Error) if `output` isn't a 2D [1, numClasses] tensor, or if its
// values don't sum to ~1.0 (within `sumTolerance`) -- catching both
// wrong-shape models and raw-logit exports (which need softmax the caller
// hasn't applied) up front instead of silently misinterpreting them.
std::vector<ClassPrediction> decodeClassificationOutput(
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    float sumTolerance = 0.05f);

class ClassificationModel {
public:
    // Loads the ONNX classifier. `classNames` is used as-is (empty means
    // label classes by index) -- resolving where names come from (a
    // browsed file, auto-detected metadata, ...) is the caller's job, not
    // this class's. `inputWidth`/`inputHeight` must match what the model
    // was exported with -- unlike detection, classifiers don't share one
    // fixed size (real examples found at 448x576 and 576x960), so
    // there's no safe default to fall back to. On failure, isValid() is
    // false and errorOut holds a human-readable reason -- construction
    // never throws.
    ClassificationModel(
        const std::string& onnxPath,
        const std::vector<std::string>& classNames,
        int inputWidth,
        int inputHeight,
        std::string& errorOut);

    bool isValid() const { return valid_; }

    // Runs resize -> forward pass -> decode on `frame`, returning all
    // classes sorted by probability descending. Returns an empty vector
    // if the model isn't valid or `frame` is empty.
    std::vector<ClassPrediction> infer(const cv::Mat& frame);

private:
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    int inputWidth_ = 224;
    int inputHeight_ = 224;
    bool valid_ = false;
};

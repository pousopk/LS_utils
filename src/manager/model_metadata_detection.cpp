#include "manager/model_metadata_detection.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

ModelAutoDetectResult autoDetectModel(const std::string& onnxPath) {
    ModelAutoDetectResult result;

    onnx::ModelProto model;
    std::string loadError;
    if (!loadOnnxModelProto(onnxPath, model, loadError)) {
        result.error = loadError;
        return result;
    }

    OnnxInputShape shape;
    std::string shapeError;
    if (!extractInputShape(model, shape, shapeError)) {
        result.error = shapeError;
        return result;
    }

    result.shapeDetected = true;
    result.inputWidth = shape.width;
    result.inputHeight = shape.height;
    result.classNames = extractClassNames(model);

    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxPath);
        if (net.empty()) {
            return result;
        }
        cv::Mat blankFrame = cv::Mat::zeros(shape.height, shape.width, CV_8UC3);
        cv::Mat blob = cv::dnn::blobFromImage(
            blankFrame, 1.0 / 255.0, cv::Size(shape.width, shape.height), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        if (output.dims == 3 && output.size[0] == 1) {
            result.suggestedMode = DetectedTaskMode::Detection;
        } else if (output.dims == 2 && output.size[0] == 1) {
            result.suggestedMode = DetectedTaskMode::Classification;
        }
    } catch (const cv::Exception&) {
        // Mode guess failed -- leave suggestedMode at Unknown. Shape and
        // class names (already set above) are still useful on their own.
    }

    return result;
}

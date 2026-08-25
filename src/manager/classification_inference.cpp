#include "manager/classification_inference.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>

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

std::vector<ClassPrediction> decodeClassificationOutput(
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    float sumTolerance) {
    if (output.dims != 2 || output.size[0] != 1) {
        CV_Error(cv::Error::StsError,
            "Unexpected classifier output shape (expected [1, numClasses])");
    }

    const int numClasses = output.size[1];
    const float* data = output.ptr<float>(0);

    float sum = 0.0f;
    for (int i = 0; i < numClasses; ++i) {
        sum += data[i];
    }
    if (std::fabs(sum - 1.0f) > sumTolerance) {
        CV_Error(cv::Error::StsError,
            "Classifier output does not look like normalized probabilities "
            "(values summed to " + std::to_string(sum) + ", expected ~1.0) -- "
            "this model may output raw logits instead");
    }

    std::vector<ClassPrediction> predictions;
    predictions.reserve(numClasses);
    for (int i = 0; i < numClasses; ++i) {
        ClassPrediction prediction;
        prediction.classId = i;
        prediction.probability = data[i];
        prediction.className = (i < static_cast<int>(classNames.size()))
            ? classNames[i]
            : ("class_" + std::to_string(i));
        predictions.push_back(std::move(prediction));
    }

    std::sort(predictions.begin(), predictions.end(), [](const ClassPrediction& a, const ClassPrediction& b) {
        return a.probability > b.probability;
    });
    return predictions;
}

ClassificationModel::ClassificationModel(
    const std::string& onnxPath,
    const std::string& classNamesPath,
    int inputWidth,
    int inputHeight,
    std::string& errorOut)
    : inputWidth_(inputWidth), inputHeight_(inputHeight) {
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

std::vector<ClassPrediction> ClassificationModel::infer(const cv::Mat& frame) {
    if (!valid_ || frame.empty()) {
        return {};
    }

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputWidth_, inputHeight_));

    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(inputWidth_, inputHeight_), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    return decodeClassificationOutput(output, classNames_);
}

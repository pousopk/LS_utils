#include "manager/classification_inference.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

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

std::vector<ClassPrediction> ClassificationModel::infer(const cv::Mat& frame) {
    if (!valid_ || frame.empty()) {
        return {};
    }

    cv::Mat prepared;
    if (hints_.maintainAspectRatio) {
        const LetterboxTransform transform = computeLetterboxTransform(
            frame.cols, frame.rows, inputWidth_, inputHeight_, hints_.padCenter);
        prepared = letterboxResize(frame, inputWidth_, inputHeight_, transform, hints_.padFill);
    } else {
        cv::resize(frame, prepared, cv::Size(inputWidth_, inputHeight_));
    }

    const std::vector<float> inputData = hwcBgrToNchwFloat(prepared, hints_.inputScale);
    const std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputData.data()), inputData.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    cv::Mat output = ortValueToMat(outputs.front());

    return decodeClassificationOutput(output, classNames_);
}

#include "manager/anomaly_inference.hpp"

#include <algorithm>

AnomalyScore scoreAnomalyHeatmap(const cv::Mat& heatmap, float scoreMin, float scoreMax, float threshold) {
    AnomalyScore result;
    if (heatmap.empty()) {
        return result;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(heatmap, &minVal, &maxVal);
    result.rawScore = static_cast<float>(maxVal);

    const float range = scoreMax - scoreMin;
    result.score = (range > 0.0f) ? std::clamp((result.rawScore - scoreMin) / range, 0.0f, 1.0f)
                                   : std::clamp(result.rawScore, 0.0f, 1.0f);
    result.isAnomalous = result.score >= threshold;
    return result;
}

AnomalyModel::AnomalyModel(
    const std::string& onnxPath,
    int inputWidth,
    int inputHeight,
    const OnnxPreprocessingHints& hints,
    float scoreMin,
    float scoreMax,
    std::string& errorOut)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), hints_(hints), scoreMin_(scoreMin), scoreMax_(scoreMax) {
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

AnomalyResult AnomalyModel::infer(const cv::Mat& frame, float threshold) {
    if (!valid_ || frame.empty()) {
        return {};
    }

    const LetterboxTransform transform =
        computeLetterboxTransform(frame.cols, frame.rows, inputWidth_, inputHeight_, hints_.padCenter);
    const cv::Mat letterboxed = letterboxResize(frame, inputWidth_, inputHeight_, transform, hints_.padFill);

    std::vector<float> inputData = hwcBgrToNchwFloat(letterboxed, hints_.inputScale);
    applyChannelNormalization(inputData, inputHeight_, inputWidth_, hints_.channelMean, hints_.channelStd);
    const std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputData.data()), inputData.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    cv::Mat rawOutput = ortValueToMat(outputs.front());

    // Expected shape: [1, 1, inputHeight, inputWidth] -- a full-resolution
    // single-channel heatmap matching the model's own input size.
    if (rawOutput.dims != 4 || rawOutput.size[0] != 1 || rawOutput.size[1] != 1) {
        CV_Error(
            cv::Error::StsError,
            "Unexpected anomaly model output shape (expected [1, 1, height, width]) -- "
            "this model may not be a pixel-level anomaly detector");
    }

    const cv::Mat rawHeatmap(rawOutput.size[2], rawOutput.size[3], CV_32F, rawOutput.ptr<float>());

    const AnomalyScore scoreResult = scoreAnomalyHeatmap(rawHeatmap, scoreMin_, scoreMax_, threshold);

    AnomalyResult result;
    result.rawScore = scoreResult.rawScore;
    result.score = scoreResult.score;
    result.isAnomalous = scoreResult.isAnomalous;
    result.heatmap = unletterboxMask(rawHeatmap, transform, frame.cols, frame.rows);
    return result;
}

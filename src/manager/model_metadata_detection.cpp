#include "manager/model_metadata_detection.hpp"
#include "manager/onnx_runtime_env.hpp"

#include <opencv2/core.hpp>

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
    result.hints = extractPreprocessingHints(model);

    OrtSessionResult session = createOrtSession(onnxPath);
    if (!session.session) {
        // Mode guess unavailable -- shape and class names (already set
        // above) are still useful on their own.
        return result;
    }

    try {
        cv::Mat blankFrame = cv::Mat::zeros(shape.height, shape.width, CV_8UC3);
        const std::vector<float> inputData = hwcBgrToNchwFloat(blankFrame, result.hints.inputScale);
        const std::vector<int64_t> inputShape = {1, 3, shape.height, shape.width};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputData.data()), inputData.size(), inputShape.data(), inputShape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        const std::string inputName = session.session->GetInputNameAllocated(0, allocator).get();
        const std::string outputName = session.session->GetOutputNameAllocated(0, allocator).get();
        const char* inputNames[] = {inputName.c_str()};
        const char* outputNames[] = {outputName.c_str()};

        std::vector<Ort::Value> outputs =
            session.session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        const cv::Mat output = ortValueToMat(outputs.front());

        if (output.dims == 3 && output.size[0] == 1) {
            result.suggestedMode = DetectedTaskMode::Detection;
        } else if (output.dims == 2 && output.size[0] == 1) {
            result.suggestedMode = DetectedTaskMode::Classification;
        }
    } catch (const Ort::Exception&) {
        // Mode guess failed -- leave suggestedMode at Unknown. Shape and
        // class names (already set above) are still useful on their own.
    }

    return result;
}

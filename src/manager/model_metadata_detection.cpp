#include "manager/model_metadata_detection.hpp"
#include "manager/onnx_runtime_env.hpp"

#include <opencv2/core.hpp>

ResolvedModelTask resolveModelTask(
    const std::string& taskHint, const std::vector<int64_t>& outputShape, int inputWidth, int inputHeight,
    size_t classCount) {
    ResolvedModelTask result;

    if (taskHint == "detect") {
        result.mode = DetectedTaskMode::Detection;
        return result;
    }
    if (taskHint == "obb") {
        result.mode = DetectedTaskMode::ObbDetection;
        return result;
    }
    if (taskHint == "classify") {
        result.mode = DetectedTaskMode::Classification;
        return result;
    }
    if (taskHint == "vad") {
        result.mode = DetectedTaskMode::Anomaly;
        return result;
    }
    if (taskHint == "segment" || taskHint == "pose") {
        result.error = "Unsupported model task '" + taskHint + "' (only detect, obb, classify and anomaly)";
        return result;
    }

    // No (recognized) task metadata -- guess from the output shape.
    if (outputShape.empty()) {
        result.error = "Could not determine the model's task: no 'task' metadata and the test inference failed";
        return result;
    }
    if (outputShape.size() == 3 && outputShape[0] == 1) {
        const int64_t channels = outputShape[1];
        const auto boxColumns = static_cast<int64_t>(classCount) + 4;
        if (channels == boxColumns) {
            result.mode = DetectedTaskMode::Detection;
        } else if (channels == boxColumns + 1) {
            result.mode = DetectedTaskMode::ObbDetection;
        } else {
            result.error = "Detection output has " + std::to_string(channels) + " channels, expected "
                + std::to_string(boxColumns) + " (boxes) or " + std::to_string(boxColumns + 1) + " (OBB) for "
                + std::to_string(classCount) + " classes";
        }
        return result;
    }
    if (outputShape.size() == 2 && outputShape[0] == 1) {
        result.mode = DetectedTaskMode::Classification;
        return result;
    }
    if (outputShape.size() == 4 && outputShape[0] == 1 && outputShape[1] == 1 && outputShape[2] == inputHeight
        && outputShape[3] == inputWidth) {
        result.mode = DetectedTaskMode::Anomaly;
        return result;
    }
    result.error = "Could not determine the model's task: no 'task' metadata and unrecognized output shape";
    return result;
}

namespace {

// Runs one blank-frame forward pass and returns the first output's shape,
// or an empty vector if the session can't be created or the run fails.
std::vector<int64_t> probeOutputShape(const std::string& onnxPath, int width, int height, float inputScale) {
    OrtSessionResult session = createOrtSession(onnxPath);
    if (!session.session) {
        return {};
    }
    try {
        cv::Mat blankFrame = cv::Mat::zeros(height, width, CV_8UC3);
        const std::vector<float> inputData = hwcBgrToNchwFloat(blankFrame, inputScale);
        const std::vector<int64_t> inputShape = {1, 3, height, width};

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
        return outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    } catch (const Ort::Exception&) {
        return {};
    }
}

bool isKnownTaskHint(const std::string& taskHint) {
    return taskHint == "detect" || taskHint == "obb" || taskHint == "classify" || taskHint == "vad"
        || taskHint == "segment" || taskHint == "pose";
}

} // namespace

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
        result.error = shapeError + " -- re-export the model with a fixed input size";
        return result;
    }

    result.inputWidth = shape.width;
    result.inputHeight = shape.height;
    result.classNames = extractClassNames(model);
    result.hints = extractPreprocessingHints(model);
    result.anomalyHints = extractAnomalyScoreHints(model);

    const std::string taskHint = extractTaskHint(model);
    const std::vector<int64_t> outputShape = isKnownTaskHint(taskHint)
        ? std::vector<int64_t>{}
        : probeOutputShape(onnxPath, shape.width, shape.height, result.hints.inputScale);

    const ResolvedModelTask task =
        resolveModelTask(taskHint, outputShape, shape.width, shape.height, result.classNames.size());
    if (task.mode == DetectedTaskMode::Unknown) {
        result.error = task.error;
        return result;
    }
    if (task.mode != DetectedTaskMode::Anomaly && result.classNames.empty()) {
        result.error = "Model has no class names in its metadata ('names' or 'categories')";
        return result;
    }

    result.mode = task.mode;
    return result;
}

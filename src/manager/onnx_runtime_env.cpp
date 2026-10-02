#include "manager/onnx_runtime_env.hpp"

Ort::Env& sharedOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vision_app");
    return env;
}

OrtSessionResult createOrtSession(const std::string& onnxPath) {
    OrtSessionResult result;

    Ort::SessionOptions sessionOptions;
    try {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        result.gpuActive = true;
    } catch (const Ort::Exception&) {
        // No compatible GPU/CUDA/cuDNN -- fall back to CPU, not an error.
        result.gpuActive = false;
    }

    if (result.gpuActive) {
        try {
            result.session = std::make_unique<Ort::Session>(sharedOrtEnv(), onnxPath.c_str(), sessionOptions);
            return result;
        } catch (const Ort::Exception&) {
            // Appending the CUDA provider doesn't touch the GPU; device init
            // happens at session creation, so a broken driver/CUDA state
            // surfaces here. Retry on CPU -- a bad model will fail there too.
            result.gpuActive = false;
            sessionOptions = Ort::SessionOptions();
        }
    }

    try {
        result.session = std::make_unique<Ort::Session>(sharedOrtEnv(), onnxPath.c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        result.error = std::string("Failed to load ONNX model: ") + e.what();
        result.session.reset();
    }

    return result;
}

std::vector<float> hwcBgrToNchwFloat(const cv::Mat& hwcBgr, float scale) {
    const int height = hwcBgr.rows;
    const int width = hwcBgr.cols;
    const size_t planeSize = static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<float> chw(planeSize * 3);

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = hwcBgr.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const uint8_t b = row[x * 3 + 0];
            const uint8_t g = row[x * 3 + 1];
            const uint8_t r = row[x * 3 + 2];
            const size_t pixelIndex = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            chw[0 * planeSize + pixelIndex] = static_cast<float>(r) * scale;
            chw[1 * planeSize + pixelIndex] = static_cast<float>(g) * scale;
            chw[2 * planeSize + pixelIndex] = static_cast<float>(b) * scale;
        }
    }
    return chw;
}

cv::Mat ortValueToMat(Ort::Value& value) {
    const Ort::TensorTypeAndShapeInfo shapeInfo = value.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = shapeInfo.GetShape();
    std::vector<int> dims(shape.begin(), shape.end());
    cv::Mat view(static_cast<int>(dims.size()), dims.data(), CV_32F, value.GetTensorMutableData<float>());
    return view.clone();
}

void applyChannelNormalization(
    std::vector<float>& nchw, int height, int width,
    const std::array<float, 3>& mean, const std::array<float, 3>& stddev) {
    const size_t planeSize = static_cast<size_t>(height) * static_cast<size_t>(width);
    for (size_t channel = 0; channel < 3; ++channel) {
        float* plane = nchw.data() + channel * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            plane[i] = (plane[i] - mean[channel]) / stddev[channel];
        }
    }
}

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

// A single process-wide ONNX Runtime environment -- ONNX Runtime
// recommends exactly one Env per process, shared across all sessions.
Ort::Env& sharedOrtEnv();

struct OrtSessionResult {
    std::unique_ptr<Ort::Session> session;
    bool gpuActive = false;
    std::string error;
};

// Creates an ONNX Runtime session for `onnxPath`, attempting the CUDA
// execution provider first. GPU failure (no compatible GPU/CUDA/cuDNN, or
// a CUDA device that fails to initialise) is never an error here -- it's
// caught, `gpuActive` is left false, and the session is created on CPU. Only a genuine session-creation
// failure (missing/corrupt file, incompatible graph) sets `error` and
// leaves `session` null.
OrtSessionResult createOrtSession(const std::string& onnxPath);

// Converts an already-resized HWC BGR uint8 cv::Mat into a channel-planar
// (NCHW) float buffer in RGB order (matching the swapRB=true convention
// this codebase's preprocessing already relied on), each value multiplied
// by `scale`.
std::vector<float> hwcBgrToNchwFloat(const cv::Mat& hwcBgr, float scale);

// Applies `(x - mean[c]) / stddev[c]` in-place to each channel c of an
// NCHW float buffer already produced by hwcBgrToNchwFloat (channels in
// RGB order, matching that function's convention). A no-op if mean is
// all zeros and stddev is all ones (OnnxPreprocessingHints's default).
void applyChannelNormalization(
    std::vector<float>& nchw, int height, int width,
    const std::array<float, 3>& mean, const std::array<float, 3>& stddev);

// Wraps an ONNX Runtime output tensor's data into an owned (cloned)
// cv::Mat with dims taken from the tensor's shape -- the Ort::Value's
// buffer doesn't outlive the Run() call's result vector, so this must
// copy, not just view.
cv::Mat ortValueToMat(Ort::Value& value);

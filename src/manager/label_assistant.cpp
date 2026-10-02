#include "manager/label_assistant.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace {

bool isImageExtension(const std::filesystem::path& path) {
    static const std::vector<std::string> extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm"};
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

std::vector<std::filesystem::path> listImageFiles(const std::string& folderPath) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::exists(folderPath, ec) || !fs::is_directory(folderPath, ec)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(folderPath, ec)) {
        if (entry.is_regular_file() && isImageExtension(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

LabelAssistantResult runAutoLabel(
    const std::string& imageFolderPath,
    ModelTask mode,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& classify,
    const std::function<std::vector<Detection>(const cv::Mat&)>& detect,
    const std::function<void(int completed, int total)>& onProgress,
    const std::atomic<bool>* cancelRequested) {
    LabelAssistantResult result;

    const auto files = listImageFiles(imageFolderPath);
    if (files.empty()) {
        result.error = "No recognized image files found in: " + imageFolderPath;
        return result;
    }

    const int totalFiles = static_cast<int>(files.size());
    int completed = 0;

    for (const auto& path : files) {
        if (cancelRequested != nullptr && cancelRequested->load()) {
            return result;
        }
        const cv::Mat frame = cv::imread(path.string());
        if (frame.empty()) {
            continue;
        }
        completed++;

        if (mode == ModelTask::Classification) {
            const std::vector<ClassPrediction> predictions = classify(frame);
            if (!predictions.empty()) {
                DraftClassificationLabel draft;
                draft.imageFilename = path.filename().string();
                draft.predictedLabel = predictions.front().className;
                draft.confidence = predictions.front().probability;
                result.classificationDrafts.push_back(std::move(draft));
            }
        } else {
            const std::vector<Detection> detections = detect(frame);
            DraftDetectionLabel draft;
            draft.imageFilename = path.filename().string();
            draft.imageWidth = frame.cols;
            draft.imageHeight = frame.rows;
            draft.boxes.reserve(detections.size());
            for (const auto& detection : detections) {
                DraftDetectionBox box;
                box.box = detection.box;
                box.className = detection.className;
                box.confidence = detection.confidence;
                box.rotationDegrees = detection.rotationDegrees;
                draft.boxes.push_back(std::move(box));
            }
            result.detectionDrafts.push_back(std::move(draft));
        }

        if (onProgress) {
            onProgress(completed, totalFiles);
        }
    }

    result.imagesProcessed = completed;
    return result;
}

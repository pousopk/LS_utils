#include "manager/batch_evaluation.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
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

const ImageGroundTruth* findGroundTruth(const LabelStudioImportResult* groundTruth, const std::string& filename) {
    if (!groundTruth) {
        return nullptr;
    }
    for (const auto& img : groundTruth->images) {
        if (img.imageFilename == filename) {
            return &img;
        }
    }
    return nullptr;
}

} // namespace

TimingStats computeTimingStats(std::vector<double> inferenceMsSamples) {
    TimingStats stats;
    stats.count = static_cast<int>(inferenceMsSamples.size());
    if (stats.count == 0) {
        return stats;
    }

    std::sort(inferenceMsSamples.begin(), inferenceMsSamples.end());
    double sum = 0.0;
    for (double v : inferenceMsSamples) {
        sum += v;
    }
    stats.meanMs = sum / stats.count;

    const auto percentile = [&](double p) {
        const double idx = p * (stats.count - 1);
        const size_t lower = static_cast<size_t>(std::floor(idx));
        const size_t upper = static_cast<size_t>(std::ceil(idx));
        if (lower == upper) {
            return inferenceMsSamples[lower];
        }
        const double frac = idx - static_cast<double>(lower);
        return inferenceMsSamples[lower] * (1.0 - frac) + inferenceMsSamples[upper] * frac;
    };
    stats.medianMs = percentile(0.5);
    stats.p95Ms = percentile(0.95);
    return stats;
}

BatchEvaluationResult runDetectionBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<Detection>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress,
    const std::atomic<bool>* cancelRequested) {
    BatchEvaluationResult result;

    const auto files = listImageFiles(imageFolderPath);
    if (files.empty()) {
        result.error = "No recognized image files found in: " + imageFolderPath;
        return result;
    }

    const int totalFiles = static_cast<int>(files.size());
    std::vector<double> timings;
    for (const auto& path : files) {
        if (cancelRequested != nullptr && cancelRequested->load()) {
            break;
        }
        const cv::Mat frame = cv::imread(path.string());
        if (frame.empty()) {
            continue;
        }
        result.imagesFound++;

        BatchImageResult imageResult;
        imageResult.imageFilename = path.filename().string();

        const auto startedAt = std::chrono::steady_clock::now();
        imageResult.detections = infer(frame);
        imageResult.inferenceMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - startedAt).count();
        timings.push_back(imageResult.inferenceMs);

        if (const ImageGroundTruth* gt = findGroundTruth(groundTruth, imageResult.imageFilename)) {
            imageResult.groundTruthBoxes = gt->boxes;
            imageResult.hasGroundTruth = true;
            result.imagesWithGroundTruth++;
        }

        result.images.push_back(std::move(imageResult));
        if (onProgress) {
            onProgress(result.imagesFound, totalFiles);
        }
    }

    result.timing = computeTimingStats(timings);
    return result;
}

BatchEvaluationResult runClassificationBatchEvaluation(
    const std::string& imageFolderPath,
    const LabelStudioImportResult* groundTruth,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& infer,
    const std::function<void(int completed, int total)>& onProgress,
    const std::atomic<bool>* cancelRequested) {
    BatchEvaluationResult result;

    const auto files = listImageFiles(imageFolderPath);
    if (files.empty()) {
        result.error = "No recognized image files found in: " + imageFolderPath;
        return result;
    }

    const int totalFiles = static_cast<int>(files.size());
    std::vector<double> timings;
    for (const auto& path : files) {
        if (cancelRequested != nullptr && cancelRequested->load()) {
            break;
        }
        const cv::Mat frame = cv::imread(path.string());
        if (frame.empty()) {
            continue;
        }
        result.imagesFound++;

        BatchImageResult imageResult;
        imageResult.imageFilename = path.filename().string();

        const auto startedAt = std::chrono::steady_clock::now();
        imageResult.predictions = infer(frame);
        imageResult.inferenceMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - startedAt).count();
        timings.push_back(imageResult.inferenceMs);

        if (const ImageGroundTruth* gt = findGroundTruth(groundTruth, imageResult.imageFilename)) {
            imageResult.groundTruthLabel = gt->classificationLabel;
            imageResult.hasGroundTruth = true;
            result.imagesWithGroundTruth++;
        }

        result.images.push_back(std::move(imageResult));
        if (onProgress) {
            onProgress(result.imagesFound, totalFiles);
        }
    }

    result.timing = computeTimingStats(timings);
    return result;
}

std::vector<DetectionEvaluationItem> toDetectionEvaluationItems(const BatchEvaluationResult& result) {
    std::vector<DetectionEvaluationItem> items;
    for (const auto& image : result.images) {
        if (!image.hasGroundTruth) {
            continue;
        }
        DetectionEvaluationItem item;
        item.predictions = image.detections;
        item.groundTruth = image.groundTruthBoxes;
        items.push_back(std::move(item));
    }
    return items;
}

std::vector<ClassificationEvaluationItem> toClassificationEvaluationItems(const BatchEvaluationResult& result) {
    std::vector<ClassificationEvaluationItem> items;
    for (const auto& image : result.images) {
        if (!image.hasGroundTruth || image.predictions.empty()) {
            continue;
        }
        ClassificationEvaluationItem item;
        item.predictedLabel = image.predictions.front().className;  // top-1
        item.trueLabel = image.groundTruthLabel;
        items.push_back(std::move(item));
    }
    return items;
}

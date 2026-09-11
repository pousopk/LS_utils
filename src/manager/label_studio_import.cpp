#include "manager/label_studio_import.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace {

// Finds the first value in `data` whose basename looks like an image file
// (ends in a common image extension), returning just the basename -- the
// full URL/upload-path Label Studio stores generally won't match a local
// filesystem path.
std::string findImageFilename(const nlohmann::json& data) {
    static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
    if (!data.is_object()) {
        return "";
    }
    for (auto it = data.begin(); it != data.end(); ++it) {
        if (!it.value().is_string()) {
            continue;
        }
        const std::string value = it.value().get<std::string>();
        for (const auto& ext : extensions) {
            if (value.size() >= ext.size()
                && std::equal(ext.rbegin(), ext.rend(), value.rbegin(), [](char a, char b) {
                       return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
                   })) {
                const size_t slashPos = value.find_last_of("/\\");
                return slashPos == std::string::npos ? value : value.substr(slashPos + 1);
            }
        }
    }
    return "";
}

} // namespace

LabelStudioImportResult parseLabelStudioExport(const nlohmann::json& tasks) {
    LabelStudioImportResult result;
    if (!tasks.is_array()) {
        result.error = "Expected a JSON array of tasks at the top level";
        return result;
    }

    for (const auto& task : tasks) {
        if (!task.contains("data") || !task.contains("annotations") || !task["annotations"].is_array()
            || task["annotations"].empty()) {
            result.skippedCount++;
            continue;
        }

        const std::string imageFilename = findImageFilename(task["data"]);
        if (imageFilename.empty()) {
            result.skippedCount++;
            continue;
        }

        const auto& firstAnnotation = task["annotations"][0];
        if (!firstAnnotation.contains("result") || !firstAnnotation["result"].is_array()) {
            result.skippedCount++;
            continue;
        }

        ImageGroundTruth groundTruth;
        groundTruth.imageFilename = imageFilename;

        for (const auto& item : firstAnnotation["result"]) {
            if (!item.contains("type") || !item.contains("value")) {
                result.skippedCount++;
                continue;
            }
            const std::string type = item["type"].get<std::string>();
            const auto& value = item["value"];

            if (type == "rectanglelabels") {
                groundTruth.hasDetectionAnnotation = true;
                if (value.contains("rotation") && value["rotation"].get<double>() != 0.0) {
                    result.skippedCount++;
                    continue;
                }
                if (!value.contains("rectanglelabels") || !value["rectanglelabels"].is_array()
                    || value["rectanglelabels"].empty()) {
                    result.skippedCount++;
                    continue;
                }
                if (!item.contains("original_width") || !item.contains("original_height")) {
                    result.skippedCount++;
                    continue;
                }

                const double originalWidth = item["original_width"].get<double>();
                const double originalHeight = item["original_height"].get<double>();

                GroundTruthBox box;
                box.box = cv::Rect(
                    static_cast<int>(std::round(value["x"].get<double>() / 100.0 * originalWidth)),
                    static_cast<int>(std::round(value["y"].get<double>() / 100.0 * originalHeight)),
                    static_cast<int>(std::round(value["width"].get<double>() / 100.0 * originalWidth)),
                    static_cast<int>(std::round(value["height"].get<double>() / 100.0 * originalHeight)));
                box.className = value["rectanglelabels"][0].get<std::string>();
                groundTruth.boxes.push_back(std::move(box));
            } else if (type == "choices") {
                groundTruth.hasClassificationAnnotation = true;
                if (!value.contains("choices") || !value["choices"].is_array() || value["choices"].empty()) {
                    result.skippedCount++;
                    continue;
                }
                groundTruth.classificationLabel = value["choices"][0].get<std::string>();
            } else {
                result.skippedCount++;
            }
        }

        result.images.push_back(std::move(groundTruth));
    }

    return result;
}

LabelStudioImportResult loadLabelStudioExport(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file) {
        LabelStudioImportResult result;
        result.error = "Could not open file: " + jsonPath;
        return result;
    }

    nlohmann::json tasks;
    try {
        file >> tasks;
    } catch (const nlohmann::json::parse_error& e) {
        LabelStudioImportResult result;
        result.error = std::string("Failed to parse JSON: ") + e.what();
        return result;
    }

    return parseLabelStudioExport(tasks);
}

PredictionResultAndScore buildClassificationPredictionResult(
    const DraftClassificationLabel& draft, const std::string& choicesFromName, const std::string& imageToName) {
    nlohmann::json result;
    result["from_name"] = choicesFromName;
    result["to_name"] = imageToName;
    result["type"] = "choices";
    result["value"]["choices"] = nlohmann::json::array({draft.predictedLabel});

    PredictionResultAndScore out;
    out.result = nlohmann::json::array({result});
    out.score = draft.confidence;
    return out;
}

PredictionResultAndScore buildDetectionPredictionResult(
    const DraftDetectionLabel& draft, const std::string& rectangleLabelsFromName, const std::string& imageToName) {
    nlohmann::json results = nlohmann::json::array();
    float confidenceSum = 0.0f;

    for (const auto& box : draft.boxes) {
        nlohmann::json result;
        result["from_name"] = rectangleLabelsFromName;
        result["to_name"] = imageToName;
        result["type"] = "rectanglelabels";
        result["original_width"] = draft.imageWidth;
        result["original_height"] = draft.imageHeight;

        const double width = draft.imageWidth > 0 ? static_cast<double>(draft.imageWidth) : 0.0;
        const double height = draft.imageHeight > 0 ? static_cast<double>(draft.imageHeight) : 0.0;
        result["value"]["x"] = width > 0.0 ? (static_cast<double>(box.box.x) / width * 100.0) : 0.0;
        result["value"]["y"] = height > 0.0 ? (static_cast<double>(box.box.y) / height * 100.0) : 0.0;
        result["value"]["width"] = width > 0.0 ? (static_cast<double>(box.box.width) / width * 100.0) : 0.0;
        result["value"]["height"] = height > 0.0 ? (static_cast<double>(box.box.height) / height * 100.0) : 0.0;
        result["value"]["rotation"] = 0;
        result["value"]["rectanglelabels"] = nlohmann::json::array({box.className});

        results.push_back(std::move(result));
        confidenceSum += box.confidence;
    }

    PredictionResultAndScore out;
    out.score = draft.boxes.empty() ? 0.0f : confidenceSum / static_cast<float>(draft.boxes.size());
    out.result = std::move(results);
    return out;
}

std::vector<DraftDetectionBox> parseDetectionResultBoxes(
    const nlohmann::json& resultArray, const std::string& rectangleLabelsFromName) {
    std::vector<DraftDetectionBox> boxes;
    if (!resultArray.is_array()) {
        return boxes;
    }

    for (const auto& item : resultArray) {
        if (!item.contains("type") || item["type"] != "rectanglelabels") {
            continue;
        }
        if (!item.contains("from_name") || item["from_name"] != rectangleLabelsFromName) {
            continue;
        }
        if (!item.contains("value") || !item.contains("original_width") || !item.contains("original_height")) {
            continue;
        }
        const auto& value = item["value"];
        if (value.contains("rotation") && value["rotation"].get<double>() != 0.0) {
            continue;
        }
        if (!value.contains("rectanglelabels") || !value["rectanglelabels"].is_array() || value["rectanglelabels"].empty()) {
            continue;
        }

        const double originalWidth = item["original_width"].get<double>();
        const double originalHeight = item["original_height"].get<double>();

        DraftDetectionBox box;
        box.box = cv::Rect(
            static_cast<int>(std::round(value["x"].get<double>() / 100.0 * originalWidth)),
            static_cast<int>(std::round(value["y"].get<double>() / 100.0 * originalHeight)),
            static_cast<int>(std::round(value["width"].get<double>() / 100.0 * originalWidth)),
            static_cast<int>(std::round(value["height"].get<double>() / 100.0 * originalHeight)));
        box.className = value["rectanglelabels"][0].get<std::string>();
        boxes.push_back(std::move(box));
    }

    return boxes;
}

std::optional<std::string> parseChoiceResultLabel(
    const nlohmann::json& resultArray, const std::string& choicesFromName) {
    if (!resultArray.is_array()) {
        return std::nullopt;
    }
    for (const auto& item : resultArray) {
        if (!item.contains("type") || item["type"] != "choices") {
            continue;
        }
        if (!item.contains("from_name") || item["from_name"] != choicesFromName) {
            continue;
        }
        if (!item.contains("value") || !item["value"].contains("choices") || !item["value"]["choices"].is_array()
            || item["value"]["choices"].empty()) {
            continue;
        }
        return item["value"]["choices"][0].get<std::string>();
    }
    return std::nullopt;
}


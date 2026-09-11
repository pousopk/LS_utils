#include "manager/label_studio_import.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace {

// MSB-first bit writer -- matches Python's f'{value:0Nb}' (fixed-width,
// most-significant-bit-first) fields concatenated into a byte stream,
// zero-padded to a byte boundary at the end. Mirrors encode_rle's
// out_str-building + bits2byte finalization exactly.
class RleBitWriter {
public:
    void writeBits(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; --i) {
            const uint8_t bit = static_cast<uint8_t>((value >> i) & 1u);
            currentByte_ = static_cast<uint8_t>((currentByte_ << 1) | bit);
            if (++bitsInCurrentByte_ == 8) {
                bytes_.push_back(currentByte_);
                currentByte_ = 0;
                bitsInCurrentByte_ = 0;
            }
        }
    }

    std::vector<int> finish() {
        // Matches Label Studio's own (quirky) padding exactly: its Python
        // reference computes `nzfill = 8 - (len(bits) % 8)`, which is 8 --
        // not 0 -- when the stream is already byte-aligned. That appends
        // one extra all-zero byte in the aligned case, not zero padding.
        // Verified against real ground-truth vectors; do not "fix" this
        // to skip padding when aligned, that would break interop.
        if (bitsInCurrentByte_ == 0) {
            bytes_.push_back(0);
        } else {
            currentByte_ = static_cast<uint8_t>(currentByte_ << (8 - bitsInCurrentByte_));
            bytes_.push_back(currentByte_);
            currentByte_ = 0;
            bitsInCurrentByte_ = 0;
        }
        return std::vector<int>(bytes_.begin(), bytes_.end());
    }

private:
    std::vector<uint8_t> bytes_;
    uint8_t currentByte_ = 0;
    int bitsInCurrentByte_ = 0;
};

// MSB-first bit reader over a byte sequence (each element 0-255) -- the
// inverse of RleBitWriter, matching decode_rle's InputStream/access_bit.
class RleBitReader {
public:
    explicit RleBitReader(const std::vector<int>& bytes) : bytes_(bytes) {}

    uint32_t readBits(int numBits) {
        uint32_t value = 0;
        for (int i = 0; i < numBits; ++i) {
            const int byteIndex = bitPos_ / 8;
            const int bitIndexInByte = 7 - (bitPos_ % 8);
            const uint8_t byteVal =
                byteIndex < static_cast<int>(bytes_.size()) ? static_cast<uint8_t>(bytes_[byteIndex]) : 0;
            const uint32_t bit = (byteVal >> bitIndexInByte) & 1u;
            value = (value << 1) | bit;
            ++bitPos_;
        }
        return value;
    }

private:
    const std::vector<int>& bytes_;
    int bitPos_ = 0;
};

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

std::vector<int> encodeMaskToLabelStudioRle(const cv::Mat& mask) {
    std::vector<uint8_t> array;
    array.reserve(static_cast<size_t>(mask.rows) * static_cast<size_t>(mask.cols) * 4);
    for (int r = 0; r < mask.rows; ++r) {
        const uint8_t* rowPtr = mask.ptr<uint8_t>(r);
        for (int c = 0; c < mask.cols; ++c) {
            const uint8_t value = rowPtr[c];
            array.push_back(value);
            array.push_back(value);
            array.push_back(value);
            array.push_back(value);
        }
    }

    const uint32_t num = static_cast<uint32_t>(array.size());
    constexpr int wordSize = 8;
    constexpr int rleSizes[4] = {3, 4, 8, 16};

    RleBitWriter writer;
    writer.writeBits(num, 32);
    writer.writeBits(static_cast<uint32_t>(wordSize - 1), 5);
    for (const int size : rleSizes) {
        writer.writeBits(static_cast<uint32_t>(size - 1), 4);
    }

    size_t i = 0;
    while (i < array.size()) {
        size_t runEnd = i + 1;
        while (runEnd < array.size() && array[runEnd] == array[i]) {
            ++runEnd;
        }
        const size_t runLength = runEnd - i;
        const uint8_t value = array[i];

        if (runLength == 1) {
            writer.writeBits(0, 1); // individual-values flag
            writer.writeBits(0, 2); // rle-size index 0
            writer.writeBits(0, 3); // length - 1 = 0
            writer.writeBits(value, 8);
        } else {
            size_t remaining = runLength;
            while (remaining > 0) {
                const size_t chunk = std::min<size_t>(remaining, 65536);
                writer.writeBits(1, 1); // repeated flag
                if (chunk <= 8) {
                    writer.writeBits(0, 2);
                    writer.writeBits(static_cast<uint32_t>(chunk - 1), 3);
                } else if (chunk <= 16) {
                    writer.writeBits(1, 2);
                    writer.writeBits(static_cast<uint32_t>(chunk - 1), 4);
                } else if (chunk <= 256) {
                    writer.writeBits(2, 2);
                    writer.writeBits(static_cast<uint32_t>(chunk - 1), 8);
                } else {
                    writer.writeBits(3, 2);
                    writer.writeBits(static_cast<uint32_t>(chunk - 1), 16);
                }
                writer.writeBits(value, 8);
                remaining -= chunk;
            }
        }
        i = runEnd;
    }

    return writer.finish();
}

cv::Mat decodeLabelStudioRleToMask(const std::vector<int>& rle, int width, int height) {
    RleBitReader reader(rle);
    const uint32_t num = reader.readBits(32);
    const uint32_t wordSize = reader.readBits(5) + 1;
    uint32_t rleSizes[4];
    for (uint32_t& size : rleSizes) {
        size = reader.readBits(4) + 1;
    }

    std::vector<uint8_t> out(num, 0);
    uint32_t i = 0;
    while (i < num) {
        const uint32_t flag = reader.readBits(1);
        const uint32_t sizeIndex = reader.readBits(2);
        const uint32_t runLength = reader.readBits(rleSizes[sizeIndex]) + 1;
        const uint32_t j = std::min(i + runLength, num);
        if (flag) {
            const uint32_t value = reader.readBits(wordSize);
            for (uint32_t k = i; k < j; ++k) {
                out[k] = static_cast<uint8_t>(value);
            }
            i = j;
        } else {
            while (i < j) {
                const uint32_t value = reader.readBits(wordSize);
                out[i] = static_cast<uint8_t>(value);
                ++i;
            }
        }
    }

    // Every pixel was repeated 4x (RGBA-shaped); Label Studio reads back
    // channel index 3 (see decode_from_annotation's `[:, :, 3]`). Since
    // this app writes all 4 copies identically, any index would do.
    cv::Mat mask(height, width, CV_8UC1);
    for (int r = 0; r < height; ++r) {
        uint8_t* rowPtr = mask.ptr<uint8_t>(r);
        for (int c = 0; c < width; ++c) {
            const size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c)) * 4 + 3;
            rowPtr[c] = idx < out.size() ? out[idx] : 0;
        }
    }
    return mask;
}

nlohmann::json buildBrushLabelResult(
    const std::vector<DraftBrushRegion>& regions, const std::string& brushLabelsFromName,
    const std::string& imageToName, int imageWidth, int imageHeight) {
    nlohmann::json results = nlohmann::json::array();
    for (const auto& region : regions) {
        if (region.mask.empty()) {
            continue;
        }
        nlohmann::json item;
        item["type"] = "brushlabels";
        item["from_name"] = brushLabelsFromName;
        item["to_name"] = imageToName;
        item["original_width"] = imageWidth;
        item["original_height"] = imageHeight;
        item["image_rotation"] = 0;
        item["value"]["format"] = "rle";
        item["value"]["rle"] = encodeMaskToLabelStudioRle(region.mask);
        item["value"]["brushlabels"] = nlohmann::json::array({region.className});
        results.push_back(std::move(item));
    }
    return results;
}

std::vector<DraftBrushRegion> parseBrushResultRegions(
    const nlohmann::json& resultArray, const std::string& brushLabelsFromName) {
    std::vector<DraftBrushRegion> regions;
    if (!resultArray.is_array()) {
        return regions;
    }

    for (const auto& item : resultArray) {
        if (!item.contains("type") || item["type"] != "brushlabels") {
            continue;
        }
        if (!item.contains("from_name") || item["from_name"] != brushLabelsFromName) {
            continue;
        }
        if (!item.contains("value") || !item["value"].contains("rle") || !item["value"]["rle"].is_array()) {
            continue;
        }
        if (!item.contains("original_width") || !item.contains("original_height")) {
            continue;
        }
        if (!item["value"].contains("brushlabels") || !item["value"]["brushlabels"].is_array()
            || item["value"]["brushlabels"].empty()) {
            continue;
        }

        const int width = item["original_width"].get<int>();
        const int height = item["original_height"].get<int>();
        std::vector<int> rle;
        for (const auto& byteValue : item["value"]["rle"]) {
            rle.push_back(byteValue.get<int>());
        }

        DraftBrushRegion region;
        region.mask = decodeLabelStudioRleToMask(rle, width, height);
        region.className = item["value"]["brushlabels"][0].get<std::string>();
        regions.push_back(std::move(region));
    }

    return regions;
}


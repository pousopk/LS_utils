#include "manager/onnx_metadata.hpp"

#include <cctype>
#include <fstream>
#include <map>
#include <optional>

bool loadOnnxModelProto(const std::string& path, onnx::ModelProto& out, std::string& errorOut) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        errorOut = "Could not open file: " + path;
        return false;
    }
    if (!out.ParseFromIstream(&file)) {
        errorOut = "Failed to parse as an ONNX model: " + path;
        return false;
    }
    return true;
}

bool extractInputShape(const onnx::ModelProto& model, OnnxInputShape& out, std::string& errorOut) {
    if (model.graph().input_size() == 0) {
        errorOut = "Model graph has no inputs";
        return false;
    }
    const onnx::ValueInfoProto& input = model.graph().input(0);
    if (!input.type().has_tensor_type()) {
        errorOut = "Model's first input is not a tensor type";
        return false;
    }
    const auto& tensorType = input.type().tensor_type();
    if (!tensorType.has_shape() || tensorType.shape().dim_size() != 4) {
        errorOut = "Model's first input does not have a 4D shape (expected NCHW)";
        return false;
    }

    int64_t dims[4];
    for (int i = 0; i < 4; ++i) {
        const auto& dim = tensorType.shape().dim(i);
        if (dim.value_case() != onnx::TensorShapeProto_Dimension::kDimValue) {
            errorOut = "Model's input has a dynamic/symbolic dimension (expected fixed NCHW)";
            return false;
        }
        dims[i] = dim.dim_value();
    }

    out.channels = static_cast<int>(dims[1]);
    out.height = static_cast<int>(dims[2]);
    out.width = static_cast<int>(dims[3]);
    return true;
}

namespace {
std::vector<std::string> parseNamesDict(const std::string& text) {
    std::map<int, std::string> byIndex;
    size_t pos = text.find('{');
    if (pos == std::string::npos) {
        return {};
    }
    ++pos;
    while (pos < text.size()) {
        while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == ',')) {
            ++pos;
        }
        if (pos >= text.size() || text[pos] == '}') {
            break;
        }
        const size_t colon = text.find(':', pos);
        if (colon == std::string::npos) {
            return {};
        }
        int key = 0;
        try {
            key = std::stoi(text.substr(pos, colon - pos));
        } catch (const std::exception&) {
            return {};
        }
        pos = colon + 1;
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos >= text.size() || (text[pos] != '\'' && text[pos] != '"')) {
            return {};
        }
        const char quote = text[pos];
        ++pos;
        const size_t valueEnd = text.find(quote, pos);
        if (valueEnd == std::string::npos) {
            return {};
        }
        byIndex[key] = text.substr(pos, valueEnd - pos);
        pos = valueEnd + 1;
    }
    if (byIndex.empty()) {
        return {};
    }
    std::vector<std::string> names;
    names.reserve(byIndex.size());
    for (const auto& entry : byIndex) {
        names.push_back(entry.second);
    }
    return names;
}
std::vector<std::string> parseCategoriesList(const std::string& text) {
    std::vector<std::string> names;
    size_t pos = 0;
    while (true) {
        pos = text.find("'name'", pos);
        if (pos == std::string::npos) {
            break;
        }
        pos = text.find(':', pos);
        if (pos == std::string::npos) {
            break;
        }
        ++pos;
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos >= text.size() || (text[pos] != '\'' && text[pos] != '"')) {
            break;
        }
        const char quote = text[pos];
        ++pos;
        const size_t valueEnd = text.find(quote, pos);
        if (valueEnd == std::string::npos) {
            break;
        }
        names.push_back(text.substr(pos, valueEnd - pos));
        pos = valueEnd + 1;
    }
    return names;
}

// Finds 'key': <value> inside a Python-dict-repr string and returns the
// raw text immediately after the colon, up to the next top-level ',' or
// closing bracket (trimmed). Returns nullopt if the key isn't found.
std::optional<std::string> findRawValueAfterKey(const std::string& text, const std::string& key) {
    const std::string pattern = "'" + key + "'";
    size_t pos = text.find(pattern);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = text.find(':', pos + pattern.size());
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }

    size_t end = pos;
    int depth = 0;
    while (end < text.size()) {
        const char c = text[end];
        if (c == '{' || c == '[') {
            ++depth;
        } else if (c == '}' || c == ']') {
            if (depth == 0) {
                break;
            }
            --depth;
        } else if (c == ',' && depth == 0) {
            break;
        }
        ++end;
    }

    std::string raw = text.substr(pos, end - pos);
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) {
        raw.pop_back();
    }
    return raw;
}

std::optional<double> parseNumber(const std::string& raw) {
    try {
        size_t idx = 0;
        const double value = std::stod(raw, &idx);
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<bool> parseBool(const std::string& raw) {
    if (raw == "True") {
        return true;
    }
    if (raw == "False") {
        return false;
    }
    return std::nullopt;
}

std::string parseQuotedString(const std::string& raw) {
    if (raw.size() >= 2 && (raw.front() == '\'' || raw.front() == '"') && raw.back() == raw.front()) {
        return raw.substr(1, raw.size() - 2);
    }
    return raw;
}

} // namespace

std::vector<std::string> extractClassNames(const onnx::ModelProto& model) {
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "names") {
            if (auto names = parseNamesDict(prop.value()); !names.empty()) {
                return names;
            }
        }
    }
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "categories") {
            if (auto names = parseCategoriesList(prop.value()); !names.empty()) {
                return names;
            }
        }
    }
    return {};
}

OnnxPreprocessingHints extractPreprocessingHints(const onnx::ModelProto& model) {
    OnnxPreprocessingHints hints;

    std::string pixelNormalization;
    std::string padding;
    std::string maintainAspectRatio;
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "pixel_normalization") {
            pixelNormalization = prop.value();
        } else if (prop.key() == "padding") {
            padding = prop.value();
        } else if (prop.key() == "maintain_aspect_ratio") {
            maintainAspectRatio = prop.value();
        }
    }

    if (!pixelNormalization.empty()) {
        bool rawPixelRange = false;
        if (auto maxValue = findRawValueAfterKey(pixelNormalization, "max_value")) {
            if (auto v = parseNumber(*maxValue); v && *v > 1.0) {
                rawPixelRange = true;
            }
        }
        if (auto value = findRawValueAfterKey(pixelNormalization, "value")) {
            if (auto v = parseNumber(*value); v && *v > 1.0) {
                const auto enabled = findRawValueAfterKey(pixelNormalization, "enabled");
                const bool isEnabled = !enabled || parseBool(*enabled).value_or(true);
                if (isEnabled) {
                    rawPixelRange = true;
                }
            }
        }
        if (rawPixelRange) {
            hints.inputScale = 1.0f;
        }
    }

    if (!padding.empty()) {
        if (auto position = findRawValueAfterKey(padding, "position")) {
            const std::string pos = parseQuotedString(*position);
            if (pos == "top_left") {
                hints.padCenter = false;
            } else if (pos == "center") {
                hints.padCenter = true;
            }
        }
        if (auto fill = findRawValueAfterKey(padding, "fill")) {
            if (auto v = parseNumber(*fill)) {
                hints.padFill = static_cast<float>(*v);
            }
        }
    }

    if (!maintainAspectRatio.empty()) {
        if (auto v = parseBool(maintainAspectRatio)) {
            hints.maintainAspectRatio = *v;
        }
    }

    return hints;
}

AnomalyScoreHints extractAnomalyScoreHints(const onnx::ModelProto& model) {
    AnomalyScoreHints hints;

    std::string vadParams;
    std::string autoThreshold;
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "vad_params") {
            vadParams = prop.value();
        } else if (prop.key() == "auto_threshold") {
            autoThreshold = prop.value();
        }
    }

    if (!vadParams.empty()) {
        const auto minimum = findRawValueAfterKey(vadParams, "minimum");
        const auto maximum = findRawValueAfterKey(vadParams, "maximum");
        if (minimum && maximum) {
            const auto minVal = parseNumber(*minimum);
            const auto maxVal = parseNumber(*maximum);
            if (minVal && maxVal) {
                hints.minimum = static_cast<float>(*minVal);
                hints.maximum = static_cast<float>(*maxVal);
                hints.present = true;
            }
        }
    }

    if (!autoThreshold.empty()) {
        if (const auto v = parseNumber(autoThreshold)) {
            hints.threshold = static_cast<float>(*v);
            hints.present = true;
        }
    }

    return hints;
}

std::string extractTaskHint(const onnx::ModelProto& model) {
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "task") {
            return prop.value();
        }
    }
    return {};
}

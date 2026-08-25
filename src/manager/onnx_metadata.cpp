#include "manager/onnx_metadata.hpp"

#include <cctype>
#include <fstream>
#include <map>

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
} // namespace

std::vector<std::string> extractClassNames(const onnx::ModelProto& model) {
    for (const auto& prop : model.metadata_props()) {
        if (prop.key() == "names") {
            return parseNamesDict(prop.value());
        }
    }
    return {};
}

#pragma once

#include "onnx-ml.pb.h"

#include <string>
#include <vector>

struct OnnxInputShape {
    int width = 0;
    int height = 0;
    int channels = 0;
};

// Reads and parses an ONNX model file into `out`. Fails cleanly (returns
// false + message) on a missing/corrupt/non-ONNX file.
bool loadOnnxModelProto(const std::string& path, onnx::ModelProto& out, std::string& errorOut);

// Reads `model.graph().input(0)`'s declared tensor shape (NCHW). Requires
// exactly 4 concrete (non-symbolic) dimensions -- fails rather than
// guessing if any dimension is dynamic or the shape isn't 4D.
bool extractInputShape(const onnx::ModelProto& model, OnnxInputShape& out, std::string& errorOut);

// Best-effort parse of a `names` metadata_props entry (Ultralytics-style
// embedded Python dict repr, e.g. "{0: 'person', 1: 'car'}") into an
// ordered vector by integer key. Returns an empty vector if the entry is
// absent or doesn't match that shape -- never throws.
std::vector<std::string> extractClassNames(const onnx::ModelProto& model);

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

// Best-effort parse of class names from metadata_props, trying two known
// shapes in order: an Ultralytics-style `names` dict repr (e.g.
// "{0: 'person', 1: 'car'}"), or a `categories` list of dicts (e.g.
// "[{'id': ..., 'name': 'car', 'threshold': ...}, ...]"), taking each
// dict's `name` field in list order. Returns an empty vector if neither
// is present/parseable -- never throws.
std::vector<std::string> extractClassNames(const onnx::ModelProto& model);

// Preprocessing parameters inferred from a model's metadata, each
// defaulting to this app's existing hardcoded behavior so a model with no
// such metadata behaves exactly as before.
struct OnnxPreprocessingHints {
    // Multiplier applied to raw 0-255 pixel values before feeding the
    // network. 1/255 (the default) matches the standard Ultralytics
    // convention (caller normalizes); 1.0 means the graph normalizes
    // internally (a `pixel_normalization` metadata entry declaring a
    // 0-255 range) and raw pixel values should be passed through.
    float inputScale = 1.0f / 255.0f;
    // Letterbox fill value (0-255, replicated across channels).
    float padFill = 114.0f;
    // true = center the image in the padded canvas (this app's existing
    // detection behavior); false = anchor it top-left, per a `padding`
    // metadata entry with position "top_left".
    bool padCenter = true;
    // Classification only: whether to letterbox-pad (preserving aspect
    // ratio) instead of this app's existing plain (aspect-distorting)
    // resize. Driven by a `maintain_aspect_ratio` metadata entry.
    bool maintainAspectRatio = false;
};

// Reads `pixel_normalization`, `padding`, and `maintain_aspect_ratio`
// metadata_props entries (each optional, each independently best-effort)
// into an OnnxPreprocessingHints. Any entry that's absent or doesn't parse
// leaves the corresponding field at its default -- never throws.
OnnxPreprocessingHints extractPreprocessingHints(const onnx::ModelProto& model);

#pragma once

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct GroundTruthBox {
    cv::Rect box;            // pixel coordinates, original image space
    std::string className;
};

struct ImageGroundTruth {
    std::string imageFilename;           // basename, used for matching to dataset folder files
    std::vector<GroundTruthBox> boxes;   // populated for detection ("rectanglelabels") tasks
    std::string classificationLabel;     // populated for classification ("choices") tasks (empty if none)

    // Whether this image actually had a "rectanglelabels"/"choices" result
    // present, regardless of whether it produced usable data (e.g. a
    // rotated box still sets hasDetectionAnnotation even though it's
    // skipped from `boxes`). Distinguishes "labeled with zero objects"
    // from "never labeled for this task type" -- an image can have both,
    // one, or neither, independent of the other.
    bool hasDetectionAnnotation = false;
    bool hasClassificationAnnotation = false;
};

struct LabelStudioImportResult {
    std::vector<ImageGroundTruth> images;
    int skippedCount = 0;   // rotated boxes, unrecognized image fields, unsupported result types, etc.
    std::string error;      // set only on a hard failure (unreadable file / not valid JSON / not an array)
};

struct DraftClassificationLabel {
    std::string imageFilename;   // basename, matching this file's existing convention
    std::string predictedLabel;
    float confidence = 0.0f;     // 0.0-1.0, mapped directly to Label Studio's prediction "score"
};

struct DraftDetectionBox {
    cv::Rect box;                 // pixel coordinates, original image space
    std::string className;
    float confidence = 0.0f;
};

struct DraftDetectionLabel {
    std::string imageFilename;    // basename, matching this file's existing convention
    int imageWidth = 0;           // original image dimensions -- needed to convert
    int imageHeight = 0;          // box pixel coordinates to Label Studio's percentage convention
    std::vector<DraftDetectionBox> boxes;
};

// Parses an already-loaded Label Studio export (a JSON array of tasks)
// into ground truth. Pure function -- no file I/O, unit tested directly.
LabelStudioImportResult parseLabelStudioExport(const nlohmann::json& tasks);

// Reads and parses a Label Studio JSON export file. Fails cleanly (sets
// `error`, leaves `images` empty) on a missing/unreadable file or JSON
// that isn't an array of tasks.
LabelStudioImportResult loadLabelStudioExport(const std::string& jsonPath);

struct PredictionResultAndScore {
    nlohmann::json result;   // the Label Studio "result" array for one task's prediction
    float score = 0.0f;
};

// Builds the `result` array + `score` for a single classification draft,
// ready to attach to an already-existing Label Studio task (one task at a
// time, via its Predictions API). Pure.
PredictionResultAndScore buildClassificationPredictionResult(
    const DraftClassificationLabel& draft, const std::string& choicesFromName, const std::string& imageToName);

// Builds the `result` array (one entry per box) + mean-confidence `score`
// for a single detection draft. Pure.
PredictionResultAndScore buildDetectionPredictionResult(
    const DraftDetectionLabel& draft, const std::string& rectangleLabelsFromName, const std::string& imageToName);

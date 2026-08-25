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
};

struct LabelStudioImportResult {
    std::vector<ImageGroundTruth> images;
    int skippedCount = 0;   // rotated boxes, unrecognized image fields, unsupported result types, etc.
    std::string error;      // set only on a hard failure (unreadable file / not valid JSON / not an array)
};

// Parses an already-loaded Label Studio export (a JSON array of tasks)
// into ground truth. Pure function -- no file I/O, unit tested directly.
LabelStudioImportResult parseLabelStudioExport(const nlohmann::json& tasks);

// Reads and parses a Label Studio JSON export file. Fails cleanly (sets
// `error`, leaves `images` empty) on a missing/unreadable file or JSON
// that isn't an array of tasks.
LabelStudioImportResult loadLabelStudioExport(const std::string& jsonPath);

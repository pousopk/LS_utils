#pragma once

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <optional>
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

// Pure function: the inverse of buildDetectionPredictionResult -- extracts
// every "rectanglelabels" result item whose from_name matches
// `rectangleLabelsFromName` back into pixel-space boxes (using each
// item's own original_width/original_height, converting its x/y/width/
// height percentages back to pixels the same way parseLabelStudioExport
// does). Items with a nonzero rotation are skipped (same as
// parseLabelStudioExport -- this app doesn't support rotated boxes).
// confidence is not meaningful for a real annotation's box and is left at
// its default (0.0f). Used to seed the box editor from an existing
// annotation or prediction.
std::vector<DraftDetectionBox> parseDetectionResultBoxes(
    const nlohmann::json& resultArray, const std::string& rectangleLabelsFromName);

// Pure function: the inverse of buildClassificationPredictionResult --
// returns the first "choices" result item's first choice value whose
// from_name matches `choicesFromName`, or std::nullopt if no such item
// exists. This app's classification editor is single-select for phase 1,
// so only the first choice is used even if a result item somehow has
// more than one.
std::optional<std::string> parseChoiceResultLabel(
    const nlohmann::json& resultArray, const std::string& choicesFromName);

// Pure function: encodes a binary mask (CV_8UC1, values expected to be 0
// or 255) into Label Studio's proprietary brush RLE format -- a verbatim
// port of label-studio-converter's encode_rle/mask2rle (brush.py). Each
// pixel is repeated 4x (Label Studio's RGBA-shaped layout; this app
// writes all 4 copies identically) and flattened row-major before
// encoding. See the phase 3 design doc for the exact bit-packing format
// and the ground-truth vectors this is tested against.
std::vector<int> encodeMaskToLabelStudioRle(const cv::Mat& mask);

// Pure function: the inverse -- decodes an RLE byte sequence (each int
// 0-255, as Label Studio's JSON `rle` field represents it) back into a
// `height` x `width` CV_8UC1 mask. A verbatim port of
// label-studio-converter's decode_rle.
cv::Mat decodeLabelStudioRleToMask(const std::vector<int>& rle, int width, int height);

struct DraftBrushRegion {
    cv::Mat mask;             // CV_8UC1, values 0 or 255, same dimensions as the image
    std::string className;
};

// Builds one "brushlabels" result item per region (skipping any with an
// empty mask) via encodeMaskToLabelStudioRle -- mirrors
// buildDetectionPredictionResult's shape, but masks are direct annotation
// data, not predictions, so there's no confidence/score field.
nlohmann::json buildBrushLabelResult(
    const std::vector<DraftBrushRegion>& regions, const std::string& brushLabelsFromName,
    const std::string& imageToName, int imageWidth, int imageHeight);

// The inverse: decodes every "brushlabels" result item whose from_name
// matches `brushLabelsFromName` back into DraftBrushRegions, using each
// item's own original_width/original_height (same per-item convention as
// parseDetectionResultBoxes). Items missing an "rle" value, or with no
// brushlabels class name, are skipped.
std::vector<DraftBrushRegion> parseBrushResultRegions(
    const nlohmann::json& resultArray, const std::string& brushLabelsFromName);

#pragma once

#include "manager/classification_inference.hpp"
#include "manager/model_task.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

struct LabelAssistantResult {
    std::vector<DraftClassificationLabel> classificationDrafts;  // populated when mode == Classification
    std::vector<DraftDetectionLabel> detectionDrafts;             // populated when mode == Detection
    int imagesProcessed = 0;   // successfully-decoded images run through the model
    std::string error;         // hard failure: folder doesn't exist / no recognized images
};

// Scans `imageFolderPath` for recognized image files and runs `classify`
// (mode == Classification) or `detect` (mode == Detection) on each one --
// exactly one of the two is invoked, matching `mode`; the other may be a
// null std::function and is never called. Both modes produce exactly one
// draft per successfully-decoded image: Classification's is its top-1
// prediction; Detection's holds however many boxes were found, including
// zero -- an image with no detections still gets a draft (with an empty
// `boxes` list), so the caller can show/filter/export it like any other
// result instead of it silently vanishing from the run.
// `onProgress` (if non-null) is called once per successfully-decoded image
// processed, with `total` fixed at the whole folder's recognized-file
// count. `cancelRequested` (if non-null and observed true) stops the loop
// early, returning whatever was completed so far with `error` left empty
// (a cancelled run is not a hard failure).
LabelAssistantResult runAutoLabel(
    const std::string& imageFolderPath,
    ModelTask mode,
    const std::function<std::vector<ClassPrediction>(const cv::Mat&)>& classify,
    const std::function<std::vector<Detection>(const cv::Mat&)>& detect,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

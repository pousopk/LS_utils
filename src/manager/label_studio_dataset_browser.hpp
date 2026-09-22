#pragma once

#include "manager/label_studio_import.hpp"

#include <nlohmann/json_fwd.hpp>

#include <optional>
#include <string>
#include <vector>

// One task's browsing-relevant summary, extracted from a Label Studio
// tasks-list API response (the same raw shape fetchAllLabelStudioTasksRaw
// produces) -- everything the Dataset Browser's filters and grid need,
// without re-parsing the raw JSON on every filter change.
struct DatasetTaskSummary {
    int taskId = 0;
    std::string imagePath;                  // task.data[dataImageKey]
    bool hasAnnotation = false;
    bool hasPrediction = false;
    std::vector<std::string> classNames;    // every class name in any annotation/prediction result, deduplicated and sorted
    std::optional<float> minConfidence;     // lowest predictions[i].score present; unset if no prediction has a numeric score
    std::optional<float> maxConfidence;     // highest predictions[i].score present; unset if no prediction has a numeric score
};

// Pure function: parses a Label Studio tasks-list API response (bare
// array or an object with a "tasks" array -- same tolerance as
// selectAllTaskSummaries) into one DatasetTaskSummary per task.
// hasAnnotation/hasPrediction use the same total_annotations/
// total_predictions-with-array-length-fallback convention as
// selectAllTaskSummaries. classNames are collected from every
// annotations[].result[] and predictions[].result[] item's
// value.rectanglelabels/choices/brushlabels array, whichever is present
// -- covers classification, detection, rotated detection (which reuses
// rectanglelabels plus a rotation field, not a separate key), and masks.
// minConfidence/maxConfidence come from every predictions[].score that's
// present and numeric. Tasks missing `id`, or missing `data[dataImageKey]`
// as a string, are skipped, matching selectAllTaskSummaries/
// selectUnlabeledTasks's existing convention.
std::vector<DatasetTaskSummary> summarizeDatasetTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey);

enum class DatasetPresenceFilter { Any, Has, Lacks };

enum class DatasetConfidenceFilterMode { None, LessThan, GreaterThan };

struct DatasetFilterSpec {
    DatasetPresenceFilter annotationFilter = DatasetPresenceFilter::Any;
    DatasetPresenceFilter predictionFilter = DatasetPresenceFilter::Any;
    std::string classNameFilter;   // empty = no class-name filtering; otherwise an exact match against any of classNames
    DatasetConfidenceFilterMode confidenceFilterMode = DatasetConfidenceFilterMode::None;
    float confidenceThreshold = 0.5f;   // compared against maxConfidence (GreaterThan) or minConfidence (LessThan)
};

// Pure function: returns the taskId of every summary matching every
// active filter dimension in `filter` (dimensions left at their default
// -- Any / empty / None -- are ignored). LessThan/GreaterThan each need
// at least one prediction with a numeric score to evaluate: LessThan
// passes if the summary's *lowest* confidence is below the threshold
// (i.e. at least one prediction is that low); GreaterThan passes if its
// *highest* confidence is above it. A summary with no scored prediction
// at all never passes an active confidence filter, regardless of
// threshold.
std::vector<int> filterDatasetTasks(const std::vector<DatasetTaskSummary>& summaries, const DatasetFilterSpec& filter);

// Pure function: returns the subset of `allTasksRaw` (the same raw
// tasks-list JSON fetchAllLabelStudioTasksRaw produces) whose task ids
// are in `matchingTaskIds`, as a bare JSON array of unmodified task
// objects, in `allTasksRaw`'s original order. This is the Dataset
// Browser's entire export file: since parseLabelStudioExport/
// loadLabelStudioExport already treat this exact shape as a Label
// Studio export, no new serialization format is needed. Tasks in
// `matchingTaskIds` not found in `allTasksRaw` are silently skipped
// (defensive; shouldn't happen since both come from the same fetch).
nlohmann::json buildDatasetExportJson(const nlohmann::json& allTasksRaw, const std::vector<int>& matchingTaskIds);

// Kept separate (rather than one merged list) so callers can pick just
// one -- the Dataset Browser only ever bakes annotations *or*
// predictions into a thumbnail at a time (DatasetOverlayMode), never
// both, since it colors boxes by class name rather than by source and
// mixing both lists together would make that ambiguous to read.
struct DatasetBoxesToDraw {
    std::vector<DraftDetectionBox> annotationBoxes;
    std::vector<DraftDetectionBox> predictionBoxes;
};

// Pure function: finds `taskId` in `allTasksRaw` (the same raw shape
// fetchAllLabelStudioTasksRaw produces) and returns its annotation boxes
// and prediction boxes separately, via parseDetectionResultBoxes with
// `rectangleLabelsFromName` -- the same result-parsing this app's own
// Labeling/Label Assistant windows already use, so this stays correct
// for rotated boxes (parseDetectionResultBoxes handles the `rotation`
// field the same way regardless of caller) without duplicating that
// logic. Both lists are empty if `rectangleLabelsFromName` is empty (the
// project has no RectangleLabels control tag -- e.g. classification-only
// -- so there's nothing box-shaped to draw) or the task isn't found.
DatasetBoxesToDraw boxesToDrawForTask(
    const nlohmann::json& allTasksRaw, const std::string& rectangleLabelsFromName, int taskId);

// Same split as DatasetBoxesToDraw, for brush masks.
struct DatasetMasksToDraw {
    std::vector<DraftBrushRegion> annotationMasks;
    std::vector<DraftBrushRegion> predictionMasks;
};

// Pure function: finds `taskId` in `allTasksRaw` and returns its
// annotation masks and prediction masks separately, via
// parseBrushResultRegions with `brushLabelsFromName` -- mirrors
// boxesToDrawForTask exactly, one result type (BrushLabels) instead of
// another (RectangleLabels). Each DraftBrushRegion's mask is CV_8UC1 at
// the image's original resolution (from that result item's own
// original_width/original_height, same per-item convention
// parseDetectionResultBoxes uses); the caller resizes it to whatever
// target size it's compositing onto. Both lists are empty if
// `brushLabelsFromName` is empty (the project has no BrushLabels control
// tag) or the task isn't found.
DatasetMasksToDraw masksToDrawForTask(
    const nlohmann::json& allTasksRaw, const std::string& brushLabelsFromName, int taskId);

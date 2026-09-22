#pragma once

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

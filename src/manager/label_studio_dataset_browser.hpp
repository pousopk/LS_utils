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

#pragma once

#include "manager/item_filters.hpp"
#include "manager/label_studio_import.hpp"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    std::string createdAt;                  // task.created_at verbatim (ISO-8601 UTC); empty if missing or not a string
    std::optional<std::time_t> createdAtEpoch;   // createdAt parsed (parseIso8601Utc); unset if empty or unparseable
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

struct DatasetFilterSpec {
    PresenceFilter annotation;
    PresenceFilter prediction;
    ClassFilter cls;
    // LessThan tests a task's lowest prediction score, GreaterThan its
    // highest (see filterDatasetTasks).
    ConfidenceFilter confidence;
    TimeWindowFilter time;   // on createdAtEpoch
};

// Pure function: returns the taskId of every summary passing every
// filter in `filter` (AND). Confidence: LessThan passes if the task's
// *lowest* prediction score is below the threshold (at least one
// prediction is that low); GreaterThan if its *highest* is above; a task
// with no scored prediction never passes an active confidence filter.
// Time: tested on createdAtEpoch; a task without one never passes an
// active window; an inactive window (Off, or input that doesn't resolve)
// passes everything. Result order: AroundTime sorts closest to the
// center first (ties by task id); otherwise `summaries` order.
std::vector<int> filterDatasetTasks(const std::vector<DatasetTaskSummary>& summaries, const DatasetFilterSpec& filter);

// Pure function: every class name across `summaries`, sorted and
// de-duplicated -- the Dataset Browser's class filter options.
std::vector<std::string> collectDatasetClassNames(const std::vector<DatasetTaskSummary>& summaries);

// How many tasks were created on each local calendar day (from
// createdAtEpoch; tasks without one are skipped) -- the days the Dataset
// Browser's time-filter calendars mark. Not pure: the local day depends
// on the process's timezone setting.
std::map<CalendarDate, int> countTasksPerLocalDay(const std::vector<DatasetTaskSummary>& summaries);

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

// Pure function (apart from writing to `out`): the streaming counterpart
// to buildDatasetExportJson, for building export.json one fetched page
// at a time instead of from the whole project held in memory. Writes
// every task in `pageTasks` (bare array or an object wrapping "tasks",
// same tolerance as buildDatasetExportJson) whose id is still in
// `remainingTaskIds` to `out` as an unmodified JSON array element, and
// erases that id -- so a task re-delivered on a later page (offset
// pagination shifting while tasks are added during a long export) is
// written only once --
// dump(2), preceded by ",\n" unless it's the first element written, as
// tracked across calls by `wroteAnyElement`. The caller writes "[\n"
// before the first page and "\n]\n" after the last, so tasks keep
// their fetch order and the result is the same shape
// parseLabelStudioExport/loadLabelStudioExport already read. Returns how
// many tasks were written by this call.
size_t writeMatchingTasksAsJsonArrayElements(
    const nlohmann::json& pageTasks, std::unordered_set<int>& remainingTaskIds, std::ostream& out,
    bool& wroteAnyElement);

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

// Pure function: one pass over `allTasksRaw`, building a taskId ->
// DatasetBoxesToDraw map for every task -- same extraction as
// boxesToDrawForTask, but for every task at once instead of searching
// for one task per call. Exists so repeated per-task lookups (e.g. once
// per newly-visible grid cell while scrolling) are O(1) map lookups
// against a result built once, rather than each independently
// re-scanning the whole task list (which made scrolling visibly slow
// once a project had more than a couple hundred tasks). Returns an
// empty map immediately if `rectangleLabelsFromName` is empty, matching
// boxesToDrawForTask's "nothing to draw" convention.
std::unordered_map<int, DatasetBoxesToDraw> buildBoxesByTaskId(
    const nlohmann::json& allTasksRaw, const std::string& rectangleLabelsFromName);

// Pure function: one pass over `tasksJson` (bare array or {"tasks": [...]}),
// building a taskId -> ImageGroundTruth map from each task's first
// annotation via parseLabelStudioExport -- the same parser used for a
// manually-exported ground-truth file, so Benchmark's Label Studio source
// sees exactly the ground truth a file export would give it. Tasks without
// an integer `id`, or that parseLabelStudioExport skips (no annotation, no
// image field), get no entry.
std::unordered_map<int, ImageGroundTruth> buildGroundTruthByTaskId(const nlohmann::json& tasksJson);

// Same one-pass-instead-of-per-lookup idea as buildBoxesByTaskId, for masksToDrawForTask.
std::unordered_map<int, DatasetMasksToDraw> buildMasksByTaskId(
    const nlohmann::json& allTasksRaw, const std::string& brushLabelsFromName);

// One brush mask kept in Label Studio's own RLE encoding rather than
// decoded. A decoded mask is a full-resolution CV_8UC1 (~2MB at 1080p),
// so decoding every task's masks up front -- what buildMasksByTaskId
// does -- grows without bound with project size; the encoded RLE is
// typically a few KB, and decodeDatasetMasks turns it into pixels only
// for the thumbnails actually being built.
struct DatasetEncodedMask {
    std::vector<uint8_t> rle;   // Label Studio brush RLE bytes, as in value.rle
    int width = 0;              // original_width
    int height = 0;             // original_height
    std::string className;      // value.brushlabels[0]
};

// Same annotation/prediction split as DatasetMasksToDraw, still encoded.
struct DatasetEncodedMasks {
    std::vector<DatasetEncodedMask> annotationMasks;
    std::vector<DatasetEncodedMask> predictionMasks;
};

// Pure function: one pass over `tasksJson` (a full task list or a single
// page of one -- same response-shape tolerance as buildMasksByTaskId),
// collecting each task's brush masks without decoding them. Accepts
// exactly the result items parseBrushResultRegions would, except that an
// item whose rle holds a value outside 0..255 is skipped instead of
// decoded into garbage. Unlike buildMasksByTaskId, only tasks that
// actually have at least one mask get an entry -- callers already treat
// a missing entry as "nothing to draw". Returns an empty map if
// `brushLabelsFromName` is empty.
std::unordered_map<int, DatasetEncodedMasks> buildEncodedMasksByTaskId(
    const nlohmann::json& tasksJson, const std::string& brushLabelsFromName);

// Pure function: decodes each of `masks` via decodeLabelStudioRleToMask,
// in order -- the lazy counterpart to buildEncodedMasksByTaskId.
std::vector<DraftBrushRegion> decodeDatasetMasks(const std::vector<DatasetEncodedMask>& masks);

// Pure function: builds a taskId -> index-into-`summaries` lookup, so
// repeated per-task summary lookups don't need a linear scan over the
// whole list (the same "std::find_if over the whole vector per grid
// cell" problem buildBoxesByTaskId/buildMasksByTaskId address, just for
// DatasetTaskSummary instead of raw JSON). The returned indices are only
// valid against the exact `summaries` vector passed in -- rebuild this
// alongside any reassignment of that vector, not just once.
std::unordered_map<int, size_t> indexSummariesByTaskId(const std::vector<DatasetTaskSummary>& summaries);

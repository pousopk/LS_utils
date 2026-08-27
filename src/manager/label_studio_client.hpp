#pragma once

#include "manager/label_studio_import.hpp"

#include <atomic>
#include <functional>
#include <map>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <vector>

// Pure function: extracts `attributeName`'s value from the first XML tag
// matching `tagName` found in `xml` (e.g. tagName="Image",
// attributeName="name" matches `<Image name="..." .../>`, regardless of
// attribute order). Returns an empty string if no matching tag, or no such
// attribute on it, is found. A simple attribute-syntax extractor, not a
// general XML parser -- matches Label Studio's predictable
// labeling-config tag format.
std::string extractXmlTagAttribute(const std::string& xml, const std::string& tagName, const std::string& attributeName);

// Pure function: extractXmlTagAttribute(xml, tagName, "name").
std::string extractXmlTagNameAttribute(const std::string& xml, const std::string& tagName);

struct LabelStudioLabelingConfig {
    std::string fromName;
    std::string toName;
    std::string dataImageKey;   // the key in task.data holding the image path, e.g. "image"
    std::string error;          // empty on success
};

// Fetches the project's labeling config via GET
// {baseUrl}/api/projects/{projectId}/ (its `label_config` field, a raw
// XML string) and extracts from_name/to_name for the given task type: the
// `name` attribute of the first <RectangleLabels> tag (isDetection=true)
// or <Choices> tag (isDetection=false), plus the `name` attribute of the
// <Image> tag. Also extracts dataImageKey from the <Image> tag's `value`
// attribute (its `$`-prefix stripped) -- the key under task.data that
// holds the image path, needed to know what to download in "Label Studio
// Project" source mode. Best-effort -- if the config has more than one
// matching tag, the first one wins. On any failure (network, non-2xx, tag
// not found), returns with `error` set and the other fields left empty.
LabelStudioLabelingConfig fetchLabelStudioLabelingConfig(
    const std::string& baseUrl, int projectId, const std::string& apiToken, bool isDetection);

// Pure function: parses a Label Studio tasks-list API response --
// handling both a bare JSON array and an object with a "tasks" array,
// since this varies across Label Studio versions -- and returns the
// `count` highest task ids present, sorted ascending (oldest of the
// selected group first, i.e. chronological creation order). Used to
// identify "the tasks this push just created": since this app uploads
// images one at a time, sequentially, with nothing else able to create
// tasks in the project concurrently during that loop, the tasks it just
// created are exactly the highest-numbered `count` task ids afterward, in
// upload order. Returns fewer than `count` entries if the project has
// fewer than `count` tasks total, or an empty vector if the response is
// unparseable/malformed.
std::vector<int> selectMostRecentTaskIds(const nlohmann::json& tasksJson, size_t count);

struct LabelStudioPredictionInput {
    std::string imageFilename;                // basename, used only for status/logging
    std::string localImagePath;               // full local filesystem path to upload
    PredictionResultAndScore resultAndScore;   // from buildClassificationPredictionResult/buildDetectionPredictionResult
};

struct LabelStudioPushSummary {
    int created = 0;            // new tasks created with a prediction successfully attached
    int uploadFailed = 0;       // the image upload (multipart POST) itself failed
    int taskNotResolved = 0;    // uploaded successfully, but its resulting task id couldn't be found afterward
    int predictionFailed = 0;   // task id resolved, but attaching the prediction to it failed
    std::string error;          // set only on a hard failure (couldn't fetch the task list after uploading at all)
};

// For each of `predictions`: uploads its image (a real multipart file
// upload of localImagePath's bytes, not a path reference) to the project
// via Label Studio's import endpoint, creating a brand-new task -- this
// app never tries to reuse or match an existing task, so re-running this
// on the same folder creates duplicate tasks each time. After all uploads
// complete, fetches the project's tasks once and resolves the resulting
// task ids via selectMostRecentTaskIds (Label Studio's import response
// doesn't reliably return task ids directly, and the per-upload id it
// does return -- file_upload_ids -- turns out not to correspond to the
// task list's own `file_upload` field, which is a filename string, not
// that id; there is no confirmed API for translating between the two, so
// this app relies on upload+task-creation being sequential and
// exclusively its own instead). For each resolved task, attaches the
// prediction via Label Studio's Predictions API. If fewer task ids come
// back than uploads succeeded, every uploaded prediction in this push is
// counted as unresolved (rather than guessing a partial mapping). Only a
// failure to fetch the task list after uploading sets `error` and aborts
// the operation -- individual upload/attach failures are folded into the
// matching summary counters instead, matching this app's existing
// "surface a summary, don't fail the whole run over one bad item"
// convention for folder-scan operations.
LabelStudioPushSummary pushDraftsAsNewLabelStudioTasks(
    const std::string& baseUrl, int projectId, const std::string& apiToken,
    const std::vector<LabelStudioPredictionInput>& predictions);

struct LabelStudioUnlabeledTask {
    int taskId = 0;
    std::string imagePath;   // task.data[dataImageKey], e.g. "/data/upload/11/xxx.png"
};

// Pure function: parses a Label Studio tasks-list API response (bare
// array or an object with a "tasks" array) into the subset of tasks that
// have no existing predictions or annotations -- read from
// total_predictions/total_annotations when present, falling back to the
// length of the predictions/annotations arrays otherwise. Tasks missing
// `id`, or missing `data[dataImageKey]` as a string, are skipped.
std::vector<LabelStudioUnlabeledTask> selectUnlabeledTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey);

// Pure function: parses the leading numeric filename stem (before the
// extension) of `filename` as a task id, e.g. "123.png" -> 123. Returns
// std::nullopt if the stem is empty or contains anything but digits. Used
// to recover the task id this app itself encoded into a downloaded
// image's local filename (see fetchAndDownloadUnlabeledTasks) -- not a
// general-purpose parser.
std::optional<int> parseTaskIdFromFilename(const std::string& filename);

struct LabelStudioDownloadResult {
    int downloaded = 0;      // images successfully downloaded
    int downloadFailed = 0;  // an unlabeled task's image failed to download
    std::string error;       // set only on a hard failure (couldn't fetch the task list at all)
};

// Fetches the project's tasks, selects the unlabeled ones (via
// selectUnlabeledTasks), and downloads each one's image (an authenticated
// GET to `{baseUrl}{imagePath}`, same auth as every other call here) into
// `outputFolder` as `<taskId><original extension>` -- the task id is
// encoded directly in the filename so no separate id-to-file mapping
// needs to be tracked; `parseTaskIdFromFilename` recovers it later.
// `outputFolder` is not created or cleared by this function -- the caller
// (this window's "Label Studio Project" source mode) is responsible for
// giving it a clean, already-existing directory. `onProgress` (if
// non-null) is called once per download attempt, with `total` fixed at
// the unlabeled-task count. `cancelRequested` (if non-null and observed
// true) stops the loop early, returning whatever completed so far with
// `error` left empty. Individual download failures are counted in
// `downloadFailed`, not treated as fatal; only a failure to fetch the
// task list itself sets `error`.
LabelStudioDownloadResult fetchAndDownloadUnlabeledTasks(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& dataImageKey,
    const std::string& outputFolder, const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

struct LabelStudioKnownTaskPrediction {
    int taskId = 0;
    PredictionResultAndScore resultAndScore;
};

struct LabelStudioAttachSummary {
    int created = 0;   // predictions successfully attached
    int failed = 0;    // the create-prediction request itself failed
};

// Attaches each of `predictions` directly to its already-known task id
// via Label Studio's Predictions API -- no upload, no task resolution;
// used by "Label Studio Project" source mode, where the task id was
// already known from the download step. Every attempt is made regardless
// of earlier failures; `failed` counts individual create failures.
LabelStudioAttachSummary attachPredictionsToKnownTasks(
    const std::string& baseUrl, const std::string& apiToken,
    const std::vector<LabelStudioKnownTaskPrediction>& predictions);

struct LabelStudioLabeledTask {
    int taskId = 0;
    std::string imagePath;   // task.data[dataImageKey], e.g. "/data/upload/11/xxx.png"
};

// Pure function: parses a Label Studio tasks-list API response (bare
// array or an object with a "tasks" array) into the subset of tasks that
// have at least one existing annotation -- read from total_annotations
// when present, falling back to the length of the annotations array
// otherwise. Predictions are ignored entirely (a task can have a
// prediction from a prior push and still count as unlabeled here).
// Tasks missing `id`, or missing `data[dataImageKey]` as a string, are
// skipped.
std::vector<LabelStudioLabeledTask> selectLabeledTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey);

struct LabelStudioGroundTruthDataset {
    LabelStudioImportResult groundTruth;   // built by parseLabelStudioExport on the raw task list, unchanged
    int downloaded = 0;
    int downloadFailed = 0;
    std::string error;   // set only on a hard failure (couldn't fetch the task list at all)
};

// Fetches the project's tasks once, and from that single response: builds
// ground truth via parseLabelStudioExport (the exact same parser already
// used for a manually-exported file -- a task's `data`+`annotations`
// shape from this API matches an export's shape exactly, confirmed
// against a real server), and downloads each labeled task's image (via
// selectLabeledTasks) into `outputFolder`, named by its own original
// basename (not `<taskId>.ext` -- the ground truth parseLabelStudioExport
// built also references that same basename, so this app's existing
// filename-based ground-truth-to-local-file matching, used everywhere
// else in Batch mode, works unmodified). `outputFolder` is not created or
// cleared by this function -- the caller is responsible for giving it a
// clean, already-existing directory. `onProgress`/`cancelRequested`
// behave exactly as fetchAndDownloadUnlabeledTasks's. Individual download
// failures are counted in `downloadFailed`, not fatal; only a failure to
// fetch the task list itself sets `error`.
LabelStudioGroundTruthDataset fetchAndDownloadLabeledDataset(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& dataImageKey,
    const std::string& outputFolder, const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

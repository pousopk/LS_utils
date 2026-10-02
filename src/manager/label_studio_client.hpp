#pragma once

#include "manager/label_studio_dataset_browser.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/time_parse.hpp"

#include <atomic>
#include <ctime>
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

struct TimestampMatchQuery {
    std::time_t timestamp = 0;        // epoch seconds, e.g. from parseTypedLocalTimestamp
    long long toleranceSeconds = 0;   // symmetric window: [timestamp - toleranceSeconds, timestamp + toleranceSeconds]
};

struct TimestampMatchCandidate {
    int taskId = 0;
    std::string imagePath;      // task.data[dataImageKey]
    std::time_t createdAt = 0;  // epoch seconds
    long long deltaSeconds = 0; // createdAt - query.timestamp (negative if the task predates the query)
};

// Pure function: for each of `queries` (result vector is the same length
// and order), returns every task in `tasksJson` whose `created_at` falls
// within [timestamp - toleranceSeconds, timestamp + toleranceSeconds],
// sorted by abs(deltaSeconds) ascending (closest first). Tasks missing
// `id`, missing/unparseable `created_at`, or missing `data[dataImageKey]`
// as a string are skipped -- same tolerance for partial task records as
// selectUnlabeledTasks. A task can appear under more
// than one query if the windows overlap.
std::vector<std::vector<TimestampMatchCandidate>> matchTasksToTimestamps(
    const nlohmann::json& tasksJson, const std::string& dataImageKey, const std::vector<TimestampMatchQuery>& queries);

struct LabelStudioTaskImageDownload {
    int downloaded = 0;
    int downloadFailed = 0;
};

// Downloads each of `candidates`' images into `outputFolder` as
// `<taskId><original extension>`, same convention as
// fetchAndDownloadUnlabeledTasks. Candidates are not deduplicated by this
// function -- pass an already-deduplicated list if the same task appears
// under more than one timestamp query. `outputFolder` is not created or
// cleared by this function. `onProgress`/`cancelRequested` behave exactly
// as fetchAndDownloadUnlabeledTasks's.
LabelStudioTaskImageDownload downloadLabelStudioTaskImages(
    const std::string& baseUrl, const std::string& apiToken, const std::vector<TimestampMatchCandidate>& candidates,
    const std::string& outputFolder, const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

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

enum class LabelStudioControlTagType {
    RectangleLabels,
    Choices,
    BrushLabels,
};

struct LabelStudioControlTag {
    LabelStudioControlTagType type;
    std::string name;                     // from_name
    std::string toName;                   // to_name
    std::vector<std::string> labels;      // child <Label value="..."/> or <Choice value="..."/>, in config order
};

struct LabelStudioProjectConfig {
    std::string dataImageKey;                        // from <Image value="$..."/>, leading '$' stripped
    std::vector<LabelStudioControlTag> controlTags;   // every RectangleLabels/Choices tag found, in document order
    std::string error;                                // empty on success
};

// Pure function: parses a Label Studio project's raw label_config XML into
// every RectangleLabels/Choices control tag it contains (BrushLabels is not
// parsed yet -- phase 3), each with its own from_name/to_name and full
// label/choice value list -- unlike fetchLabelStudioLabelingConfig, which
// only looks at the first matching tag of one type. Uses pugixml (a real
// parser) rather than extractXmlTagAttribute's regex approach, since this
// needs to walk repeated/nested tags reliably. Returns with `error` set if
// the XML doesn't parse, or no <Image> tag is found.
LabelStudioProjectConfig parseLabelStudioProjectConfigXml(const std::string& labelConfigXml);

// Fetches the project's labeling config via GET
// {baseUrl}/api/projects/{projectId}/ (same endpoint as
// fetchLabelStudioLabelingConfig) and parses it with
// parseLabelStudioProjectConfigXml.
LabelStudioProjectConfig fetchLabelStudioProjectConfigDetailed(
    const std::string& baseUrl, int projectId, const std::string& apiToken);

struct LabelStudioProjectSummary {
    int id = 0;
    std::string title;
};

// Pure function: parses a Label Studio projects-list API response --
// handling both a bare JSON array and an object with a "results" array
// (Label Studio's DRF-paginated shape, same tolerance this file already
// applies to the tasks-list response) -- into project summaries. Entries
// missing `id` (as an integer) or `title` (as a string) are skipped.
std::vector<LabelStudioProjectSummary> parseLabelStudioProjects(const nlohmann::json& projectsJson);

struct LabelStudioProjectListResult {
    std::vector<LabelStudioProjectSummary> projects;
    std::string error;   // set only on a hard failure (network, non-2xx, unparseable body)
};

// Pure function: decides whether a paginated Label Studio list fetch
// (tasks, projects, ...) should request another page, given the page
// just parsed and how many items it contained. Two response shapes are
// in play, since this varies by endpoint (and Label Studio version):
// - DRF-style pagination (e.g. /api/projects/): the response object
//   always has a "next" key, holding either a URL string (more pages
//   exist) or null (this was the last page). When present, this key is
//   authoritative regardless of page size.
// - Label Studio's native tasks-list shape (/api/tasks/): there is no
//   "next" key at all, so the only signal is page size -- a full page
//   (pageItemCount == pageSize) might not be the last one; a short or
//   empty page (pageItemCount < pageSize) definitely is.
bool shouldFetchNextLabelStudioPage(const nlohmann::json& parsedPage, size_t pageItemCount, int pageSize);

// Fetches every project visible to this API token via GET
// {baseUrl}/api/projects/, paginating exactly like
// fetchAllLabelStudioTasksRaw (page/page_size params, following `next`
// until exhausted or a short page is seen, via
// shouldFetchNextLabelStudioPage), then parses the accumulated raw array
// with parseLabelStudioProjects.
LabelStudioProjectListResult fetchLabelStudioProjects(const std::string& baseUrl, const std::string& apiToken);

// Pure function: parses a Label Studio tasks-list API response --
// handling both a bare JSON array and an object with a "tasks" array,
// since this varies across Label Studio versions -- and returns the
// `count` highest task ids present, sorted ascending (oldest of the
// selected group first, i.e. chronological creation order). Used to
// identify "the tasks this push just created": since this app uploads
// images one at a time, sequentially, this assumes nothing else creates
// tasks in the project concurrently during that loop, so the tasks it
// just created are exactly the highest-numbered `count` task ids
// afterward, in upload order. pushDraftsAsNewLabelStudioTasks checks
// that assumption via countLabelStudioTasks before trusting this.
// Returns fewer than `count` entries if the project has fewer than
// `count` tasks total, or an empty vector if the response is
// unparseable/malformed.
std::vector<int> selectMostRecentTaskIds(const nlohmann::json& tasksJson, size_t count);

// Pure function: counts the tasks present in a Label Studio tasks-list
// API response -- same response-shape tolerance as
// selectMostRecentTaskIds (bare array, or an object wrapping a "tasks"
// array). Unlike selectAllTaskSummaries, this doesn't need
// a dataImageKey and doesn't skip anything; it exists purely so
// pushDraftsAsNewLabelStudioTasks can detect whether the project's task
// count changed by more than its own uploads accounted for (i.e.
// something else created or removed tasks concurrently), in which case
// selectMostRecentTaskIds' "top N ids are ours" assumption no longer
// holds. Returns 0 if the response is unparseable/malformed.
size_t countLabelStudioTasks(const nlohmann::json& tasksJson);

struct LabelStudioPredictionInput {
    std::string imageFilename;                // basename, used only for status/logging
    std::string localImagePath;               // full local filesystem path to upload
    PredictionResultAndScore resultAndScore;   // from buildClassificationPredictionResult/buildDetectionPredictionResult
};

struct LabelStudioPushSummary {
    int created = 0;            // new tasks created with a prediction successfully attached
    int uploadFailed = 0;       // the image upload (multipart POST) itself failed
    int taskNotResolved = 0;    // uploaded successfully, but its resulting task id couldn't be found afterward,
                                 // or couldn't be trusted (the project's task count changed by more than this
                                 // push accounted for -- see pushDraftsAsNewLabelStudioTasks)
    int predictionFailed = 0;   // task id resolved, but attaching the prediction to it failed
    std::string error;          // set only on a hard failure (couldn't fetch the task list after uploading at all)
};

// Fetches the project's task count before doing anything else (see
// countLabelStudioTasks), then for each of `predictions`: uploads its
// image (a real multipart file upload of localImagePath's bytes, not a
// path reference) to the project via Label Studio's import endpoint,
// creating a brand-new task -- this app never tries to reuse or match an
// existing task, so re-running this on the same folder creates duplicate
// tasks each time. After all uploads complete, fetches the project's
// tasks once more and resolves the resulting task ids via
// selectMostRecentTaskIds (Label Studio's import response doesn't
// reliably return task ids directly, and the per-upload id it does
// return -- file_upload_ids -- turns out not to correspond to the task
// list's own `file_upload` field, which is a filename string, not that
// id; there is no confirmed API for translating between the two, so this
// app relies on upload+task-creation being sequential and exclusively
// its own instead). Before trusting that resolution, checks that the
// task count grew by exactly `uploaded.size()`; if it didn't (something
// else created or removed tasks in this project while the loop ran),
// every uploaded prediction in this push is counted as unresolved rather
// than risking attaching a prediction to a task this app didn't create.
// The same "don't guess a partial mapping" handling applies if fewer
// task ids come back than uploads succeeded. For each task id that is
// resolved and trusted, attaches the prediction via Label Studio's
// Predictions API. Only a failure to fetch the task list (before
// uploading, or after) sets `error` and aborts the operation --
// individual upload/attach failures are folded into the matching summary
// counters instead, matching this app's existing "surface a summary,
// don't fail the whole run over one bad item" convention for
// folder-scan operations. `onProgress` is called once per completed
// upload, then once per completed attach attempt, as a single running
// count against a fixed `total` of `predictions.size() * 2` (upload
// attempts + an upper bound on attach attempts, since at most
// `predictions.size()` uploads can succeed) -- fewer than `total`
// completions is normal whenever some uploads fail, since those don't
// get an attach attempt. `cancelRequested` is checked between items in
// both loops, same as fetchAndDownloadUnlabeledTasks's.
LabelStudioPushSummary pushDraftsAsNewLabelStudioTasks(
    const std::string& baseUrl, int projectId, const std::string& apiToken,
    const std::vector<LabelStudioPredictionInput>& predictions,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

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

// Pure function: the summary-based equivalent of selectUnlabeledTasks --
// every summary with neither an annotation nor a prediction. Matches it
// exactly, since summarizeDatasetTasks derives hasAnnotation/
// hasPrediction with the same totals-with-array-length-fallback rule and
// skips the same tasks (missing `id` or image key). Exists so the
// shared task list doesn't have to keep every task's raw JSON around.
std::vector<LabelStudioUnlabeledTask> selectUnlabeledFromSummaries(const std::vector<DatasetTaskSummary>& summaries);

// Pure function: the local filename a task's image is downloaded as --
// `<taskId><original extension>`, e.g. (123, "/data/upload/1/x.png") ->
// "123.png". Encoding the task id lets parseTaskIdFromFilename recover it.
std::string taskImageLocalFilename(int taskId, const std::string& imagePath);

// Pure function: parses the leading numeric filename stem (before the
// extension) of `filename` as a task id, e.g. "123.png" -> 123. Returns
// std::nullopt if the stem is empty or contains anything but digits. Used
// to recover the task id this app itself encoded into a downloaded
// image's local filename (see fetchAndDownloadUnlabeledTasks) -- not a
// general-purpose parser.
std::optional<int> parseTaskIdFromFilename(const std::string& filename);

struct LabelStudioTaskImagesDownloadResult {
    int downloaded = 0;      // images successfully downloaded
    int downloadFailed = 0;  // a task's image failed to download
};

// Downloads each of `tasks`' images (an authenticated GET to
// `{baseUrl}{imagePath}`, same auth as every other call here) into
// `outputFolder` as `<taskId><original extension>` -- the task id is
// encoded directly in the filename so no separate id-to-file mapping
// needs to be tracked; `parseTaskIdFromFilename` recovers it later.
// `outputFolder` is not created or cleared by this function -- the caller
// is responsible for giving it a clean, already-existing directory.
// `onProgress` (if non-null) is called once per download attempt, with
// `total` fixed at `tasks.size()`. `cancelRequested` (if non-null and
// observed true) stops the loop early, returning whatever completed so
// far. Individual download failures are counted in `downloadFailed`, not
// fatal -- unlike the fused fetchAndDownloadUnlabeledTasks this replaces,
// there's no longer a hard-failure mode here at all, since selecting
// which tasks to download (the only step that used to fail hard, by
// failing to fetch the task list) now happens separately, before this is
// ever called (see selectUnlabeledTasks, called from
// startLabelAssistantRun against the shared project data).
LabelStudioTaskImagesDownloadResult downloadUnlabeledTaskImages(
    const std::string& baseUrl, const std::string& apiToken, const std::vector<LabelStudioUnlabeledTask>& tasks,
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
// `onProgress`/`cancelRequested` behave exactly as
// fetchAndDownloadUnlabeledTasks's -- checked between predictions, so a
// cancel mid-batch stops before attaching the rest (already-attached ones
// are not undone).
LabelStudioAttachSummary attachPredictionsToKnownTasks(
    const std::string& baseUrl, const std::string& apiToken,
    const std::vector<LabelStudioKnownTaskPrediction>& predictions,
    const std::function<void(int completed, int total)>& onProgress = nullptr,
    const std::atomic<bool>* cancelRequested = nullptr);

struct LabelStudioTaskSummary {
    int taskId = 0;
    std::string imagePath;    // task.data[dataImageKey]
    bool hasAnnotation = false;
    bool hasPrediction = false;
};

// Pure function: parses a Label Studio tasks-list API response (bare array
// or an object with a "tasks" array) into a summary of every task --
// unlike selectUnlabeledTasks, this keeps every task
// regardless of label state, flagging each one's annotation/prediction
// presence instead of filtering. Reads total_annotations/total_predictions
// when present, falling back to the annotations/predictions array lengths
// otherwise (same convention as selectUnlabeledTasks).
// Tasks missing `id`, or missing `data[dataImageKey]` as a string, are
// skipped.
std::vector<LabelStudioTaskSummary> selectAllTaskSummaries(
    const nlohmann::json& tasksJson, const std::string& dataImageKey);

// Pure function: formats `t` as Label Studio's Datetime filter value,
// ISO-8601 UTC with milliseconds, e.g. "2026-08-31T22:00:00.000Z".
std::string formatIso8601Utc(std::time_t t);

// Pure function: RFC 3986 percent-encoding for a URL query value --
// everything except unreserved characters (A-Z a-z 0-9 - _ . ~) becomes
// %XX. Used for the JSON `query` parameter below.
std::string percentEncodeQueryValue(const std::string& value);

// A half-open window on a task's `created_at` (its import time): tasks
// with fromInclusive <= created_at < toExclusive. Either side unset means
// no limit on that side; both unset means the whole project.
struct TaskCreatedAtBounds {
    std::optional<std::time_t> fromInclusive;
    std::optional<std::time_t> toExclusive;

    bool unbounded() const { return !fromInclusive && !toExclusive; }
};

// Pure function: the JSON string for Label Studio's tasks-list `query`
// parameter that asks the server to filter on created_at --
// {"filters":{"conjunction":"and","items":[{"filter":
// "filter:tasks:created_at","operator":"greater_or_equal"|"less",
// "type":"Datetime","value":<formatIso8601Utc>}...]}}. Empty string if
// `bounds` is unbounded (no query parameter is sent at all).
std::string buildLabelStudioCreatedAtQuery(const TaskCreatedAtBounds& bounds);

// Pure function: whether a task's raw `created_at` string falls within
// `bounds`. Always true when unbounded; otherwise false for an empty or
// unparseable value, since it can't be shown to be in range. The local
// check that keeps a date-filtered load correct even on a Label Studio
// version that ignores the `query` filter.
bool isTaskCreatedWithin(const std::string& createdAt, const TaskCreatedAtBounds& bounds);

// One parsed page of Label Studio's tasks-list API response.
struct LabelStudioTaskPage {
    const nlohmann::json* tasks = nullptr;   // points into the parsed page; nullptr if no task array found
    std::optional<int> total;                // the response's integer "total" field, if present
};

// Pure function: locates the task array in one parsed tasks-list page --
// a bare array, or an object with a "tasks" or "results" array (Label
// Studio's exact response shape isn't pinned to one version) -- plus
// Label Studio's native "total" task count when the response carries
// one. `tasks` points into `parsedPage`, so it's only valid while
// `parsedPage` is.
LabelStudioTaskPage parseLabelStudioTaskPage(const nlohmann::json& parsedPage);

// Pure function: whether a failed page request is worth retrying -- a
// request-level failure (timeout, connection reset), HTTP 429, or any
// 5xx. Anything else (401, 404, ...) won't fix itself on a retry.
bool isRetryableLabelStudioFailure(bool requestFailed, long httpCode);

struct LabelStudioTaskPageProgress {
    int fetchedTasks = 0;              // tasks delivered so far, including the current page
    std::optional<int> totalTasks;     // from the response's "total", if Label Studio reported one
};

// Called once per fetched page. `pageTasks` is only valid for the
// duration of the call. Return false to stop fetching (the fetch then
// reports Cancelled).
using LabelStudioTaskPageCallback =
    std::function<bool(const nlohmann::json& pageTasks, const LabelStudioTaskPageProgress& progress)>;

enum class LabelStudioPagedFetchOutcome { Completed, Cancelled, Failed };

enum class LabelStudioTaskPageAction { FetchNext, Done, HitPageCap };

// Pure function: what forEachLabelStudioTaskPage does after delivering
// page number `page` (1-based). Done once Label Studio's reported total
// has been reached -- a project of exactly N*pageSize tasks would
// otherwise request one page past the end, which Label Studio answers
// with 404 -- or when shouldFetchNextLabelStudioPage says this was the
// last page. HitPageCap if another page is still expected but `page` is
// already `maxPages`: reported as a failure rather than silently
// treating a truncated list as complete. FetchNext otherwise.
LabelStudioTaskPageAction nextLabelStudioTaskPageAction(
    const nlohmann::json& parsedPage, size_t pageItemCount, int pageSize, const LabelStudioTaskPageProgress& progress,
    int page, int maxPages);

// Fetches every task in the project (with predictions embedded, via Label
// Studio's `fields=all` query param) one page at a time, handing each
// page to `onPage` and then freeing it -- so, unlike
// fetchAllLabelStudioTasksRaw, the caller decides what (if anything) to
// keep, and a 100k+-task project never has to sit in memory as one JSON
// tree. Decides after each page via nextLabelStudioTaskPageAction: stops
// once Label Studio's reported total is reached (or on a short page), and
// fails -- rather than silently reporting a truncated list as complete --
// if a 1000-page safety cap (200k tasks) is hit with more still expected.
// Each page request gets a 120s timeout and up to 3 attempts (1s, then
// 2s backoff) when isRetryableLabelStudioFailure says so -- one slow or
// flaky page no longer throws away everything fetched before it.
// `cancelRequested` (if non-null) is checked between pages, during
// backoff, and inside curl's transfer callback, so an in-flight request
// aborts within about a second of it being set. On Failed, `error` names
// the page and attempt that failed. A bounded `createdAtBounds` adds
// buildLabelStudioCreatedAtQuery's `query` parameter to every page
// request, so Label Studio only sends tasks imported in that window (and
// its "total" becomes the filtered count); callers still check
// isTaskCreatedWithin themselves, since not every server version honors
// the filter.
LabelStudioPagedFetchOutcome forEachLabelStudioTaskPage(
    const std::string& baseUrl, int projectId, const std::string& apiToken,
    const LabelStudioTaskPageCallback& onPage, const std::atomic<bool>* cancelRequested, std::string& error,
    const TaskCreatedAtBounds& createdAtBounds = {});

// Convenience wrapper over forEachLabelStudioTaskPage that accumulates
// every task into one JSON array (same response-shape tolerance and page
// cap). Holds the whole project in memory, so it's only suitable for
// small projects or flows that genuinely need every raw task -- the
// shared task list the Label Studio tabs use no longer goes through
// this. Still used directly by the push/attach, timestamp search,
// ground truth, and labeled-dataset download flows in this file.
bool fetchAllLabelStudioTasksRaw(
    const std::string& baseUrl, int projectId, const std::string& apiToken, nlohmann::json& allTasks,
    std::string& error);

struct LabelStudioTaskListResult {
    std::vector<LabelStudioTaskSummary> tasks;
    std::string error;   // set only on a hard failure to fetch the task list
};

// Fetches the project's full task list (the same paged fetch every other
// flow here uses) and summarizes it via selectAllTaskSummaries.
LabelStudioTaskListResult fetchLabelStudioTaskSummaries(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& dataImageKey);

struct LabelStudioTaskDetail {
    int taskId = 0;
    std::string imagePath;                                    // task.data[dataImageKey]
    std::optional<int> annotationId;                          // set if the task has at least one annotation
    nlohmann::json annotationResult = nlohmann::json::array(); // annotations[0].result if present, else []
    nlohmann::json predictionResult = nlohmann::json::array(); // predictions[0].result if present, else []
    std::string error;                                        // empty on success
};

// Pure function: extracts one task's id, image path, and (if present) its
// first annotation's id+result and first prediction's result. This app
// works with a single annotation per task (see the design doc) -- if a
// task somehow has more than one, only annotations[0] is used. Returns
// with `error` set if `id` or `data[dataImageKey]` (as a string) is
// missing.
LabelStudioTaskDetail parseLabelStudioTaskDetail(const nlohmann::json& taskJson, const std::string& dataImageKey);

// Fetches a single task via GET {baseUrl}/api/tasks/{taskId}/ (which
// Label Studio returns with its annotations/predictions arrays embedded)
// and parses it with parseLabelStudioTaskDetail.
LabelStudioTaskDetail fetchLabelStudioTaskById(
    const std::string& baseUrl, const std::string& apiToken, int taskId, const std::string& dataImageKey);

// Fetches `imagePath`'s raw bytes via GET {baseUrl}{imagePath}, with the
// same auth as every other call here -- returns the bytes in memory
// instead of writing them to a file. downloadTaskImage (below) is
// implemented in terms of this; the Dataset Browser's thumbnail worker
// uses it directly, decoding straight from memory with no scratch file
// per thumbnail.
bool downloadTaskImageBytes(
    const std::string& baseUrl, const std::string& apiToken, const std::string& imagePath, std::string& outBytes,
    std::string& error);

// Downloads the image at `{baseUrl}{imagePath}` (imagePath already
// absolute, e.g. "/data/upload/11/xxx.png", as found in a task's `data`)
// with the same token auth as every other call here, writing the raw
// bytes to `localOutputPath`. Returns false on any failure (network,
// non-2xx, or the local file couldn't be written), with `error` set.
bool downloadTaskImage(
    const std::string& baseUrl, const std::string& apiToken, const std::string& imagePath,
    const std::string& localOutputPath, std::string& error);

struct LabelStudioAnnotationWriteResult {
    bool success = false;
    std::string error;   // set only when success is false
};

// Creates a brand-new annotation on `taskId` via POST
// {baseUrl}/api/tasks/{taskId}/annotations/, with `resultArray` as its
// `result` field. Used the first time a task is labeled in-app (no
// existing annotation to update).
LabelStudioAnnotationWriteResult createLabelStudioAnnotation(
    const std::string& baseUrl, const std::string& apiToken, int taskId, const nlohmann::json& resultArray);

// Updates an already-existing annotation via PATCH
// {baseUrl}/api/annotations/{annotationId}/ (Label Studio's annotation
// update endpoint is keyed by the annotation's own id, not the task's).
// Used when the task being labeled already had an annotation (loaded via
// fetchLabelStudioTaskById's `annotationId`).
LabelStudioAnnotationWriteResult updateLabelStudioAnnotation(
    const std::string& baseUrl, const std::string& apiToken, int annotationId, const nlohmann::json& resultArray);

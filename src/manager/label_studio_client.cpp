#include "manager/label_studio_client.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

std::string normalizeBaseUrl(const std::string& baseUrl) {
    if (!baseUrl.empty() && baseUrl.back() == '/') {
        return baseUrl.substr(0, baseUrl.size() - 1);
    }
    return baseUrl;
}

// Performs a GET request to `url` with a Label Studio token auth header.
// Returns false only on a request-level (network) failure, with
// `networkError` set -- a non-2xx response is still a "successful"
// request as far as this function is concerned; the caller checks
// `httpCode` itself.
bool performGet(
    const std::string& url, const std::string& apiToken, std::string& responseBody, long& httpCode,
    std::string& networkError) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        networkError = "Failed to initialize HTTP client";
        return false;
    }

    const std::string authHeader = "Authorization: Token " + apiToken;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, authHeader.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    const CURLcode res = curl_easy_perform(curl);
    bool ok = true;
    if (res != CURLE_OK) {
        networkError = std::string("Request failed: ") + curl_easy_strerror(res);
        ok = false;
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return ok;
}

} // namespace

std::string extractXmlTagAttribute(const std::string& xml, const std::string& tagName, const std::string& attributeName) {
    const std::regex tagRegex("<" + tagName + "\\b[^>]*>");
    std::smatch tagMatch;
    if (!std::regex_search(xml, tagMatch, tagRegex)) {
        return "";
    }
    const std::string tag = tagMatch.str();
    const std::regex attrRegex("\\b" + attributeName + "=\"([^\"]*)\"");
    std::smatch attrMatch;
    if (!std::regex_search(tag, attrMatch, attrRegex)) {
        return "";
    }
    return attrMatch[1].str();
}

std::string extractXmlTagNameAttribute(const std::string& xml, const std::string& tagName) {
    return extractXmlTagAttribute(xml, tagName, "name");
}

namespace {

// timegm is POSIX (available on this app's only build target, Linux) --
// unlike mktime, it interprets `tm` as UTC instead of the local timezone,
// which is exactly what both callers below need.
std::optional<std::time_t> parseUtcWithFormat(const std::string& value, const char* format) {
    std::tm tm{};
    std::istringstream iss(value);
    iss >> std::get_time(&tm, format);
    if (iss.fail()) {
        return std::nullopt;
    }
    const std::time_t result = timegm(&tm);
    if (result == static_cast<std::time_t>(-1)) {
        return std::nullopt;
    }
    return result;
}

} // namespace

std::optional<std::time_t> parseIso8601Utc(const std::string& value) {
    return parseUtcWithFormat(value, "%Y-%m-%dT%H:%M:%S");
}

std::optional<std::time_t> parseTypedUtcTimestamp(const std::string& value) {
    return parseUtcWithFormat(value, "%Y/%m/%d %H:%M:%S");
}

LabelStudioLabelingConfig fetchLabelStudioLabelingConfig(
    const std::string& baseUrl, int projectId, const std::string& apiToken, bool isDetection) {
    LabelStudioLabelingConfig result;
    if (baseUrl.empty()) {
        result.error = "Label Studio base URL is empty";
        return result;
    }

    const std::string url = normalizeBaseUrl(baseUrl) + "/api/projects/" + std::to_string(projectId) + "/";

    std::string responseBody;
    long httpCode = 0;
    std::string networkError;
    if (!performGet(url, apiToken, responseBody, httpCode, networkError)) {
        result.error = networkError;
        return result;
    }
    if (httpCode < 200 || httpCode >= 300) {
        result.error = "Label Studio returned HTTP " + std::to_string(httpCode) + ": " + responseBody;
        return result;
    }

    try {
        const auto parsed = nlohmann::json::parse(responseBody);
        if (!parsed.contains("label_config") || !parsed["label_config"].is_string()) {
            result.error = "Project response has no label_config";
            return result;
        }
        const std::string labelConfig = parsed["label_config"].get<std::string>();
        result.toName = extractXmlTagNameAttribute(labelConfig, "Image");
        result.fromName = extractXmlTagNameAttribute(labelConfig, isDetection ? "RectangleLabels" : "Choices");
        const std::string rawDataKey = extractXmlTagAttribute(labelConfig, "Image", "value");
        result.dataImageKey = (!rawDataKey.empty() && rawDataKey.front() == '$') ? rawDataKey.substr(1) : rawDataKey;
        if (result.toName.empty() || result.fromName.empty() || result.dataImageKey.empty()) {
            result.error = "Could not find the expected tags in this project's labeling config";
        }
    } catch (const nlohmann::json::parse_error&) {
        result.error = "Failed to parse project response";
    }

    return result;
}

std::vector<int> selectMostRecentTaskIds(const nlohmann::json& tasksJson, size_t count) {
    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }

    std::vector<int> ids;
    if (tasks->is_array()) {
        for (const auto& task : *tasks) {
            if (task.contains("id") && task["id"].is_number_integer()) {
                ids.push_back(task["id"].get<int>());
            }
        }
    }

    std::sort(ids.begin(), ids.end(), std::greater<int>());
    if (ids.size() > count) {
        ids.resize(count);
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::vector<LabelStudioUnlabeledTask> selectUnlabeledTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey) {
    std::vector<LabelStudioUnlabeledTask> result;

    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }
    if (!tasks->is_array()) {
        return result;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }

        int predictionCount = 0;
        if (task.contains("total_predictions") && task["total_predictions"].is_number_integer()) {
            predictionCount = task["total_predictions"].get<int>();
        } else if (task.contains("predictions") && task["predictions"].is_array()) {
            predictionCount = static_cast<int>(task["predictions"].size());
        }

        int annotationCount = 0;
        if (task.contains("total_annotations") && task["total_annotations"].is_number_integer()) {
            annotationCount = task["total_annotations"].get<int>();
        } else if (task.contains("annotations") && task["annotations"].is_array()) {
            annotationCount = static_cast<int>(task["annotations"].size());
        }

        if (predictionCount > 0 || annotationCount > 0) {
            continue;
        }

        if (!task.contains("data") || !task["data"].is_object()) {
            continue;
        }
        const auto& data = task["data"];
        if (!data.contains(dataImageKey) || !data[dataImageKey].is_string()) {
            continue;
        }

        LabelStudioUnlabeledTask unlabeled;
        unlabeled.taskId = task["id"].get<int>();
        unlabeled.imagePath = data[dataImageKey].get<std::string>();
        result.push_back(std::move(unlabeled));
    }

    return result;
}

std::vector<LabelStudioLabeledTask> selectLabeledTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey) {
    std::vector<LabelStudioLabeledTask> result;

    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }
    if (!tasks->is_array()) {
        return result;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }

        int annotationCount = 0;
        if (task.contains("total_annotations") && task["total_annotations"].is_number_integer()) {
            annotationCount = task["total_annotations"].get<int>();
        } else if (task.contains("annotations") && task["annotations"].is_array()) {
            annotationCount = static_cast<int>(task["annotations"].size());
        }
        if (annotationCount <= 0) {
            continue;
        }

        if (!task.contains("data") || !task["data"].is_object()) {
            continue;
        }
        const auto& data = task["data"];
        if (!data.contains(dataImageKey) || !data[dataImageKey].is_string()) {
            continue;
        }

        LabelStudioLabeledTask labeled;
        labeled.taskId = task["id"].get<int>();
        labeled.imagePath = data[dataImageKey].get<std::string>();
        result.push_back(std::move(labeled));
    }

    return result;
}

std::vector<std::vector<TimestampMatchCandidate>> matchTasksToTimestamps(
    const nlohmann::json& tasksJson, const std::string& dataImageKey, const std::vector<TimestampMatchQuery>& queries) {
    std::vector<std::vector<TimestampMatchCandidate>> result(queries.size());

    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }
    if (!tasks->is_array()) {
        return result;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        if (!task.contains("created_at") || !task["created_at"].is_string()) {
            continue;
        }
        const std::optional<std::time_t> createdAt = parseIso8601Utc(task["created_at"].get<std::string>());
        if (!createdAt) {
            continue;
        }
        if (!task.contains("data") || !task["data"].is_object()) {
            continue;
        }
        const auto& data = task["data"];
        if (!data.contains(dataImageKey) || !data[dataImageKey].is_string()) {
            continue;
        }

        TimestampMatchCandidate base;
        base.taskId = task["id"].get<int>();
        base.imagePath = data[dataImageKey].get<std::string>();
        base.createdAt = *createdAt;

        for (size_t i = 0; i < queries.size(); ++i) {
            const long long delta =
                static_cast<long long>(*createdAt) - static_cast<long long>(queries[i].timestamp);
            if (std::llabs(delta) > queries[i].toleranceSeconds) {
                continue;
            }
            TimestampMatchCandidate candidate = base;
            candidate.deltaSeconds = delta;
            result[i].push_back(candidate);
        }
    }

    for (auto& candidates : result) {
        std::sort(
            candidates.begin(), candidates.end(),
            [](const TimestampMatchCandidate& a, const TimestampMatchCandidate& b) {
                return std::llabs(a.deltaSeconds) < std::llabs(b.deltaSeconds);
            });
    }

    return result;
}

std::optional<int> parseTaskIdFromFilename(const std::string& filename) {
    const std::string stem = std::filesystem::path(filename).stem().string();
    if (stem.empty()) {
        return std::nullopt;
    }
    for (char c : stem) {
        if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
            return std::nullopt;
        }
    }
    try {
        return std::stoi(stem);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

namespace {

// This tool's own tag on predictions it creates, so a re-push can find and
// replace its own prior predictions without touching anyone else's.
constexpr const char* kModelVersion = "vision_app_label_assistant";

// Fetches every task in the project (with predictions embedded, via Label
// Studio's `fields=all` query param), paging defensively: stops when a
// page returns fewer tasks than requested, an empty page, or (if present)
// a null/absent "next" link. Handles a bare-array response, or an object
// with a "tasks" or "results" array -- Label Studio's exact response
// shape here isn't pinned to one version. Capped at 1000 pages as a
// safety valve against an unexpected server response looping forever.
bool fetchAllLabelStudioTasksRaw(
    const std::string& baseUrl, int projectId, const std::string& apiToken, nlohmann::json& allTasks,
    std::string& error) {
    allTasks = nlohmann::json::array();
    const int pageSize = 200;

    for (int page = 1; page <= 1000; ++page) {
        const std::string url = normalizeBaseUrl(baseUrl) + "/api/tasks/?project=" + std::to_string(projectId)
            + "&fields=all&page=" + std::to_string(page) + "&page_size=" + std::to_string(pageSize);

        std::string responseBody;
        long httpCode = 0;
        std::string networkError;
        if (!performGet(url, apiToken, responseBody, httpCode, networkError)) {
            error = networkError;
            return false;
        }
        if (httpCode < 200 || httpCode >= 300) {
            error = "Label Studio returned HTTP " + std::to_string(httpCode) + ": " + responseBody;
            return false;
        }

        nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(responseBody);
        } catch (const nlohmann::json::parse_error&) {
            error = "Failed to parse task list response";
            return false;
        }

        const nlohmann::json* pageTasks = &parsed;
        bool hasMore = false;
        if (parsed.is_object()) {
            if (parsed.contains("tasks") && parsed["tasks"].is_array()) {
                pageTasks = &parsed["tasks"];
            } else if (parsed.contains("results") && parsed["results"].is_array()) {
                pageTasks = &parsed["results"];
            }
            if (parsed.contains("next") && !parsed["next"].is_null()) {
                hasMore = true;
            }
        }
        if (!pageTasks->is_array()) {
            break;
        }

        for (const auto& task : *pageTasks) {
            allTasks.push_back(task);
        }

        if (pageTasks->empty() || pageTasks->size() < static_cast<size_t>(pageSize)) {
            hasMore = false;
        }
        if (!hasMore) {
            break;
        }
    }

    return true;
}

// Uploads `imagePath`'s bytes as a real multipart file upload to the
// project's import endpoint, creating a brand-new task. Returns true only
// on a 2xx response; the caller resolves the resulting task id
// separately (see selectMostRecentTaskIds) since the id this endpoint
// does return (`file_upload_ids`) doesn't correspond to anything the task
// list itself exposes.
bool uploadImage(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& imagePath,
    std::string& error) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        error = "Failed to initialize HTTP client";
        return false;
    }

    curl_mime* mime = curl_mime_init(curl);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filedata(part, imagePath.c_str());

    const std::string url = normalizeBaseUrl(baseUrl) + "/api/projects/" + std::to_string(projectId) + "/import";
    const std::string authHeader = "Authorization: Token " + apiToken;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, authHeader.c_str());

    std::string responseBody;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    const CURLcode res = curl_easy_perform(curl);
    bool ok = false;
    if (res != CURLE_OK) {
        error = std::string("Request failed: ") + curl_easy_strerror(res);
    } else {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode >= 200 && httpCode < 300) {
            ok = true;
        } else {
            error = "Label Studio returned HTTP " + std::to_string(httpCode) + ": " + responseBody;
        }
    }

    curl_mime_free(mime);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return ok;
}

// Downloads the image at `{baseUrl}{imagePath}` (imagePath already
// absolute, e.g. "/data/upload/11/xxx.png", as found in a task's `data`)
// with the same token auth as every other call here, writing the raw
// bytes to `localOutputPath`. Returns false on any failure (network,
// non-2xx, or the local file couldn't be written), with `error` set.
bool downloadTaskImage(
    const std::string& baseUrl, const std::string& apiToken, const std::string& imagePath,
    const std::string& localOutputPath, std::string& error) {
    const std::string url = normalizeBaseUrl(baseUrl) + imagePath;

    std::string responseBody;
    long httpCode = 0;
    std::string networkError;
    if (!performGet(url, apiToken, responseBody, httpCode, networkError)) {
        error = networkError;
        return false;
    }
    if (httpCode < 200 || httpCode >= 300) {
        error = "Label Studio returned HTTP " + std::to_string(httpCode) + " downloading " + imagePath;
        return false;
    }

    std::ofstream file(localOutputPath, std::ios::binary);
    if (!file) {
        error = "Could not open file for writing: " + localOutputPath;
        return false;
    }
    file.write(responseBody.data(), static_cast<std::streamsize>(responseBody.size()));
    if (!file) {
        error = "Failed to write downloaded image to: " + localOutputPath;
        return false;
    }

    return true;
}

bool createLabelStudioPredictionInternal(
    const std::string& baseUrl, const std::string& apiToken, int taskId, const nlohmann::json& resultArray,
    float score) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        return false;
    }

    nlohmann::json body;
    body["task"] = taskId;
    body["result"] = resultArray;
    body["score"] = score;
    body["model_version"] = kModelVersion;
    const std::string bodyStr = body.dump();

    const std::string url = normalizeBaseUrl(baseUrl) + "/api/predictions/";
    const std::string authHeader = "Authorization: Token " + apiToken;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, authHeader.c_str());

    std::string responseBody;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, bodyStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(bodyStr.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    const CURLcode res = curl_easy_perform(curl);
    bool ok = false;
    if (res == CURLE_OK) {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        ok = httpCode >= 200 && httpCode < 300;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return ok;
}

} // namespace

LabelStudioPushSummary pushDraftsAsNewLabelStudioTasks(
    const std::string& baseUrl, int projectId, const std::string& apiToken,
    const std::vector<LabelStudioPredictionInput>& predictions) {
    LabelStudioPushSummary summary;

    std::vector<const LabelStudioPredictionInput*> uploaded;
    uploaded.reserve(predictions.size());

    for (const auto& prediction : predictions) {
        std::string uploadError;
        if (uploadImage(baseUrl, projectId, apiToken, prediction.localImagePath, uploadError)) {
            uploaded.push_back(&prediction);
        } else {
            summary.uploadFailed++;
        }
    }

    if (uploaded.empty()) {
        return summary;
    }

    nlohmann::json allTasks;
    std::string fetchError;
    if (!fetchAllLabelStudioTasksRaw(baseUrl, projectId, apiToken, allTasks, fetchError)) {
        summary.error = fetchError;
        return summary;
    }

    const std::vector<int> recentTaskIds = selectMostRecentTaskIds(allTasks, uploaded.size());
    if (recentTaskIds.size() < uploaded.size()) {
        summary.taskNotResolved += static_cast<int>(uploaded.size());
        return summary;
    }

    for (size_t i = 0; i < uploaded.size(); ++i) {
        if (createLabelStudioPredictionInternal(
                baseUrl, apiToken, recentTaskIds[i], uploaded[i]->resultAndScore.result,
                uploaded[i]->resultAndScore.score)) {
            summary.created++;
        } else {
            summary.predictionFailed++;
        }
    }

    return summary;
}

LabelStudioDownloadResult fetchAndDownloadUnlabeledTasks(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& dataImageKey,
    const std::string& outputFolder, const std::function<void(int completed, int total)>& onProgress,
    const std::atomic<bool>* cancelRequested) {
    LabelStudioDownloadResult result;

    nlohmann::json allTasks;
    std::string fetchError;
    if (!fetchAllLabelStudioTasksRaw(baseUrl, projectId, apiToken, allTasks, fetchError)) {
        result.error = fetchError;
        return result;
    }

    const std::vector<LabelStudioUnlabeledTask> unlabeled = selectUnlabeledTasks(allTasks, dataImageKey);
    const int total = static_cast<int>(unlabeled.size());
    int completed = 0;

    for (const auto& task : unlabeled) {
        if (cancelRequested != nullptr && cancelRequested->load()) {
            return result;
        }

        const std::string extension = std::filesystem::path(task.imagePath).extension().string();
        const std::string localPath =
            (std::filesystem::path(outputFolder) / (std::to_string(task.taskId) + extension)).string();

        std::string downloadError;
        if (downloadTaskImage(baseUrl, apiToken, task.imagePath, localPath, downloadError)) {
            result.downloaded++;
        } else {
            result.downloadFailed++;
        }

        completed++;
        if (onProgress) {
            onProgress(completed, total);
        }
    }

    return result;
}

LabelStudioAttachSummary attachPredictionsToKnownTasks(
    const std::string& baseUrl, const std::string& apiToken,
    const std::vector<LabelStudioKnownTaskPrediction>& predictions) {
    LabelStudioAttachSummary summary;

    for (const auto& prediction : predictions) {
        if (createLabelStudioPredictionInternal(
                baseUrl, apiToken, prediction.taskId, prediction.resultAndScore.result,
                prediction.resultAndScore.score)) {
            summary.created++;
        } else {
            summary.failed++;
        }
    }

    return summary;
}

LabelStudioGroundTruthDataset fetchAndDownloadLabeledDataset(
    const std::string& baseUrl, int projectId, const std::string& apiToken, const std::string& dataImageKey,
    const std::string& outputFolder, const std::function<void(int completed, int total)>& onProgress,
    const std::atomic<bool>* cancelRequested) {
    LabelStudioGroundTruthDataset result;

    nlohmann::json allTasks;
    std::string fetchError;
    if (!fetchAllLabelStudioTasksRaw(baseUrl, projectId, apiToken, allTasks, fetchError)) {
        result.error = fetchError;
        return result;
    }

    result.groundTruth = parseLabelStudioExport(allTasks);

    const std::vector<LabelStudioLabeledTask> labeled = selectLabeledTasks(allTasks, dataImageKey);
    const int total = static_cast<int>(labeled.size());
    int completed = 0;

    for (const auto& task : labeled) {
        if (cancelRequested != nullptr && cancelRequested->load()) {
            return result;
        }

        const std::string basename = std::filesystem::path(task.imagePath).filename().string();
        const std::string localPath = (std::filesystem::path(outputFolder) / basename).string();

        std::string downloadError;
        if (downloadTaskImage(baseUrl, apiToken, task.imagePath, localPath, downloadError)) {
            result.downloaded++;
        } else {
            result.downloadFailed++;
        }

        completed++;
        if (onProgress) {
            onProgress(completed, total);
        }
    }

    return result;
}

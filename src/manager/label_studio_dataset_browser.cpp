#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ostream>

namespace {

void collectClassNamesFromResultArray(const nlohmann::json& resultArray, std::vector<std::string>& outClassNames) {
    if (!resultArray.is_array()) {
        return;
    }
    for (const auto& item : resultArray) {
        if (!item.is_object() || !item.contains("value") || !item["value"].is_object()) {
            continue;
        }
        const auto& value = item["value"];
        for (const char* key : {"rectanglelabels", "choices", "brushlabels"}) {
            if (!value.contains(key) || !value[key].is_array()) {
                continue;
            }
            for (const auto& label : value[key]) {
                if (label.is_string()) {
                    outClassNames.push_back(label.get<std::string>());
                }
            }
        }
    }
}

void collectConfidenceFromPredictions(
    const nlohmann::json& predictions, std::optional<float>& minConfidence, std::optional<float>& maxConfidence) {
    if (!predictions.is_array()) {
        return;
    }
    for (const auto& prediction : predictions) {
        if (!prediction.is_object() || !prediction.contains("score") || !prediction["score"].is_number()) {
            continue;
        }
        const float score = prediction["score"].get<float>();
        minConfidence = minConfidence ? std::min(*minConfidence, score) : score;
        maxConfidence = maxConfidence ? std::max(*maxConfidence, score) : score;
    }
}

DatasetBoxesToDraw extractBoxesFromTaskJson(const nlohmann::json& task, const std::string& rectangleLabelsFromName) {
    DatasetBoxesToDraw out;
    if (rectangleLabelsFromName.empty()) {
        return out;
    }
    if (task.contains("annotations") && task["annotations"].is_array()) {
        for (const auto& annotation : task["annotations"]) {
            if (annotation.is_object() && annotation.contains("result")) {
                const auto annotationBoxes = parseDetectionResultBoxes(annotation["result"], rectangleLabelsFromName);
                out.annotationBoxes.insert(out.annotationBoxes.end(), annotationBoxes.begin(), annotationBoxes.end());
            }
        }
    }
    if (task.contains("predictions") && task["predictions"].is_array()) {
        for (const auto& prediction : task["predictions"]) {
            if (prediction.is_object() && prediction.contains("result")) {
                const auto predictionBoxes = parseDetectionResultBoxes(prediction["result"], rectangleLabelsFromName);
                out.predictionBoxes.insert(out.predictionBoxes.end(), predictionBoxes.begin(), predictionBoxes.end());
            }
        }
    }
    return out;
}

DatasetMasksToDraw extractMasksFromTaskJson(const nlohmann::json& task, const std::string& brushLabelsFromName) {
    DatasetMasksToDraw out;
    if (brushLabelsFromName.empty()) {
        return out;
    }
    if (task.contains("annotations") && task["annotations"].is_array()) {
        for (const auto& annotation : task["annotations"]) {
            if (annotation.is_object() && annotation.contains("result")) {
                const auto annotationMasks = parseBrushResultRegions(annotation["result"], brushLabelsFromName);
                out.annotationMasks.insert(out.annotationMasks.end(), annotationMasks.begin(), annotationMasks.end());
            }
        }
    }
    if (task.contains("predictions") && task["predictions"].is_array()) {
        for (const auto& prediction : task["predictions"]) {
            if (prediction.is_object() && prediction.contains("result")) {
                const auto predictionMasks = parseBrushResultRegions(prediction["result"], brushLabelsFromName);
                out.predictionMasks.insert(out.predictionMasks.end(), predictionMasks.begin(), predictionMasks.end());
            }
        }
    }
    return out;
}

// Resolves `tasksJson` to the actual task array, tolerating a bare array
// or an object wrapping one in "tasks" -- the same tolerance every other
// function in this file applies.
const nlohmann::json* resolveTaskArray(const nlohmann::json& tasksJson) {
    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }
    return tasks->is_array() ? tasks : nullptr;
}

} // namespace

std::vector<DatasetTaskSummary> summarizeDatasetTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey) {
    std::vector<DatasetTaskSummary> summaries;

    const nlohmann::json* tasks = resolveTaskArray(tasksJson);
    if (tasks == nullptr) {
        return summaries;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        if (!task.contains("data") || !task["data"].contains(dataImageKey) || !task["data"][dataImageKey].is_string()) {
            continue;
        }

        DatasetTaskSummary summary;
        summary.taskId = task["id"].get<int>();
        summary.imagePath = task["data"][dataImageKey].get<std::string>();
        if (task.contains("created_at") && task["created_at"].is_string()) {
            summary.createdAt = task["created_at"].get<std::string>();
        }

        if (task.contains("total_annotations") && task["total_annotations"].is_number_integer()) {
            summary.hasAnnotation = task["total_annotations"].get<int>() > 0;
        } else if (task.contains("annotations") && task["annotations"].is_array()) {
            summary.hasAnnotation = !task["annotations"].empty();
        }
        if (task.contains("total_predictions") && task["total_predictions"].is_number_integer()) {
            summary.hasPrediction = task["total_predictions"].get<int>() > 0;
        } else if (task.contains("predictions") && task["predictions"].is_array()) {
            summary.hasPrediction = !task["predictions"].empty();
        }

        std::vector<std::string> classNames;
        if (task.contains("annotations") && task["annotations"].is_array()) {
            for (const auto& annotation : task["annotations"]) {
                if (annotation.is_object() && annotation.contains("result")) {
                    collectClassNamesFromResultArray(annotation["result"], classNames);
                }
            }
        }
        if (task.contains("predictions") && task["predictions"].is_array()) {
            for (const auto& prediction : task["predictions"]) {
                if (prediction.is_object() && prediction.contains("result")) {
                    collectClassNamesFromResultArray(prediction["result"], classNames);
                }
            }
            collectConfidenceFromPredictions(task["predictions"], summary.minConfidence, summary.maxConfidence);
        }
        std::sort(classNames.begin(), classNames.end());
        classNames.erase(std::unique(classNames.begin(), classNames.end()), classNames.end());
        summary.classNames = std::move(classNames);

        summaries.push_back(std::move(summary));
    }

    return summaries;
}

std::vector<int> filterDatasetTasks(const std::vector<DatasetTaskSummary>& summaries, const DatasetFilterSpec& filter) {
    std::vector<int> matching;

    for (const auto& summary : summaries) {
        if (filter.annotationFilter == DatasetPresenceFilter::Has && !summary.hasAnnotation) {
            continue;
        }
        if (filter.annotationFilter == DatasetPresenceFilter::Lacks && summary.hasAnnotation) {
            continue;
        }
        if (filter.predictionFilter == DatasetPresenceFilter::Has && !summary.hasPrediction) {
            continue;
        }
        if (filter.predictionFilter == DatasetPresenceFilter::Lacks && summary.hasPrediction) {
            continue;
        }

        if (!filter.classNameFilter.empty()) {
            const bool hasClass =
                std::find(summary.classNames.begin(), summary.classNames.end(), filter.classNameFilter)
                != summary.classNames.end();
            if (!hasClass) {
                continue;
            }
        }

        if (filter.confidenceFilterMode == DatasetConfidenceFilterMode::LessThan) {
            if (!summary.minConfidence || !(*summary.minConfidence < filter.confidenceThreshold)) {
                continue;
            }
        } else if (filter.confidenceFilterMode == DatasetConfidenceFilterMode::GreaterThan) {
            if (!summary.maxConfidence || !(*summary.maxConfidence > filter.confidenceThreshold)) {
                continue;
            }
        }

        matching.push_back(summary.taskId);
    }

    return matching;
}

nlohmann::json buildDatasetExportJson(const nlohmann::json& allTasksRaw, const std::vector<int>& matchingTaskIds) {
    nlohmann::json out = nlohmann::json::array();

    const nlohmann::json* tasks = resolveTaskArray(allTasksRaw);
    if (tasks == nullptr) {
        return out;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        const int taskId = task["id"].get<int>();
        if (std::find(matchingTaskIds.begin(), matchingTaskIds.end(), taskId) != matchingTaskIds.end()) {
            out.push_back(task);
        }
    }

    return out;
}

DatasetBoxesToDraw boxesToDrawForTask(
    const nlohmann::json& allTasksRaw, const std::string& rectangleLabelsFromName, int taskId) {
    if (rectangleLabelsFromName.empty()) {
        return {};
    }
    const nlohmann::json* tasks = resolveTaskArray(allTasksRaw);
    if (tasks == nullptr) {
        return {};
    }
    for (const auto& task : *tasks) {
        if (task.contains("id") && task["id"].is_number_integer() && task["id"].get<int>() == taskId) {
            return extractBoxesFromTaskJson(task, rectangleLabelsFromName);
        }
    }
    return {};
}

DatasetMasksToDraw masksToDrawForTask(
    const nlohmann::json& allTasksRaw, const std::string& brushLabelsFromName, int taskId) {
    if (brushLabelsFromName.empty()) {
        return {};
    }
    const nlohmann::json* tasks = resolveTaskArray(allTasksRaw);
    if (tasks == nullptr) {
        return {};
    }
    for (const auto& task : *tasks) {
        if (task.contains("id") && task["id"].is_number_integer() && task["id"].get<int>() == taskId) {
            return extractMasksFromTaskJson(task, brushLabelsFromName);
        }
    }
    return {};
}

std::unordered_map<int, DatasetBoxesToDraw> buildBoxesByTaskId(
    const nlohmann::json& allTasksRaw, const std::string& rectangleLabelsFromName) {
    std::unordered_map<int, DatasetBoxesToDraw> result;
    if (rectangleLabelsFromName.empty()) {
        return result;
    }
    const nlohmann::json* tasks = resolveTaskArray(allTasksRaw);
    if (tasks == nullptr) {
        return result;
    }
    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        result[task["id"].get<int>()] = extractBoxesFromTaskJson(task, rectangleLabelsFromName);
    }
    return result;
}

std::unordered_map<int, DatasetMasksToDraw> buildMasksByTaskId(
    const nlohmann::json& allTasksRaw, const std::string& brushLabelsFromName) {
    std::unordered_map<int, DatasetMasksToDraw> result;
    if (brushLabelsFromName.empty()) {
        return result;
    }
    const nlohmann::json* tasks = resolveTaskArray(allTasksRaw);
    if (tasks == nullptr) {
        return result;
    }
    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        result[task["id"].get<int>()] = extractMasksFromTaskJson(task, brushLabelsFromName);
    }
    return result;
}

namespace {

// Same acceptance rules as parseBrushResultRegions (label_studio_import.cpp),
// plus: an rle value outside 0..255, a non-integer or non-positive
// dimension, or a non-string class name skips that item -- one malformed
// item among 100k+ tasks must not throw out of the worker thread.
void collectEncodedBrushMasks(
    const nlohmann::json& resultArray, const std::string& brushLabelsFromName, std::vector<DatasetEncodedMask>& out) {
    if (!resultArray.is_array()) {
        return;
    }
    for (const auto& item : resultArray) {
        if (!item.contains("type") || item["type"] != "brushlabels") {
            continue;
        }
        if (!item.contains("from_name") || item["from_name"] != brushLabelsFromName) {
            continue;
        }
        if (!item.contains("value") || !item["value"].contains("rle") || !item["value"]["rle"].is_array()) {
            continue;
        }
        if (!item.contains("original_width") || !item["original_width"].is_number_integer()
            || item["original_width"].get<int>() <= 0 || !item.contains("original_height")
            || !item["original_height"].is_number_integer() || item["original_height"].get<int>() <= 0) {
            continue;
        }
        if (!item["value"].contains("brushlabels") || !item["value"]["brushlabels"].is_array()
            || item["value"]["brushlabels"].empty() || !item["value"]["brushlabels"][0].is_string()) {
            continue;
        }

        DatasetEncodedMask mask;
        mask.width = item["original_width"].get<int>();
        mask.height = item["original_height"].get<int>();
        mask.className = item["value"]["brushlabels"][0].get<std::string>();
        mask.rle.reserve(item["value"]["rle"].size());
        bool valid = true;
        for (const auto& byteValue : item["value"]["rle"]) {
            if (!byteValue.is_number_integer() || byteValue.get<int>() < 0 || byteValue.get<int>() > 255) {
                valid = false;
                break;
            }
            mask.rle.push_back(static_cast<uint8_t>(byteValue.get<int>()));
        }
        if (valid) {
            out.push_back(std::move(mask));
        }
    }
}

DatasetEncodedMasks extractEncodedMasksFromTaskJson(const nlohmann::json& task, const std::string& brushLabelsFromName) {
    DatasetEncodedMasks out;
    if (task.contains("annotations") && task["annotations"].is_array()) {
        for (const auto& annotation : task["annotations"]) {
            if (annotation.is_object() && annotation.contains("result")) {
                collectEncodedBrushMasks(annotation["result"], brushLabelsFromName, out.annotationMasks);
            }
        }
    }
    if (task.contains("predictions") && task["predictions"].is_array()) {
        for (const auto& prediction : task["predictions"]) {
            if (prediction.is_object() && prediction.contains("result")) {
                collectEncodedBrushMasks(prediction["result"], brushLabelsFromName, out.predictionMasks);
            }
        }
    }
    return out;
}

} // namespace

std::unordered_map<int, DatasetEncodedMasks> buildEncodedMasksByTaskId(
    const nlohmann::json& tasksJson, const std::string& brushLabelsFromName) {
    std::unordered_map<int, DatasetEncodedMasks> result;
    if (brushLabelsFromName.empty()) {
        return result;
    }
    const nlohmann::json* tasks = resolveTaskArray(tasksJson);
    if (tasks == nullptr) {
        return result;
    }
    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        DatasetEncodedMasks masks = extractEncodedMasksFromTaskJson(task, brushLabelsFromName);
        if (!masks.annotationMasks.empty() || !masks.predictionMasks.empty()) {
            result[task["id"].get<int>()] = std::move(masks);
        }
    }
    return result;
}

std::vector<DraftBrushRegion> decodeDatasetMasks(const std::vector<DatasetEncodedMask>& masks) {
    std::vector<DraftBrushRegion> regions;
    regions.reserve(masks.size());
    for (const auto& encoded : masks) {
        const std::vector<int> rle(encoded.rle.begin(), encoded.rle.end());
        DraftBrushRegion region;
        region.mask = decodeLabelStudioRleToMask(rle, encoded.width, encoded.height);
        region.className = encoded.className;
        regions.push_back(std::move(region));
    }
    return regions;
}

size_t writeMatchingTasksAsJsonArrayElements(
    const nlohmann::json& pageTasks, std::unordered_set<int>& remainingTaskIds, std::ostream& out,
    bool& wroteAnyElement) {
    const nlohmann::json* tasks = resolveTaskArray(pageTasks);
    if (tasks == nullptr) {
        return 0;
    }
    size_t written = 0;
    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer()) {
            continue;
        }
        if (remainingTaskIds.erase(task["id"].get<int>()) == 0) {
            continue;
        }
        if (wroteAnyElement) {
            out << ",\n";
        }
        out << task.dump(2);
        wroteAnyElement = true;
        written++;
    }
    return written;
}

std::unordered_map<int, size_t> indexSummariesByTaskId(const std::vector<DatasetTaskSummary>& summaries) {
    std::unordered_map<int, size_t> result;
    for (size_t i = 0; i < summaries.size(); ++i) {
        result[summaries[i].taskId] = i;
    }
    return result;
}

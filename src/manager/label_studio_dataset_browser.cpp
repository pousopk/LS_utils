#include "manager/label_studio_dataset_browser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>

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

} // namespace

std::vector<DatasetTaskSummary> summarizeDatasetTasks(
    const nlohmann::json& tasksJson, const std::string& dataImageKey) {
    std::vector<DatasetTaskSummary> summaries;

    const nlohmann::json* tasks = &tasksJson;
    if (tasksJson.is_object() && tasksJson.contains("tasks") && tasksJson["tasks"].is_array()) {
        tasks = &tasksJson["tasks"];
    }
    if (!tasks->is_array()) {
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

    const nlohmann::json* tasks = &allTasksRaw;
    if (allTasksRaw.is_object() && allTasksRaw.contains("tasks") && allTasksRaw["tasks"].is_array()) {
        tasks = &allTasksRaw["tasks"];
    }
    if (!tasks->is_array()) {
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
    DatasetBoxesToDraw out;
    if (rectangleLabelsFromName.empty()) {
        return out;
    }

    const nlohmann::json* tasks = &allTasksRaw;
    if (allTasksRaw.is_object() && allTasksRaw.contains("tasks") && allTasksRaw["tasks"].is_array()) {
        tasks = &allTasksRaw["tasks"];
    }
    if (!tasks->is_array()) {
        return out;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer() || task["id"].get<int>() != taskId) {
            continue;
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
        break;
    }

    return out;
}

DatasetMasksToDraw masksToDrawForTask(
    const nlohmann::json& allTasksRaw, const std::string& brushLabelsFromName, int taskId) {
    DatasetMasksToDraw out;
    if (brushLabelsFromName.empty()) {
        return out;
    }

    const nlohmann::json* tasks = &allTasksRaw;
    if (allTasksRaw.is_object() && allTasksRaw.contains("tasks") && allTasksRaw["tasks"].is_array()) {
        tasks = &allTasksRaw["tasks"];
    }
    if (!tasks->is_array()) {
        return out;
    }

    for (const auto& task : *tasks) {
        if (!task.contains("id") || !task["id"].is_number_integer() || task["id"].get<int>() != taskId) {
            continue;
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
        break;
    }

    return out;
}

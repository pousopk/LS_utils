#include "manager/benchmark_filters.hpp"

#include <algorithm>
#include <set>

DetectionMatch matchDetections(
    const std::vector<Detection>& predictions, const std::vector<GroundTruthBox>& groundTruth, float iouThreshold) {
    DetectionMatch match;
    match.predictionMatched.assign(predictions.size(), false);
    match.groundTruthMatched.assign(groundTruth.size(), false);

    for (size_t p = 0; p < predictions.size(); ++p) {
        const Detection& prediction = predictions[p];
        int bestIdx = -1;
        float bestIoU = iouThreshold;
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            if (match.groundTruthMatched[g] || groundTruth[g].className != prediction.className) {
                continue;
            }
            const float iou = computeRotatedIoU(
                prediction.box, prediction.rotationDegrees, groundTruth[g].box, groundTruth[g].rotationDegrees);
            if (iou >= bestIoU) {
                bestIoU = iou;
                bestIdx = static_cast<int>(g);
            }
        }
        if (bestIdx >= 0) {
            match.groundTruthMatched[static_cast<size_t>(bestIdx)] = true;
            match.predictionMatched[p] = true;
        }
    }
    return match;
}

std::optional<float> benchmarkImageConfidence(
    ModelTask mode, const BenchmarkImageResult* image, BenchmarkConfidenceBasis basis) {
    if (image == nullptr) {
        return std::nullopt;
    }
    if (mode == ModelTask::Classification) {
        if (image->predictions.empty()) {
            return std::nullopt;
        }
        return image->predictions.front().probability;
    }
    if (mode == ModelTask::Anomaly) {
        if (!image->anomalyResult) {
            return std::nullopt;
        }
        return image->anomalyResult->score;
    }
    if (image->detections.empty()) {
        return std::nullopt;
    }
    const auto byConfidence = [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; };
    switch (basis) {
    case BenchmarkConfidenceBasis::Lowest:
        return std::min_element(image->detections.begin(), image->detections.end(), byConfidence)->confidence;
    case BenchmarkConfidenceBasis::Highest:
        return std::max_element(image->detections.begin(), image->detections.end(), byConfidence)->confidence;
    case BenchmarkConfidenceBasis::Mean:
        break;
    }
    float sum = 0.0f;
    for (const auto& detection : image->detections) {
        sum += detection.confidence;
    }
    return sum / static_cast<float>(image->detections.size());
}

float benchmarkImageSortConfidence(
    ModelTask mode, const BenchmarkImageResult* imageA, const BenchmarkImageResult* imageB,
    BenchmarkConfidenceBasis basis) {
    const std::optional<float> confA = benchmarkImageConfidence(mode, imageA, basis);
    const std::optional<float> confB = benchmarkImageConfidence(mode, imageB, basis);
    if (!confA && !confB) {
        return 0.0f;
    }
    if (!confA) {
        return *confB;
    }
    if (!confB) {
        return *confA;
    }
    return std::min(*confA, *confB);
}

namespace {

// True if `pred` holds for slot A or slot B; a null slot never participates.
template <typename Pred>
bool eitherSlot(const BenchmarkImageResult* imageA, const BenchmarkImageResult* imageB, Pred pred) {
    return (imageA != nullptr && pred(*imageA)) || (imageB != nullptr && pred(*imageB));
}

bool slotHasError(
    ModelTask mode, const BenchmarkImageResult& image, BenchmarkErrorFilter errorFilter,
    const std::string& className) {
    if (!image.hasGroundTruth) {
        return false;
    }
    if (mode == ModelTask::Classification) {
        if (image.predictions.empty()) {
            return false;
        }
        const std::string& predicted = image.predictions.front().className;
        if (predicted == image.groundTruthLabel) {
            return false;
        }
        return className.empty() || image.groundTruthLabel == className || predicted == className;
    }
    const DetectionMatch match = matchDetections(image.detections, image.groundTruthBoxes);
    const bool wantFalsePositives =
        errorFilter == BenchmarkErrorFilter::AnyError || errorFilter == BenchmarkErrorFilter::FalsePositives;
    const bool wantMissed = errorFilter == BenchmarkErrorFilter::AnyError || errorFilter == BenchmarkErrorFilter::Missed;
    if (wantFalsePositives) {
        for (size_t p = 0; p < image.detections.size(); ++p) {
            if (!match.predictionMatched[p] && (className.empty() || image.detections[p].className == className)) {
                return true;
            }
        }
    }
    if (wantMissed) {
        for (size_t g = 0; g < image.groundTruthBoxes.size(); ++g) {
            if (!match.groundTruthMatched[g] && (className.empty() || image.groundTruthBoxes[g].className == className)) {
                return true;
            }
        }
    }
    return false;
}

bool slotHasClass(ModelTask mode, const BenchmarkImageResult& image, const std::string& className) {
    if (mode == ModelTask::Classification) {
        if (image.hasGroundTruth && image.groundTruthLabel == className) {
            return true;
        }
        return !image.predictions.empty() && image.predictions.front().className == className;
    }
    if (image.hasGroundTruth) {
        for (const auto& box : image.groundTruthBoxes) {
            if (box.className == className) {
                return true;
            }
        }
    }
    for (const auto& detection : image.detections) {
        if (detection.className == className) {
            return true;
        }
    }
    return false;
}

bool passesConfidenceFilter(
    ModelTask mode, const BenchmarkImageResult* imageA, const BenchmarkImageResult* imageB,
    const BenchmarkImageFilters& filters) {
    if (filters.confidence.mode == ThresholdMode::None) {
        return true;
    }
    return eitherSlot(imageA, imageB, [&](const BenchmarkImageResult& image) {
        return filters.confidence.passes(benchmarkImageConfidence(mode, &image, filters.confidenceBasis));
    });
}

bool passesDetectionPresenceFilter(
    ModelTask mode, const BenchmarkImageResult* imageA, const BenchmarkImageResult* imageB,
    const PresenceFilter& presence) {
    if (mode != ModelTask::Detection || presence.presence == Presence::Any) {
        return true;
    }
    return eitherSlot(imageA, imageB, [&](const BenchmarkImageResult& image) {
        return presence.passes(!image.detections.empty());
    });
}

bool slotsDisagree(ModelTask mode, const BenchmarkImageResult& imageA, const BenchmarkImageResult& imageB) {
    if (mode == ModelTask::Classification) {
        const bool emptyA = imageA.predictions.empty();
        const bool emptyB = imageB.predictions.empty();
        if (emptyA || emptyB) {
            return emptyA != emptyB;
        }
        return imageA.predictions.front().className != imageB.predictions.front().className;
    }
    // Detection: treat B's boxes as "ground truth" for A; any unmatched box
    // on either side is a disagreement.
    std::vector<GroundTruthBox> boxesB;
    boxesB.reserve(imageB.detections.size());
    for (const auto& detection : imageB.detections) {
        boxesB.push_back(GroundTruthBox{detection.box, detection.className, detection.rotationDegrees});
    }
    const DetectionMatch match = matchDetections(imageA.detections, boxesB);
    for (const bool matched : match.predictionMatched) {
        if (!matched) {
            return true;
        }
    }
    for (const bool matched : match.groundTruthMatched) {
        if (!matched) {
            return true;
        }
    }
    return false;
}

bool passesConfusionCell(const BenchmarkImageResult* image, const BenchmarkConfusionCellFilter& cell) {
    return image != nullptr && image->hasGroundTruth && !image->predictions.empty()
        && image->groundTruthLabel == cell.trueLabel && image->predictions.front().className == cell.predictedLabel;
}

} // namespace

bool benchmarkImagePassesFilters(
    ModelTask mode, bool hasGroundTruth,
    const BenchmarkImageResult* imageA, const BenchmarkImageResult* imageB,
    const BenchmarkImageFilters& filters) {
    if (mode != ModelTask::Anomaly) {
        const bool errorFilterActive = hasGroundTruth && filters.errorFilter != BenchmarkErrorFilter::Any;
        if (errorFilterActive) {
            // With a class selected, the error must involve that class.
            const bool hasError = eitherSlot(imageA, imageB, [&](const BenchmarkImageResult& image) {
                return slotHasError(mode, image, filters.errorFilter, filters.cls.className);
            });
            if (!hasError) {
                return false;
            }
        } else if (!filters.cls.className.empty()) {
            const bool hasClass = eitherSlot(imageA, imageB, [&](const BenchmarkImageResult& image) {
                return slotHasClass(mode, image, filters.cls.className);
            });
            if (!hasClass) {
                return false;
            }
        }
    }
    if (!passesConfidenceFilter(mode, imageA, imageB, filters)) {
        return false;
    }
    if (!passesDetectionPresenceFilter(mode, imageA, imageB, filters.detections)) {
        return false;
    }
    if (filters.modelsDisagreeOnly && mode != ModelTask::Anomaly) {
        if (imageA == nullptr || imageB == nullptr || !slotsDisagree(mode, *imageA, *imageB)) {
            return false;
        }
    }
    if (filters.confusionCell && mode == ModelTask::Classification) {
        const BenchmarkImageResult* image = filters.confusionCell->slotIndex == 0 ? imageA : imageB;
        if (!passesConfusionCell(image, *filters.confusionCell)) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> collectBenchmarkClassNames(
    ModelTask mode, const BenchmarkResult& resultA, const BenchmarkResult& resultB) {
    if (mode == ModelTask::Anomaly) {
        return {};
    }
    std::set<std::string> names;
    for (const BenchmarkResult* result : {&resultA, &resultB}) {
        for (const auto& image : result->images) {
            if (mode == ModelTask::Classification) {
                if (image.hasGroundTruth && !image.groundTruthLabel.empty()) {
                    names.insert(image.groundTruthLabel);
                }
                if (!image.predictions.empty()) {
                    names.insert(image.predictions.front().className);
                }
            } else {
                for (const auto& box : image.groundTruthBoxes) {
                    names.insert(box.className);
                }
                for (const auto& detection : image.detections) {
                    names.insert(detection.className);
                }
            }
        }
    }
    return {names.begin(), names.end()};
}

std::optional<float> classPrecision(const ClassAveragePrecision& metrics) {
    const int denominator = metrics.truePositives + metrics.falsePositives;
    if (denominator == 0) {
        return std::nullopt;
    }
    return static_cast<float>(metrics.truePositives) / static_cast<float>(denominator);
}

std::optional<float> classRecall(const ClassAveragePrecision& metrics) {
    if (metrics.numGroundTruth == 0) {
        return std::nullopt;
    }
    return static_cast<float>(metrics.truePositives) / static_cast<float>(metrics.numGroundTruth);
}

const BenchmarkImageResult* findBenchmarkImage(const BenchmarkResult& result, const std::string& filename) {
    for (const auto& image : result.images) {
        if (image.imageFilename == filename) {
            return &image;
        }
    }
    return nullptr;
}

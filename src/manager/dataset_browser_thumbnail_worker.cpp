#include "manager/dataset_browser_thumbnail_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

// Rescales `boxes` (in original-image pixel space) by `scale` -- the
// same factor run() resized the decoded image by -- so the returned
// boxes are consistent with the (possibly resized) thumbnail's own pixel
// space. Rotation is scale-invariant, so only box.box and className
// carry through unchanged.
std::vector<DraftDetectionBox> scaleBoxesToThumbnail(const std::vector<DraftDetectionBox>& boxes, double scale) {
    std::vector<DraftDetectionBox> scaled;
    scaled.reserve(boxes.size());
    for (const auto& box : boxes) {
        DraftDetectionBox scaledBox = box;
        scaledBox.box = cv::Rect(
            static_cast<int>(std::lround(box.box.x * scale)), static_cast<int>(std::lround(box.box.y * scale)),
            static_cast<int>(std::lround(box.box.width * scale)), static_cast<int>(std::lround(box.box.height * scale)));
        scaled.push_back(std::move(scaledBox));
    }
    return scaled;
}

} // namespace

DatasetThumbnailWorker::DatasetThumbnailWorker() {
    thread_ = std::thread(&DatasetThumbnailWorker::run, this);
}

DatasetThumbnailWorker::~DatasetThumbnailWorker() {
    stopRequested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DatasetThumbnailWorker::setConnection(std::string baseUrl, std::string apiToken) {
    std::lock_guard<std::mutex> lock(connectionMutex_);
    baseUrl_ = std::move(baseUrl);
    apiToken_ = std::move(apiToken);
}

void DatasetThumbnailWorker::requestThumbnail(DatasetThumbnailRequest request) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (std::find(pendingIds_.begin(), pendingIds_.end(), request.taskId) != pendingIds_.end()) {
        return;
    }
    pendingIds_.push_back(request.taskId);
    pendingRequests_.push(std::move(request));
}

void DatasetThumbnailWorker::setStillWanted(std::vector<int> stillWantedTaskIds) {
    std::lock_guard<std::mutex> lock(stillWantedMutex_);
    stillWanted_ = std::move(stillWantedTaskIds);
}

void DatasetThumbnailWorker::drainResults(std::vector<DatasetThumbnailResult>& out, size_t maxResults) {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    const size_t count = std::min(maxResults, completedResults_.size());
    out.assign(
        std::make_move_iterator(completedResults_.begin()),
        std::make_move_iterator(completedResults_.begin() + static_cast<long>(count)));
    completedResults_.erase(completedResults_.begin(), completedResults_.begin() + static_cast<long>(count));
}

void DatasetThumbnailWorker::run() {
    while (!stopRequested_.load()) {
        DatasetThumbnailRequest request;
        bool hasRequest = false;
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (!pendingRequests_.empty()) {
                request = pendingRequests_.front();
                pendingRequests_.pop();
                pendingIds_.erase(
                    std::remove(pendingIds_.begin(), pendingIds_.end(), request.taskId), pendingIds_.end());
                hasRequest = true;
            }
        }
        if (!hasRequest) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(stillWantedMutex_);
            if (std::find(stillWanted_.begin(), stillWanted_.end(), request.taskId) == stillWanted_.end()) {
                continue; // scrolled past; don't waste time decoding
            }
        }

        std::string baseUrl;
        std::string apiToken;
        {
            std::lock_guard<std::mutex> lock(connectionMutex_);
            baseUrl = baseUrl_;
            apiToken = apiToken_;
        }

        DatasetThumbnailResult result;
        result.taskId = request.taskId;

        std::string bytes;
        std::string error;
        if (downloadTaskImageBytes(baseUrl, apiToken, request.imagePath, bytes, error)) {
            const std::vector<uchar> buffer(bytes.begin(), bytes.end());
            const cv::Mat decoded = cv::imdecode(buffer, cv::IMREAD_COLOR);
            if (!decoded.empty()) {
                const double computedScale = static_cast<double>(kDatasetThumbnailMaxDim)
                    / static_cast<double>(std::max(decoded.cols, decoded.rows));
                cv::Mat resized;
                double effectiveScale = 1.0;
                if (computedScale < 1.0) {
                    cv::resize(decoded, resized, cv::Size(), computedScale, computedScale, cv::INTER_AREA);
                    effectiveScale = computedScale;
                } else {
                    resized = decoded;
                }
                result.annotationBoxes = scaleBoxesToThumbnail(request.annotationBoxes, effectiveScale);
                result.predictionBoxes = scaleBoxesToThumbnail(request.predictionBoxes, effectiveScale);
                result.thumbnail = resized;
                result.success = true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(resultsMutex_);
            completedResults_.push_back(std::move(result));
        }
    }
}

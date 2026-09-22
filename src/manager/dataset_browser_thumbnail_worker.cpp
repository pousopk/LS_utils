#include "manager/dataset_browser_thumbnail_worker.hpp"

#include "manager/label_studio_client.hpp"
#include "manager/rotated_box_geometry.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

// Draws `boxes` (in original-image pixel space) onto `image` in `color`,
// scaling each box by `scale` first (the same factor run() resized the
// decoded image by) -- small, thin strokes tuned for a thumbnail, not
// annotateDetections's thicker defaults (tuned for a full-size preview
// frame). No confidence percentage in the label: parseDetectionResultBoxes
// leaves DraftDetectionBox::confidence at 0 for both annotations and
// predictions (this app's own convention, see its doc comment), so
// printing it would show a misleading "0%" on every box.
void drawBoxesOnThumbnail(
    cv::Mat& image, const std::vector<DraftDetectionBox>& boxes, double scale, const cv::Scalar& color) {
    for (const auto& box : boxes) {
        const cv::Rect scaledBox(
            static_cast<int>(std::lround(box.box.x * scale)), static_cast<int>(std::lround(box.box.y * scale)),
            static_cast<int>(std::lround(box.box.width * scale)), static_cast<int>(std::lround(box.box.height * scale)));

        if (box.rotationDegrees == 0.0f) {
            cv::rectangle(image, scaledBox, color, 1);
        } else {
            const auto corners = rotatedBoxCorners(scaledBox, box.rotationDegrees);
            std::vector<cv::Point> intCorners;
            intCorners.reserve(corners.size());
            for (const auto& corner : corners) {
                intCorners.emplace_back(
                    static_cast<int>(std::lround(corner.x)), static_cast<int>(std::lround(corner.y)));
            }
            const cv::Point* pts = intCorners.data();
            const int numPts = static_cast<int>(intCorners.size());
            cv::polylines(image, &pts, &numPts, 1, /*isClosed=*/true, color, 1);
        }

        if (!box.className.empty()) {
            const cv::Point labelOrigin(scaledBox.x, std::max(10, scaledBox.y - 2));
            cv::putText(image, box.className, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv::LINE_AA);
        }
    }
}

// Ground truth (annotation) boxes in green, matching this app's only
// other box-drawing convention (annotateDetections, model_evaluation_state.cpp).
// Predictions get a distinct orange so the two are never confused with
// each other on the same image.
const cv::Scalar kAnnotationBoxColor(0, 255, 0);
const cv::Scalar kPredictionBoxColor(0, 165, 255);

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
                drawBoxesOnThumbnail(resized, request.annotationBoxes, effectiveScale, kAnnotationBoxColor);
                drawBoxesOnThumbnail(resized, request.predictionBoxes, effectiveScale, kPredictionBoxColor);
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

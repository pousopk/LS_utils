#include "manager/dataset_browser_thumbnail_worker.hpp"

#include "manager/label_studio_client.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>

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
                const double scale = static_cast<double>(kDatasetThumbnailMaxDim)
                    / static_cast<double>(std::max(decoded.cols, decoded.rows));
                cv::Mat resized;
                if (scale < 1.0) {
                    cv::resize(decoded, resized, cv::Size(), scale, scale, cv::INTER_AREA);
                } else {
                    resized = decoded;
                }
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

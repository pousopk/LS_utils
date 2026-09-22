#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// The longer side a downloaded image is resized to before being handed
// back for texture upload -- small enough that hundreds can be resident
// at once without exhausting VRAM (see DatasetThumbnailCache's
// kResidentCap), large enough to still be recognizable in a grid cell.
constexpr int kDatasetThumbnailMaxDim = 160;

struct DatasetThumbnailRequest {
    int taskId = 0;
    std::string imagePath;   // task.data[dataImageKey], as returned by Label Studio
};

struct DatasetThumbnailResult {
    int taskId = 0;
    bool success = false;
    cv::Mat thumbnail;   // BGR, resized to kDatasetThumbnailMaxDim on its longer side; empty if !success
};

// A single background thread processing a FIFO queue of thumbnail
// requests: downloads each requested image's bytes (via
// downloadTaskImageBytes), decodes with cv::imdecode, and resizes to
// kDatasetThumbnailMaxDim -- CPU-only, no GL calls (the main thread does
// the actual texture upload from the decoded cv::Mat this hands back).
// One thread, not a pool -- matches every other worker in this app and
// avoids new curl-concurrency questions. Starts on construction and runs
// until destruction (unlike this app's other, start-once-per-job
// workers), since thumbnail requests arrive continuously while the grid
// is scrolled. Right before decoding each request (not just at enqueue
// time), checks it's still in the most recent setStillWanted() set, so a
// fast scroll doesn't waste time decoding thumbnails already scrolled
// past. Not copyable.
class DatasetThumbnailWorker {
public:
    DatasetThumbnailWorker();
    ~DatasetThumbnailWorker();
    DatasetThumbnailWorker(const DatasetThumbnailWorker&) = delete;
    DatasetThumbnailWorker& operator=(const DatasetThumbnailWorker&) = delete;

    // Sets the connection this worker downloads through -- call whenever
    // the active project's connection details change.
    void setConnection(std::string baseUrl, std::string apiToken);

    // Enqueues a request if `taskId` isn't already pending.
    void requestThumbnail(DatasetThumbnailRequest request);

    // Replaces the "still wanted" set the background thread consults
    // before decoding each request -- call this every frame with the
    // current visible+lookahead task ids.
    void setStillWanted(std::vector<int> stillWantedTaskIds);

    // Drains up to `maxResults` completed results (both successes and
    // failures) into `out`, non-blocking. Called once per frame by the
    // main thread, which does the actual GL upload.
    void drainResults(std::vector<DatasetThumbnailResult>& out, size_t maxResults);

private:
    void run();

    std::string baseUrl_;
    std::string apiToken_;
    mutable std::mutex connectionMutex_;

    std::thread thread_;
    std::atomic<bool> stopRequested_{false};

    mutable std::mutex queueMutex_;
    std::queue<DatasetThumbnailRequest> pendingRequests_;
    std::vector<int> pendingIds_;   // ids currently in pendingRequests_, so requestThumbnail can dedupe cheaply

    mutable std::mutex stillWantedMutex_;
    std::vector<int> stillWanted_;

    mutable std::mutex resultsMutex_;
    std::vector<DatasetThumbnailResult> completedResults_;
};

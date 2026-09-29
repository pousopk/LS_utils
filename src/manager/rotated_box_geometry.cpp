#include "manager/rotated_box_geometry.hpp"

#include <cmath>

cv::Point2f rotatePointClockwise(cv::Point2f point, cv::Point2f pivot, float degrees) {
    const double theta = static_cast<double>(degrees) * CV_PI / 180.0;
    const double dx = static_cast<double>(point.x) - static_cast<double>(pivot.x);
    const double dy = static_cast<double>(point.y) - static_cast<double>(pivot.y);
    const double cosT = std::cos(theta);
    const double sinT = std::sin(theta);
    const double rx = dx * cosT - dy * sinT;
    const double ry = dx * sinT + dy * cosT;
    return cv::Point2f(
        static_cast<float>(static_cast<double>(pivot.x) + rx), static_cast<float>(static_cast<double>(pivot.y) + ry));
}

std::array<cv::Point2f, 4> rotatedBoxCorners(const cv::Rect& box, float rotationDegrees) {
    const cv::Point2f pivot(static_cast<float>(box.x), static_cast<float>(box.y));
    const cv::Point2f topLeft = pivot;
    const cv::Point2f topRight(static_cast<float>(box.x + box.width), static_cast<float>(box.y));
    const cv::Point2f bottomRight(static_cast<float>(box.x + box.width), static_cast<float>(box.y + box.height));
    const cv::Point2f bottomLeft(static_cast<float>(box.x), static_cast<float>(box.y + box.height));
    return {
        rotatePointClockwise(topLeft, pivot, rotationDegrees),
        rotatePointClockwise(topRight, pivot, rotationDegrees),
        rotatePointClockwise(bottomRight, pivot, rotationDegrees),
        rotatePointClockwise(bottomLeft, pivot, rotationDegrees),
    };
}

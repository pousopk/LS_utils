#pragma once

#include <opencv2/core.hpp>

#include <array>

// Pure function: rotates `point` clockwise around `pivot` by `degrees`, in
// this codebase's image-space convention (x right, y down) -- clockwise as
// a viewer would see it on screen. The single primitive every other
// rotation-aware box function is built from.
cv::Point2f rotatePointClockwise(cv::Point2f point, cv::Point2f pivot, float degrees);

// Pure function: the 4 corners of `box` (top-left, top-right,
// bottom-right, bottom-left, in that order) after applying
// `rotationDegrees` around the box's own pivot (box.x, box.y) -- the
// top-left corner is always exactly (box.x, box.y) itself (rotating a
// point around itself is a no-op); the other 3 corners move. Used for
// drawing a rotated outline, placing handles at their true screen
// positions, and (via computeRotatedIoU) rotation-aware box matching.
std::array<cv::Point2f, 4> rotatedBoxCorners(const cv::Rect& box, float rotationDegrees);

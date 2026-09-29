#pragma once

#include "manager/label_studio_import.hpp"

#include <opencv2/core.hpp>

// Blends each region's mask onto a copy of `baseImage` (expected 3-channel
// BGR, matching cv::imread's default), tinted by colorForClassName, at a
// fixed alpha -- for visual display only, never written back to
// annotation data. Regions with an empty mask, or one whose size doesn't
// match baseImage's, are skipped (not an error -- a freshly-created
// region's mask matches the image by construction; a mismatch here would
// only happen from a bug elsewhere, and silently skipping is safer than
// crashing the display). Pure pixel manipulation, no GL/ImGui.
cv::Mat compositeMaskOverlay(const cv::Mat& baseImage, const std::vector<DraftBrushRegion>& regions);

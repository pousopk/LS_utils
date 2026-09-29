#pragma once

#include <imgui.h>

// Pure function: the largest size with `imageWidth:imageHeight`'s aspect
// ratio that fits inside maxWidth x maxHeight (each clamped to >= 1). An
// empty image returns the region itself; neither side is ever below 1px.
ImVec2 fitImageToRegion(int imageWidth, int imageHeight, float maxWidth, float maxHeight);

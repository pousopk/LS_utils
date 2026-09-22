#pragma once

#include <string>

struct LabelColor {
    unsigned char r = 200;
    unsigned char g = 200;
    unsigned char b = 200;
};

// Pure function: deterministically maps a class name to a distinct,
// visually stable color -- the same name always produces the same color
// (hash of the name -> hue, fixed saturation/value), with no per-project
// color configuration needed. Kept free of ImGui so it's testable in
// isolation; the widget layer converts LabelColor to whatever pixel
// format it needs (e.g. IM_COL32).
LabelColor colorForClassName(const std::string& className);

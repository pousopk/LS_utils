#include "manager/label_color.hpp"

#include <cmath>
#include <cstdint>

LabelColor colorForClassName(const std::string& className) {
    // FNV-1a: simple, deterministic, well-distributed for short strings.
    uint32_t hash = 2166136261u;
    for (const char c : className) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 16777619u;
    }

    const float hue = static_cast<float>(hash % 360u);
    const float saturation = 0.55f;
    const float value = 0.85f;

    // Standard HSV -> RGB conversion.
    const float c = value * saturation;
    const float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    const float m = value - c;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (hue < 60.0f) {
        r = c; g = x; b = 0.0f;
    } else if (hue < 120.0f) {
        r = x; g = c; b = 0.0f;
    } else if (hue < 180.0f) {
        r = 0.0f; g = c; b = x;
    } else if (hue < 240.0f) {
        r = 0.0f; g = x; b = c;
    } else if (hue < 300.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }

    LabelColor color;
    color.r = static_cast<unsigned char>((r + m) * 255.0f);
    color.g = static_cast<unsigned char>((g + m) * 255.0f);
    color.b = static_cast<unsigned char>((b + m) * 255.0f);
    return color;
}

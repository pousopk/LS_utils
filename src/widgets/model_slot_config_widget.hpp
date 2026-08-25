#pragma once

#include <functional>
#include <string>

// Draws the model-slot configuration block shared by the live model
// comparison window and the batch evaluation window: model/class-name
// paths with Browse buttons (the caller supplies onBrowseModel/
// onBrowseClasses to open its own file-picker popup, since each window
// owns its own picker state), input width/height fields, and -- when
// confThreshold/nmsThreshold are both non-null -- confidence/NMS
// threshold sliders (detection mode only). Displays autoDetectStatus/
// loadError below when non-empty. Returns true on the frame "Load Model"
// is clicked; the caller performs the actual load.
bool drawModelSlotConfigFields(
    const std::string& onnxPath,
    const std::string& classNamesPath,
    int& inputWidth,
    int& inputHeight,
    float* confThreshold,
    float* nmsThreshold,
    const std::string& autoDetectStatus,
    const std::string& loadError,
    const std::function<void()>& onBrowseModel,
    const std::function<void()>& onBrowseClasses);

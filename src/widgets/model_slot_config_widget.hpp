#pragma once

#include "manager/model_slot.hpp"

#include <functional>

// Draws the model-slot block shared by the Benchmark and Label Assistant
// tabs: the model path with a Browse button (the caller supplies
// onBrowseModel to open its own file-picker popup, and loads the model
// itself once a file is picked), the loaded model's one-line summary or
// load error, and the run-time thresholds for the loaded model's task
// (confidence/NMS for detection, anomaly threshold for anomaly).
// Everything else about the model is read from the ONNX file.
void drawModelSlotConfigFields(ModelSlotConfig& slot, const std::function<void()>& onBrowseModel);

# vision_ml

A C++ desktop application for labeling and model-evaluation work against [Label Studio](https://labelstud.io/) and ONNX models. The UI is GLFW + OpenGL + [Dear ImGui](https://github.com/ocornut/imgui).

It opens two windows. The **Label Studio** window has Connection, Labeling, Label Assistant, Find by Timestamp and Dataset Browser tabs that share one connection and one loaded task list. The **Model Evaluation** window loads one or two ONNX models (detector, classifier or anomaly model) and evaluates them over a batch of images — a local folder or a Label Studio project as ground truth — with precision/recall/AP metrics, confusion matrices and drill-down filters. The **Window** menu reopens either window.

## Features

### Model evaluation
- Load one or two ONNX models and compare them side by side over a batch of images.
- ONNX Runtime backend (GPU-optional) with automatic input-shape/preprocessing/class-name detection from model metadata.
- Detection, classification and anomaly modes.
- Batch mode against a folder of images or a Label Studio project, with ground-truth import, precision/recall/AP metrics, confusion matrices, drill-down filters, and a random-sample mode for large datasets on slow hardware.

### Label Assistant (auto-labeling)
- Run an already-trained ONNX classifier or detector directly over a folder of unlabeled images to draft predictions.
- Review drafted labels (sortable/filterable by confidence) with a live preview, including box overlays in Detection mode.
- Push predictions straight to a Label Studio project over its REST API, in two modes:
  - **Local Folder** — upload each image as a new task with its prediction attached.
  - **Label Studio Project** — point at an existing project instead of a folder; downloads its unlabeled tasks, runs inference on them, and attaches predictions directly to those same tasks (no upload, no duplicate tasks).

## Requirements

- CMake ≥ 3.16, a C++20 compiler
- ONNX Runtime (GPU-optional) — see [Build](#build) for how it's located
- Protobuf
- libcurl
- OpenCV
- glfw3, OpenGL
- Dear ImGui (vendored as the `imgui` git submodule)

## Build

```sh
git submodule update --init   # first time only, fetches imgui
cmake -S . -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j
```

ONNX Runtime is located via the `ONNXRUNTIME_DIR` cache variable (default `$HOME/code/onnxruntime`, e.g. `-DONNXRUNTIME_DIR=/opt/onnxruntime-linux-x64-gpu-1.19.0`). It must contain `include/onnxruntime_cxx_api.h` and `lib/libonnxruntime.so`. A CUDA-enabled build is used automatically if available and falls back to CPU otherwise.

ImGui is found through `IMGUI_DIR` (default `./imgui`, the submodule); override it with `-DIMGUI_DIR=/path/to/imgui`.

This produces `build/vision_ml`:

```sh
./build/vision_ml
```

The window layout is saved to `vision_ml.ini` in the working directory.

## Testing

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
```

## Project structure

- `src/manager/` — non-UI state and workers: Label Studio API client, shared task list, labeling, label assistant, timestamp search, dataset browser, ONNX inference, batch evaluation and metrics.
- `src/widgets/` — the ImGui presentation layer, one file per window/tab.
- `src/main.cpp` — the main loop.
- `ui_common/` — generic UI helpers (app shell, GL texture upload, file browser, theme).
- `third_party/` — vendored ONNX proto, nlohmann/json, pugixml.
- `tests/` — unit tests, registered with CTest.

cmake_minimum_required(VERSION 3.16)
project(vision_ml VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui" CACHE PATH "Dear ImGui source tree (the imgui submodule)")
if(NOT EXISTS "${IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "Dear ImGui not found at ${IMGUI_DIR}. Run `git submodule update --init` or pass -DIMGUI_DIR=...")
endif()

find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(CURL REQUIRED)

# ------------------------------------------------------------
# ONNX Runtime (GPU-optional inference backend)
# ------------------------------------------------------------
set(ONNXRUNTIME_DIR "$ENV{HOME}/code/onnxruntime" CACHE PATH
    "Path to an extracted ONNX Runtime GPU release (download from https://github.com/microsoft/onnxruntime/releases, e.g. onnxruntime-linux-x64-gpu-<version>.tgz) -- must contain include/onnxruntime_cxx_api.h and lib/libonnxruntime.so. GPU support is opportunistic at runtime (falls back to CPU automatically if the machine's CUDA/cuDNN don't match); this path just needs to contain a buildable ONNX Runtime.")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h"
   OR NOT EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    message(FATAL_ERROR
        "No ONNX Runtime found at ${ONNXRUNTIME_DIR}. Download a GPU release tarball from "
        "https://github.com/microsoft/onnxruntime/releases (onnxruntime-linux-x64-gpu-<version>.tgz), "
        "extract it, and set -DONNXRUNTIME_DIR=/path/to/extracted/dir, or place it at ${ONNXRUNTIME_DIR}.")
endif()

set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

# Every executable here links ONNX Runtime (through ml_core).
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")

function(vision_warnings target)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
endfunction()

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

file(GLOB UI_COMMON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ui_common/*.cpp)
add_library(ui_common STATIC ${UI_COMMON_SOURCES})
target_include_directories(ui_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ui_common PUBLIC imgui ${OpenCV_LIBS})
vision_warnings(ui_common)

add_library(pugixml STATIC third_party/pugixml/pugixml.cpp)
target_include_directories(pugixml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pugixml)

# The generated onnx-ml.pb.h lives in the binary dir; making it a PUBLIC
# include of its own target means every consumer (app AND tests) gets a
# build dependency on its generation -- the fix for the old clean-build race.
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS third_party/onnx/onnx-ml.proto)
add_library(onnx_proto STATIC ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(onnx_proto PUBLIC ${Protobuf_LIBRARIES})

file(GLOB ML_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/manager/*.cpp)
add_library(ml_core STATIC ${ML_CORE_SOURCES})
target_include_directories(ml_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(ml_core PUBLIC ui_common onnx_proto pugixml CURL::libcurl ${ONNXRUNTIME_LIBRARIES})
vision_warnings(ml_core)

file(GLOB ML_APP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/widgets/*.cpp)
add_executable(vision_ml ${ML_APP_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(vision_ml PRIVATE ml_core)
vision_warnings(vision_ml)

enable_testing()
# yolo_inference_tests.cpp is intentionally absent: it has been orphaned since
# commit b470a2d (it includes a header that no longer exists). Restoring or
# removing it is a separate task.
set(ML_TESTS
    batch_eval_filters_tests calendar_date_tests dataset_browser_tests detection_metrics_tests
    image_fit_tests item_filters_tests label_assistant_control_tag_tests label_assistant_filter_tests
    label_studio_client_tests label_studio_project_data_tests label_studio_session_tests labeling_state_tests
    obb_detection_tests path_picker_tests pugixml_smoke_tests)
foreach(test ${ML_TESTS})
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE ml_core)
    vision_warnings(${test})
    add_test(NAME ${test} COMMAND ${test})
endforeach()

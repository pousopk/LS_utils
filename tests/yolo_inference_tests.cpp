#include "manager/classification_inference.hpp"
#include "manager/onnx_metadata.hpp"
#include "manager/onnx_runtime_env.hpp"
#include "manager/yolo_inference.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}

bool approxEqual(float a, float b, float tolerance = 0.01f) {
    return std::fabs(a - b) <= tolerance;
}
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

void test_computeLetterboxTransform_wideFrame() {
    const LetterboxTransform t = computeLetterboxTransform(1280, 720, 640, 640);
    CHECK(approxEqual(t.scale, 0.5f));
    CHECK(approxEqual(t.padX, 0.0f));
    CHECK(approxEqual(t.padY, 140.0f));
}

void test_computeLetterboxTransform_tallFrame() {
    const LetterboxTransform t = computeLetterboxTransform(720, 1280, 640, 640);
    CHECK(approxEqual(t.scale, 0.5f));
    CHECK(approxEqual(t.padX, 140.0f));
    CHECK(approxEqual(t.padY, 0.0f));
}

void test_computeLetterboxTransform_nonSquareTarget() {
    // 1280x720 source into a 960x576 (non-square) target.
    // scale = min(960/1280, 576/720) = min(0.75, 0.8) = 0.75
    // scaledWidth=960, scaledHeight=540 -> padX=0, padY=(576-540)/2=18
    const LetterboxTransform t = computeLetterboxTransform(1280, 720, 960, 576);
    CHECK(approxEqual(t.scale, 0.75f));
    CHECK(approxEqual(t.padX, 0.0f));
    CHECK(approxEqual(t.padY, 18.0f));
}

void test_computeIoU_identical() {
    const cv::Rect r(0, 0, 10, 10);
    CHECK(approxEqual(computeIoU(r, r), 1.0f));
}

void test_computeIoU_disjoint() {
    CHECK(approxEqual(computeIoU(cv::Rect(0, 0, 10, 10), cv::Rect(20, 20, 10, 10)), 0.0f));
}

void test_computeIoU_partialOverlap() {
    // A: (0,0)-(10,10), B: (5,0)-(15,10) -> intersection 5x10=50, union 150
    const float iou = computeIoU(cv::Rect(0, 0, 10, 10), cv::Rect(5, 0, 10, 10));
    CHECK(approxEqual(iou, 50.0f / 150.0f));
}

cv::Mat makeSyntheticOutput(const std::vector<std::array<float, 5>>& rows) {
    // Each row: [cx, cy, w, h, classScore] -- single class for this test.
    cv::Mat output(static_cast<int>(rows.size()), 5, CV_32F);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (int c = 0; c < 5; ++c) {
            output.at<float>(static_cast<int>(r), c) = rows[r][c];
        }
    }
    return output;
}

void test_decodeYoloOutput_confidenceAndNms() {
    const cv::Mat output = makeSyntheticOutput({
        {50, 50, 20, 20, 0.9f},     // kept: highest-confidence of an overlapping pair
        {52, 51, 20, 20, 0.8f},     // suppressed: overlaps row 0 heavily (IoU ~0.75)
        {200, 200, 10, 10, 0.95f},  // kept: no overlap with anything
        {300, 300, 10, 10, 0.05f},  // dropped: below confidence threshold
    });
    const LetterboxTransform identity{1.0f, 0.0f, 0.0f};
    const auto detections = decodeYoloOutput(output, {"object"}, identity, 640, 640, 0.5f, 0.45f);

    CHECK(detections.size() == 2);
    bool foundHighConf = false;
    bool foundFarBox = false;
    for (const auto& d : detections) {
        CHECK(d.classId == 0);
        CHECK(d.className == "object");
        if (approxEqual(static_cast<float>(d.box.x), 40.0f, 1.0f) && approxEqual(static_cast<float>(d.box.y), 40.0f, 1.0f)) {
            foundHighConf = true;
        }
        if (approxEqual(static_cast<float>(d.box.x), 195.0f, 1.0f) && approxEqual(static_cast<float>(d.box.y), 195.0f, 1.0f)) {
            foundFarBox = true;
        }
    }
    CHECK(foundHighConf);
    CHECK(foundFarBox);
}

void test_computeBoxAgreement_matchesOverlappingSameClass() {
    std::vector<Detection> a = {
        Detection{cv::Rect(40, 40, 20, 20), 0, "object", 0.9f},
        Detection{cv::Rect(200, 200, 10, 10), 0, "object", 0.95f},
    };
    std::vector<Detection> b = {
        Detection{cv::Rect(42, 41, 20, 20), 0, "object", 0.8f},   // overlaps a[0], IoU ~0.75
        Detection{cv::Rect(500, 500, 10, 10), 1, "other", 0.7f},  // different class, no match
    };

    const BoxAgreement agreement = computeBoxAgreement(a, b, 0.45f);
    CHECK(agreement.matchedPairs == 1);
    CHECK(agreement.totalA == 2);
    CHECK(agreement.totalB == 2);
}

cv::Mat makeClassificationOutput(const std::vector<float>& probabilities) {
    cv::Mat output(1, static_cast<int>(probabilities.size()), CV_32F);
    for (size_t i = 0; i < probabilities.size(); ++i) {
        output.at<float>(0, static_cast<int>(i)) = probabilities[i];
    }
    return output;
}

void test_decodeClassificationOutput_sortsDescending() {
    const cv::Mat output = makeClassificationOutput({0.05f, 0.5f, 0.3f, 0.15f});
    const auto predictions = decodeClassificationOutput(output, {"a", "b", "c", "d"});

    CHECK(predictions.size() == 4);
    CHECK(predictions[0].classId == 1 && predictions[0].className == "b");
    CHECK(approxEqual(predictions[0].probability, 0.5f));
    CHECK(predictions[1].classId == 2 && predictions[1].className == "c");
    CHECK(predictions[2].classId == 3 && predictions[2].className == "d");
    CHECK(predictions[3].classId == 0 && predictions[3].className == "a");
}

void test_decodeClassificationOutput_rejectsBadSum() {
    const cv::Mat output = makeClassificationOutput({10.0f, 20.0f, 5.0f, 5.0f}); // sums to 40, not ~1.0
    bool threw = false;
    try {
        decodeClassificationOutput(output, {"a", "b", "c", "d"});
    } catch (const cv::Exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_decodeClassificationOutput_rejectsWrongShape() {
    int sizes[] = {1, 4, 10};
    cv::Mat output(3, sizes, CV_32F, cv::Scalar(0.0f));
    bool threw = false;
    try {
        decodeClassificationOutput(output, {});
    } catch (const cv::Exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_extractInputShape_valid() {
    onnx::ModelProto model;
    auto* input = model.mutable_graph()->add_input();
    auto* shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(448);
    shape->add_dim()->set_dim_value(576);

    OnnxInputShape outShape;
    std::string error;
    CHECK(extractInputShape(model, outShape, error));
    CHECK(outShape.channels == 3);
    CHECK(outShape.height == 448);
    CHECK(outShape.width == 576);
}

void test_extractInputShape_rejectsDynamicDim() {
    onnx::ModelProto model;
    auto* input = model.mutable_graph()->add_input();
    auto* shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(224);
    shape->add_dim()->set_dim_value(224);

    OnnxInputShape outShape;
    std::string error;
    CHECK(!extractInputShape(model, outShape, error));
    CHECK(!error.empty());
}

void test_extractInputShape_rejectsNoInputs() {
    onnx::ModelProto model;
    OnnxInputShape outShape;
    std::string error;
    CHECK(!extractInputShape(model, outShape, error));
}

void test_extractClassNames_valid() {
    onnx::ModelProto model;
    auto* prop = model.add_metadata_props();
    prop->set_key("names");
    prop->set_value("{0: 'person', 1: 'car', 2: 'bike'}");

    const auto names = extractClassNames(model);
    CHECK(names.size() == 3);
    CHECK(names[0] == "person");
    CHECK(names[1] == "car");
    CHECK(names[2] == "bike");
}

void test_extractClassNames_missingReturnsEmpty() {
    onnx::ModelProto model;
    CHECK(extractClassNames(model).empty());
}

void test_extractClassNames_malformedReturnsEmpty() {
    onnx::ModelProto model;
    auto* prop = model.add_metadata_props();
    prop->set_key("names");
    prop->set_value("not a dict at all");
    CHECK(extractClassNames(model).empty());
}

void test_extractClassNames_categoriesFormat() {
    // Real metadata string from ATL400-inspection-1-v1.onnx.
    onnx::ModelProto model;
    auto* prop = model.add_metadata_props();
    prop->set_key("categories");
    prop->set_value(
        "[{'id': 'f641d7e2-1688-40b6-b51b-190e94c984e7', 'name': 'Racor derecha', 'threshold': 0.05}, "
        "{'id': 'a3c65866-17d2-4795-8f7b-5b3c310e4e35', 'name': 'Racor mal posicionado', 'threshold': 0.6}, "
        "{'id': '49f08870-9f51-482a-9be6-3fc318b776ef', 'name': 'Racor izquierda', 'threshold': 0.3}, "
        "{'id': 'b0fe2492-69d1-4048-9d0b-6a481e55aaa5', 'name': 'No racor', 'threshold': 0.05}]");

    const auto names = extractClassNames(model);
    CHECK(names.size() == 4);
    CHECK(names[0] == "Racor derecha");
    CHECK(names[1] == "Racor mal posicionado");
    CHECK(names[2] == "Racor izquierda");
    CHECK(names[3] == "No racor");
}

void test_extractClassNames_prefersNamesOverCategories() {
    onnx::ModelProto model;
    auto* namesProp = model.add_metadata_props();
    namesProp->set_key("names");
    namesProp->set_value("{0: 'a', 1: 'b'}");
    auto* categoriesProp = model.add_metadata_props();
    categoriesProp->set_key("categories");
    categoriesProp->set_value("[{'name': 'x'}, {'name': 'y'}]");

    const auto names = extractClassNames(model);
    CHECK(names.size() == 2);
    CHECK(names[0] == "a");
    CHECK(names[1] == "b");
}

void test_extractPreprocessingHints_defaultsWhenAbsent() {
    onnx::ModelProto model;
    const OnnxPreprocessingHints hints = extractPreprocessingHints(model);
    CHECK(approxEqual(hints.inputScale, 1.0f / 255.0f));
    CHECK(approxEqual(hints.padFill, 114.0f));
    CHECK(hints.padCenter == true);
    CHECK(hints.maintainAspectRatio == false);
}

void test_extractPreprocessingHints_atl400Style() {
    // Real metadata strings from ATL400-inspection-1-v1.onnx.
    onnx::ModelProto model;
    auto* pixelNorm = model.add_metadata_props();
    pixelNorm->set_key("pixel_normalization");
    pixelNorm->set_value("{'min_value': 0, 'max_value': 255}");
    auto* padding = model.add_metadata_props();
    padding->set_key("padding");
    padding->set_value("{'type': 'constant', 'position': 'top_left', 'fill': 0.0}");

    const OnnxPreprocessingHints hints = extractPreprocessingHints(model);
    CHECK(approxEqual(hints.inputScale, 1.0f));
    CHECK(approxEqual(hints.padFill, 0.0f));
    CHECK(hints.padCenter == false);
}

void test_extractPreprocessingHints_detracStyle() {
    // Real metadata strings from detrac_tiny_int.onnx.
    onnx::ModelProto model;
    auto* pixelNorm = model.add_metadata_props();
    pixelNorm->set_key("pixel_normalization");
    pixelNorm->set_value("{'enabled': True, 'value': 255}");
    auto* padding = model.add_metadata_props();
    padding->set_key("padding");
    padding->set_value("{'type': 'constant', 'position': 'center', 'fill': 0.0}");
    auto* maintainAspect = model.add_metadata_props();
    maintainAspect->set_key("maintain_aspect_ratio");
    maintainAspect->set_value("True");

    const OnnxPreprocessingHints hints = extractPreprocessingHints(model);
    CHECK(approxEqual(hints.inputScale, 1.0f));
    CHECK(approxEqual(hints.padFill, 0.0f));
    CHECK(hints.padCenter == true);
    CHECK(hints.maintainAspectRatio == true);
}

void test_extractPreprocessingHints_disabledPixelNormalizationKeepsDefault() {
    onnx::ModelProto model;
    auto* pixelNorm = model.add_metadata_props();
    pixelNorm->set_key("pixel_normalization");
    pixelNorm->set_value("{'enabled': False, 'value': 255}");

    const OnnxPreprocessingHints hints = extractPreprocessingHints(model);
    CHECK(approxEqual(hints.inputScale, 1.0f / 255.0f));
}

void test_hwcBgrToNchwFloat_correctChannelOrderAndScale() {
    // 1x2 BGR image: pixel(0,0)=(B10,G20,R30), pixel(0,1)=(B40,G50,R60).
    cv::Mat bgr(1, 2, CV_8UC3);
    bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);

    const auto chw = hwcBgrToNchwFloat(bgr, 2.0f);
    CHECK(chw.size() == 6);
    // R plane (channel 0): 30*2, 60*2
    CHECK(approxEqual(chw[0], 60.0f));
    CHECK(approxEqual(chw[1], 120.0f));
    // G plane (channel 1): 20*2, 50*2
    CHECK(approxEqual(chw[2], 40.0f));
    CHECK(approxEqual(chw[3], 100.0f));
    // B plane (channel 2): 10*2, 40*2
    CHECK(approxEqual(chw[4], 20.0f));
    CHECK(approxEqual(chw[5], 80.0f));
}

} // namespace

int main() {
    test_computeLetterboxTransform_wideFrame();
    test_computeLetterboxTransform_tallFrame();
    test_computeLetterboxTransform_nonSquareTarget();
    test_computeIoU_identical();
    test_computeIoU_disjoint();
    test_computeIoU_partialOverlap();
    test_decodeYoloOutput_confidenceAndNms();
    test_computeBoxAgreement_matchesOverlappingSameClass();
    test_decodeClassificationOutput_sortsDescending();
    test_decodeClassificationOutput_rejectsBadSum();
    test_decodeClassificationOutput_rejectsWrongShape();
    test_extractInputShape_valid();
    test_extractInputShape_rejectsDynamicDim();
    test_extractInputShape_rejectsNoInputs();
    test_extractClassNames_valid();
    test_extractClassNames_missingReturnsEmpty();
    test_extractClassNames_malformedReturnsEmpty();
    test_extractClassNames_categoriesFormat();
    test_extractClassNames_prefersNamesOverCategories();
    test_extractPreprocessingHints_defaultsWhenAbsent();
    test_extractPreprocessingHints_atl400Style();
    test_extractPreprocessingHints_detracStyle();
    test_extractPreprocessingHints_disabledPixelNormalizationKeepsDefault();
    test_hwcBgrToNchwFloat_correctChannelOrderAndScale();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

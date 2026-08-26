#include "manager/anomaly_inference.hpp"
#include "manager/batch_evaluation.hpp"
#include "manager/classification_inference.hpp"
#include "manager/classification_metrics.hpp"
#include "manager/detection_metrics.hpp"
#include "manager/embedding_matching.hpp"
#include "manager/label_assistant.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/onnx_metadata.hpp"
#include "manager/onnx_runtime_env.hpp"
#include "manager/yolo_inference.hpp"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <utility>

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

void test_unletterboxMask_centeredPadding_cropsAndResizesBack() {
    // 100x50 source into a 100x100 canvas: scale=1, padX=0, padY=25 --
    // the "real" content occupies rows [25,75) of the 100x100 mask.
    const LetterboxTransform transform = computeLetterboxTransform(100, 50, 100, 100);
    cv::Mat mask(100, 100, CV_32F, cv::Scalar(0.0f));
    mask(cv::Rect(0, 25, 100, 50)).setTo(cv::Scalar(5.0f));

    const cv::Mat result = unletterboxMask(mask, transform, 100, 50);
    CHECK(result.rows == 50);
    CHECK(result.cols == 100);
    CHECK(approxEqual(result.at<float>(0, 0), 5.0f));
    CHECK(approxEqual(result.at<float>(49, 99), 5.0f));
}

void test_unletterboxMask_topLeftPadding_cropsFromOrigin() {
    const LetterboxTransform transform = computeLetterboxTransform(100, 50, 100, 100, /*centerPadding=*/false);
    cv::Mat mask(100, 100, CV_32F, cv::Scalar(9.0f));
    mask(cv::Rect(0, 50, 100, 50)).setTo(cv::Scalar(-1.0f));  // padding region, should be cropped away

    const cv::Mat result = unletterboxMask(mask, transform, 100, 50);
    CHECK(result.rows == 50);
    CHECK(result.cols == 100);
    CHECK(approxEqual(result.at<float>(0, 0), 9.0f));
    CHECK(approxEqual(result.at<float>(49, 0), 9.0f));
}

void test_unletterboxMask_resizesBackToOriginalDimensions() {
    // 200x100 source into a 100x100 canvas: scale=0.5, padX=0, padY=25.
    const LetterboxTransform transform = computeLetterboxTransform(200, 100, 100, 100);
    cv::Mat mask(100, 100, CV_32F, cv::Scalar(3.0f));

    const cv::Mat result = unletterboxMask(mask, transform, 200, 100);
    CHECK(result.rows == 100);
    CHECK(result.cols == 200);
    CHECK(approxEqual(result.at<float>(50, 100), 3.0f));
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

void test_scoreAnomalyHeatmap_maxReductionAndNormalization() {
    cv::Mat heatmap(4, 4, CV_32F, cv::Scalar(0.2f));
    heatmap.at<float>(2, 2) = 0.8f;

    const AnomalyScore result = scoreAnomalyHeatmap(heatmap, /*scoreMin=*/0.0f, /*scoreMax=*/1.0f, /*threshold=*/0.5f);
    CHECK(approxEqual(result.rawScore, 0.8f));
    CHECK(approxEqual(result.score, 0.8f));
    CHECK(result.isAnomalous == true);
}

void test_scoreAnomalyHeatmap_belowThresholdIsNotAnomalous() {
    cv::Mat heatmap(4, 4, CV_32F, cv::Scalar(0.3f));
    const AnomalyScore result = scoreAnomalyHeatmap(heatmap, 0.0f, 1.0f, 0.5f);
    CHECK(approxEqual(result.score, 0.3f));
    CHECK(result.isAnomalous == false);
}

void test_scoreAnomalyHeatmap_nonTrivialRangeAndClamping() {
    cv::Mat heatmap(2, 2, CV_32F, cv::Scalar(0.5f));
    // (0.5 - 0.2) / (0.6 - 0.2) = 0.75
    const AnomalyScore result = scoreAnomalyHeatmap(heatmap, 0.2f, 0.6f, 0.5f);
    CHECK(approxEqual(result.score, 0.75f));

    cv::Mat aboveMax(2, 2, CV_32F, cv::Scalar(10.0f));
    CHECK(approxEqual(scoreAnomalyHeatmap(aboveMax, 0.0f, 1.0f, 0.5f).score, 1.0f));

    cv::Mat belowMin(2, 2, CV_32F, cv::Scalar(-10.0f));
    CHECK(approxEqual(scoreAnomalyHeatmap(belowMin, 0.0f, 1.0f, 0.5f).score, 0.0f));
}

void test_scoreAnomalyHeatmap_degenerateRangeClampsRawScoreDirectly() {
    cv::Mat heatmap(2, 2, CV_32F, cv::Scalar(0.7f));
    const AnomalyScore result = scoreAnomalyHeatmap(heatmap, /*scoreMin=*/1.0f, /*scoreMax=*/1.0f, 0.5f);
    CHECK(approxEqual(result.score, 0.7f));
}

void test_scoreAnomalyHeatmap_emptyHeatmapReturnsDefault() {
    const AnomalyScore result = scoreAnomalyHeatmap(cv::Mat(), 0.0f, 1.0f, 0.5f);
    CHECK(approxEqual(result.rawScore, 0.0f));
    CHECK(approxEqual(result.score, 0.0f));
    CHECK(result.isAnomalous == false);
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

void test_extractAnomalyScoreHints_defaultsWhenAbsent() {
    onnx::ModelProto model;
    const AnomalyScoreHints hints = extractAnomalyScoreHints(model);
    CHECK(hints.present == false);
    CHECK(approxEqual(hints.minimum, 0.0f));
    CHECK(approxEqual(hints.maximum, 1.0f));
    CHECK(approxEqual(hints.threshold, 0.5f));
}

void test_extractAnomalyScoreHints_vadStyle() {
    // Real metadata strings from meifus_inspection_3_v1.onnx (anomalib EfficientAD export).
    onnx::ModelProto model;
    auto* vadParams = model.add_metadata_props();
    vadParams->set_key("vad_params");
    vadParams->set_value("{'minimum': 0.0, 'maximum': 1.0}");
    auto* autoThreshold = model.add_metadata_props();
    autoThreshold->set_key("auto_threshold");
    autoThreshold->set_value("0.5");

    const AnomalyScoreHints hints = extractAnomalyScoreHints(model);
    CHECK(hints.present == true);
    CHECK(approxEqual(hints.minimum, 0.0f));
    CHECK(approxEqual(hints.maximum, 1.0f));
    CHECK(approxEqual(hints.threshold, 0.5f));
}

void test_extractAnomalyScoreHints_nonTrivialRange() {
    onnx::ModelProto model;
    auto* vadParams = model.add_metadata_props();
    vadParams->set_key("vad_params");
    vadParams->set_value("{'minimum': -2.5, 'maximum': 10.0}");

    const AnomalyScoreHints hints = extractAnomalyScoreHints(model);
    CHECK(hints.present == true);
    CHECK(approxEqual(hints.minimum, -2.5f));
    CHECK(approxEqual(hints.maximum, 10.0f));
    CHECK(approxEqual(hints.threshold, 0.5f));  // auto_threshold absent -- default kept
}

void test_extractAnomalyScoreHints_malformedVadParamsIgnored() {
    onnx::ModelProto model;
    auto* vadParams = model.add_metadata_props();
    vadParams->set_key("vad_params");
    vadParams->set_value("{'minimum': 'not_a_number', 'maximum': 1.0}");

    const AnomalyScoreHints hints = extractAnomalyScoreHints(model);
    CHECK(hints.present == false);
    CHECK(approxEqual(hints.minimum, 0.0f));
    CHECK(approxEqual(hints.maximum, 1.0f));
}

void test_extractTaskHint_present() {
    onnx::ModelProto model;
    auto* task = model.add_metadata_props();
    task->set_key("task");
    task->set_value("vad");

    CHECK(extractTaskHint(model) == "vad");
}

void test_extractTaskHint_absentReturnsEmpty() {
    onnx::ModelProto model;
    CHECK(extractTaskHint(model).empty());
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

void test_applyChannelNormalization_noOpWithDefaultMeanStd() {
    std::vector<float> nchw = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    const std::vector<float> original = nchw;
    applyChannelNormalization(nchw, 2, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    for (size_t i = 0; i < nchw.size(); ++i) {
        CHECK(approxEqual(nchw[i], original[i]));
    }
}

void test_applyChannelNormalization_appliesPerChannelMeanAndStd() {
    // height=1, width=2 -> planeSize=2, 3 channels, 6 values total, all 0.5.
    std::vector<float> nchw = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    const std::array<float, 3> mean = {0.48145466f, 0.4578275f, 0.40821073f};
    const std::array<float, 3> stddev = {0.26862954f, 0.26130258f, 0.27577711f};
    applyChannelNormalization(nchw, 1, 2, mean, stddev);

    CHECK(approxEqual(nchw[0], (0.5f - mean[0]) / stddev[0]));
    CHECK(approxEqual(nchw[1], (0.5f - mean[0]) / stddev[0]));
    CHECK(approxEqual(nchw[2], (0.5f - mean[1]) / stddev[1]));
    CHECK(approxEqual(nchw[3], (0.5f - mean[1]) / stddev[1]));
    CHECK(approxEqual(nchw[4], (0.5f - mean[2]) / stddev[2]));
    CHECK(approxEqual(nchw[5], (0.5f - mean[2]) / stddev[2]));
}

void test_buildCategoryPrototypes_singleCategoryAveragesEmbeddings() {
    const std::vector<LabeledEmbedding> labeled = {
        LabeledEmbedding{"cat", {1.0f, 0.0f, 0.0f}},
        LabeledEmbedding{"cat", {0.0f, 1.0f, 0.0f}},
    };
    const auto prototypes = buildCategoryPrototypes(labeled);
    CHECK(prototypes.size() == 1);
    CHECK(prototypes[0].label == "cat");
    // mean of (1,0,0) and (0,1,0) is (0.5,0.5,0), L2-normalized -> (0.7071, 0.7071, 0)
    CHECK(approxEqual(prototypes[0].prototype[0], 0.70710678f));
    CHECK(approxEqual(prototypes[0].prototype[1], 0.70710678f));
    CHECK(approxEqual(prototypes[0].prototype[2], 0.0f));
}

void test_buildCategoryPrototypes_emptyInputProducesEmptyResult() {
    const auto prototypes = buildCategoryPrototypes({});
    CHECK(prototypes.empty());
}

void test_buildCategoryPrototypes_multipleCategoriesSortedByLabel() {
    const std::vector<LabeledEmbedding> labeled = {
        LabeledEmbedding{"dog", {0.0f, 1.0f}},
        LabeledEmbedding{"cat", {1.0f, 0.0f}},
    };
    const auto prototypes = buildCategoryPrototypes(labeled);
    CHECK(prototypes.size() == 2);
    CHECK(prototypes[0].label == "cat");
    CHECK(prototypes[1].label == "dog");
}

void test_matchToNearestPrototype_picksClosestBySimilarity() {
    const std::vector<CategoryPrototype> prototypes = {
        CategoryPrototype{"cat", {1.0f, 0.0f}},
        CategoryPrototype{"dog", {0.0f, 1.0f}},
    };
    const MatchResult result = matchToNearestPrototype({0.9f, 0.1f}, prototypes);
    CHECK(result.predictedLabel == "cat");
    CHECK(result.confidence > 0.9f);
}

void test_matchToNearestPrototype_picksHigherSimilarityCandidate() {
    const std::vector<CategoryPrototype> prototypes = {
        CategoryPrototype{"cat", {1.0f, 0.0f}},
        CategoryPrototype{"dog", {0.0f, 1.0f}},
    };
    const MatchResult result = matchToNearestPrototype({0.6f, 0.4f}, prototypes);
    CHECK(result.predictedLabel == "cat");
}

void test_runLabelAssistant_splitsSeedAndDraftsRemainder() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_label_assistant";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "seed_cat.jpg").string(), img);
    cv::imwrite((tempDir / "seed_dog.jpg").string(), img);
    cv::imwrite((tempDir / "unlabeled_1.jpg").string(), img);

    LabelStudioImportResult groundTruth;
    ImageGroundTruth gtCat;
    gtCat.imageFilename = "seed_cat.jpg";
    gtCat.classificationLabel = "cat";
    gtCat.hasClassificationAnnotation = true;
    groundTruth.images.push_back(gtCat);

    ImageGroundTruth gtDog;
    gtDog.imageFilename = "seed_dog.jpg";
    gtDog.classificationLabel = "dog";
    gtDog.hasClassificationAnnotation = true;
    groundTruth.images.push_back(gtDog);

    // All three test frames are pixel-identical (plain black), so every
    // fake embedding is identical too -- the point of this test is the
    // seed/draft split and the pipeline shape, not realistic matching.
    const auto embed = [](const cv::Mat& frame) -> std::vector<float> {
        return {static_cast<float>(cv::mean(frame)[0]), 0.0f};
    };

    const auto result = runLabelAssistant(tempDir.string(), groundTruth, embed);

    CHECK(result.error.empty());
    CHECK(result.seedImagesUsed == 2);
    CHECK(result.categoriesFound == 2);
    CHECK(result.imagesDrafted == 1);
    CHECK(result.drafts.size() == 1);
    CHECK(result.drafts[0].imageFilename == "unlabeled_1.jpg");
    // Identical embeddings make "cat" and "dog" prototypes identical too,
    // a tied cosine similarity -- buildCategoryPrototypes iterates labels
    // alphabetically (std::map), and matchToNearestPrototype keeps the
    // first strictly-greater match, so "cat" wins the tie deterministically.
    CHECK(result.drafts[0].predictedLabel == "cat");

    fs::remove_all(tempDir, ec);
}

void test_runLabelAssistant_noSeedGroundTruthIsHardError() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_label_assistant_no_seed";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "unlabeled.jpg").string(), img);

    LabelStudioImportResult emptyGroundTruth;
    const auto embed = [](const cv::Mat&) -> std::vector<float> { return {1.0f}; };

    const auto result = runLabelAssistant(tempDir.string(), emptyGroundTruth, embed);

    CHECK(!result.error.empty());
    CHECK(result.drafts.empty());

    fs::remove_all(tempDir, ec);
}

void test_runLabelAssistant_emptyFolderIsError() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_label_assistant_empty";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    LabelStudioImportResult groundTruth;
    const auto embed = [](const cv::Mat&) -> std::vector<float> { return {1.0f}; };
    const auto result = runLabelAssistant(tempDir.string(), groundTruth, embed);

    CHECK(!result.error.empty());
    CHECK(result.drafts.empty());

    fs::remove_all(tempDir, ec);
}

void test_runLabelAssistant_progressCallback() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_label_assistant_progress";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "seed.jpg").string(), img);
    cv::imwrite((tempDir / "unlabeled_a.jpg").string(), img);
    cv::imwrite((tempDir / "unlabeled_b.jpg").string(), img);

    LabelStudioImportResult groundTruth;
    ImageGroundTruth gt;
    gt.imageFilename = "seed.jpg";
    gt.classificationLabel = "cat";
    gt.hasClassificationAnnotation = true;
    groundTruth.images.push_back(gt);

    const auto embed = [](const cv::Mat&) -> std::vector<float> { return {1.0f}; };

    std::vector<int> completedValues;
    int lastTotal = 0;
    const auto onProgress = [&](int completed, int total) {
        completedValues.push_back(completed);
        lastTotal = total;
    };

    const auto result = runLabelAssistant(tempDir.string(), groundTruth, embed, onProgress);

    CHECK(result.error.empty());
    CHECK(lastTotal == 3);
    CHECK(completedValues.size() == 3);
    for (size_t i = 0; i < completedValues.size(); ++i) {
        CHECK(completedValues[i] == static_cast<int>(i) + 1);
    }

    fs::remove_all(tempDir, ec);
}

void test_runLabelAssistant_cancellation() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_label_assistant_cancel";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "seed.jpg").string(), img);
    cv::imwrite((tempDir / "unlabeled.jpg").string(), img);

    LabelStudioImportResult groundTruth;
    ImageGroundTruth gt;
    gt.imageFilename = "seed.jpg";
    gt.classificationLabel = "cat";
    gt.hasClassificationAnnotation = true;
    groundTruth.images.push_back(gt);

    std::atomic<bool> cancelRequested{true};
    const auto embed = [](const cv::Mat&) -> std::vector<float> { return {1.0f}; };

    const auto result = runLabelAssistant(tempDir.string(), groundTruth, embed, nullptr, &cancelRequested);

    CHECK(result.error.empty());
    CHECK(result.drafts.empty());

    fs::remove_all(tempDir, ec);
}

void test_parseLabelStudioExport_detectionBox() {
    // Real schema, verified against Label Studio's own docs.
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image": "http://x/opensource/label-studio/1.jpg"},
        "annotations": [{"result": [{
            "type": "rectanglelabels",
            "value": {"x": 50.8, "y": 5.87, "width": 12.4, "height": 10.46, "rotation": 0, "rectanglelabels": ["Moonwalker"]},
            "original_width": 600, "original_height": 403
        }]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].imageFilename == "1.jpg");
    CHECK(result.images[0].boxes.size() == 1);
    CHECK(result.images[0].boxes[0].className == "Moonwalker");
    CHECK(result.images[0].boxes[0].box.x == 305);
    CHECK(result.images[0].boxes[0].box.y == 24);
    CHECK(result.images[0].boxes[0].box.width == 74);
    CHECK(result.images[0].boxes[0].box.height == 42);
    CHECK(result.images[0].hasDetectionAnnotation == true);
    CHECK(result.images[0].hasClassificationAnnotation == false);
}

void test_parseLabelStudioExport_classification() {
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image_value": "http://x/dog_or_cat.jpg"},
        "annotations": [{"result": [{
            "type": "choices",
            "value": {"choices": ["Dog"]}
        }]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].imageFilename == "dog_or_cat.jpg");
    CHECK(result.images[0].classificationLabel == "Dog");
    CHECK(result.images[0].boxes.empty());
    CHECK(result.images[0].hasClassificationAnnotation == true);
    CHECK(result.images[0].hasDetectionAnnotation == false);
}

void test_parseLabelStudioExport_bothDetectionAndClassificationSameImage() {
    // Label Studio projects can annotate an image with more than one task
    // type at once -- both result kinds can appear in the same task's
    // result array, on the same image.
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image": "both.jpg"},
        "annotations": [{"result": [
            {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rotation": 0, "rectanglelabels": ["A"]}, "original_width": 100, "original_height": 100},
            {"type": "choices", "value": {"choices": ["Dog"]}}
        ]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].boxes.size() == 1);
    CHECK(result.images[0].boxes[0].className == "A");
    CHECK(result.images[0].classificationLabel == "Dog");
    CHECK(result.images[0].hasDetectionAnnotation == true);
    CHECK(result.images[0].hasClassificationAnnotation == true);
}

void test_parseLabelStudioExport_multipleBoxesSameImage() {
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image": "a.jpg"},
        "annotations": [{"result": [
            {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rotation": 0, "rectanglelabels": ["A"]}, "original_width": 100, "original_height": 100},
            {"type": "rectanglelabels", "value": {"x": 50, "y": 50, "width": 10, "height": 10, "rotation": 0, "rectanglelabels": ["B"]}, "original_width": 100, "original_height": 100}
        ]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].boxes.size() == 2);
    CHECK(result.images[0].boxes[0].className == "A");
    CHECK(result.images[0].boxes[1].className == "B");
}

void test_parseLabelStudioExport_rotatedBoxSkipped() {
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"image": "a.jpg"},
        "annotations": [{"result": [
            {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rotation": 15, "rectanglelabels": ["A"]}, "original_width": 100, "original_height": 100}
        ]}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.skippedCount == 1);
    // The image entry itself is still recorded, just with no boxes.
    CHECK(result.images.size() == 1);
    CHECK(result.images[0].boxes.empty());
    // It was still annotated for detection -- just with a box that couldn't
    // be used, not "never labeled at all".
    CHECK(result.images[0].hasDetectionAnnotation == true);
}

void test_parseLabelStudioExport_noRecognizableImageFieldSkipped() {
    const auto tasks = nlohmann::json::parse(R"([{
        "data": {"unrelated": "value"},
        "annotations": [{"result": []}]
    }])");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(result.error.empty());
    CHECK(result.skippedCount == 1);
    CHECK(result.images.empty());
}

void test_parseLabelStudioExport_nonArrayTopLevelIsHardError() {
    const auto tasks = nlohmann::json::parse(R"({"not": "an array"})");

    const auto result = parseLabelStudioExport(tasks);
    CHECK(!result.error.empty());
    CHECK(result.images.empty());
}

void test_buildLabelStudioPredictionsJson_singleDraft() {
    const std::vector<DraftClassificationLabel> drafts = {
        DraftClassificationLabel{"042.jpg", "defect", 0.41f}
    };
    const auto parsed = nlohmann::json::parse(
        buildLabelStudioPredictionsJson(drafts, "project_x/images", "choice", "image"));

    CHECK(parsed.is_array());
    CHECK(parsed.size() == 1);
    CHECK(parsed[0]["data"]["image"].get<std::string>() == "/data/local-files/?d=project_x/images/042.jpg");
    CHECK(parsed[0]["predictions"].size() == 1);
    CHECK(approxEqual(parsed[0]["predictions"][0]["score"].get<float>(), 0.41f));
    const auto& result = parsed[0]["predictions"][0]["result"][0];
    CHECK(result["from_name"].get<std::string>() == "choice");
    CHECK(result["to_name"].get<std::string>() == "image");
    CHECK(result["type"].get<std::string>() == "choices");
    CHECK(result["value"]["choices"][0].get<std::string>() == "defect");
}

void test_buildLabelStudioPredictionsJson_multipleDraftsInOrder() {
    const std::vector<DraftClassificationLabel> drafts = {
        DraftClassificationLabel{"a.jpg", "cat", 0.9f},
        DraftClassificationLabel{"b.jpg", "dog", 0.5f},
    };
    const auto parsed = nlohmann::json::parse(buildLabelStudioPredictionsJson(drafts, "imgs", "choice", "image"));

    CHECK(parsed.size() == 2);
    CHECK(parsed[0]["data"]["image"].get<std::string>() == "/data/local-files/?d=imgs/a.jpg");
    CHECK(parsed[0]["predictions"][0]["result"][0]["value"]["choices"][0].get<std::string>() == "cat");
    CHECK(parsed[1]["data"]["image"].get<std::string>() == "/data/local-files/?d=imgs/b.jpg");
    CHECK(parsed[1]["predictions"][0]["result"][0]["value"]["choices"][0].get<std::string>() == "dog");
}

void test_buildLabelStudioPredictionsJson_emptyDraftsIsValidEmptyArray() {
    const auto parsed = nlohmann::json::parse(
        buildLabelStudioPredictionsJson({}, "imgs", "choice", "image"));

    CHECK(parsed.is_array());
    CHECK(parsed.empty());
}

void test_buildLabelStudioPredictionsJson_prefixTrailingSlashNormalized() {
    const std::vector<DraftClassificationLabel> drafts = {DraftClassificationLabel{"a.jpg", "cat", 0.5f}};
    const auto withSlash = nlohmann::json::parse(buildLabelStudioPredictionsJson(drafts, "imgs/", "choice", "image"));
    const auto withoutSlash = nlohmann::json::parse(buildLabelStudioPredictionsJson(drafts, "imgs", "choice", "image"));

    CHECK(withSlash[0]["data"]["image"].get<std::string>() == "/data/local-files/?d=imgs/a.jpg");
    CHECK(withoutSlash[0]["data"]["image"].get<std::string>() == "/data/local-files/?d=imgs/a.jpg");
}

void test_buildLabelStudioPredictionsJson_emptyPrefixNoExtraSlash() {
    const std::vector<DraftClassificationLabel> drafts = {DraftClassificationLabel{"a.jpg", "cat", 0.5f}};
    const auto parsed = nlohmann::json::parse(buildLabelStudioPredictionsJson(drafts, "", "choice", "image"));

    CHECK(parsed[0]["data"]["image"].get<std::string>() == "/data/local-files/?d=a.jpg");
}

void test_computeDetectionMetrics_onePositiveOneFalsePositive() {
    // Hand-verified: 2 images, each with 1 ground-truth "cat" box.
    // Image 0's prediction matches perfectly (conf 0.9) -> TP.
    // Image 1's prediction doesn't overlap at all (conf 0.8) -> FP.
    // precision=[1.0, 0.5], recall=[0.5, 0.5] -> AP = (0.5-0)*1.0 + (0.5-0.5)*0.5 = 0.5.
    std::vector<DetectionEvaluationItem> items(2);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.9f}};
    items[1].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[1].predictions = {Detection{cv::Rect(100, 100, 10, 10), 0, "cat", 0.8f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(metrics.perClass.size() == 1);
    CHECK(metrics.perClass[0].className == "cat");
    CHECK(approxEqual(metrics.perClass[0].averagePrecision, 0.5f));
    CHECK(approxEqual(metrics.meanAveragePrecision, 0.5f));
    // Overall precision/recall derived from the final TP/FP counts:
    // precision = 1/(1+1) = 0.5, recall = 1/2 = 0.5.
    CHECK(metrics.perClass[0].truePositives == 1);
    CHECK(metrics.perClass[0].falsePositives == 1);
    CHECK(metrics.perClass[0].numGroundTruth == 2);
}

void test_computeDetectionMetrics_perfectDetectorIsApOne() {
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.99f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(approxEqual(metrics.meanAveragePrecision, 1.0f));
}

void test_computeDetectionMetrics_noOverlapIsApZero() {
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {Detection{cv::Rect(100, 100, 10, 10), 0, "cat", 0.9f}};

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(approxEqual(metrics.meanAveragePrecision, 0.0f));
}

void test_computeDetectionMetrics_excludesClassWithNoGroundTruth() {
    // "cat" has a perfect match (AP=1.0); "dog" has predictions but zero
    // ground truth, so it must not dilute the mean.
    std::vector<DetectionEvaluationItem> items(1);
    items[0].groundTruth = {GroundTruthBox{cv::Rect(0, 0, 10, 10), "cat"}};
    items[0].predictions = {
        Detection{cv::Rect(0, 0, 10, 10), 0, "cat", 0.9f},
        Detection{cv::Rect(50, 50, 10, 10), 1, "dog", 0.7f},
    };

    const DetectionMetrics metrics = computeDetectionMetrics(items, 0.5f);
    CHECK(metrics.perClass.size() == 2);
    CHECK(approxEqual(metrics.meanAveragePrecision, 1.0f));
}

void test_computeClassificationMetrics_accuracyAndConfusionMatrix() {
    std::vector<ClassificationEvaluationItem> items = {
        {"cat", "cat"},   // correct
        {"dog", "cat"},   // wrong
        {"dog", "dog"},   // correct
    };

    const ClassificationMetrics metrics = computeClassificationMetrics(items);
    CHECK(metrics.totalEvaluated == 3);
    CHECK(approxEqual(metrics.accuracy, 2.0f / 3.0f));
    CHECK(metrics.confusionMatrix.at("cat").at("cat") == 1);
    CHECK(metrics.confusionMatrix.at("cat").at("dog") == 1);
    CHECK(metrics.confusionMatrix.at("dog").at("dog") == 1);
}

void test_computeClassificationMetrics_emptyIsZeroAccuracy() {
    const ClassificationMetrics metrics = computeClassificationMetrics({});
    CHECK(metrics.totalEvaluated == 0);
    CHECK(approxEqual(metrics.accuracy, 0.0f));
}

void test_computeTimingStats_meanMedianP95() {
    const TimingStats stats = computeTimingStats({10.0, 20.0, 30.0, 40.0, 50.0});
    CHECK(stats.count == 5);
    CHECK(approxEqual(static_cast<float>(stats.meanMs), 30.0f));
    CHECK(approxEqual(static_cast<float>(stats.medianMs), 30.0f));
    CHECK(approxEqual(static_cast<float>(stats.p95Ms), 48.0f));  // interpolated between rank 3 (40) and 4 (50)
}

void test_runDetectionBatchEvaluation_scansAndCrossReferences() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat imgA(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat imgB(10, 10, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::imwrite((tempDir / "a.jpg").string(), imgA);
    cv::imwrite((tempDir / "b.jpg").string(), imgB);

    LabelStudioImportResult groundTruth;
    ImageGroundTruth gtA;
    gtA.imageFilename = "a.jpg";
    gtA.boxes.push_back(GroundTruthBox{cv::Rect(1, 1, 2, 2), "cat"});
    gtA.hasDetectionAnnotation = true;
    groundTruth.images.push_back(gtA);
    // "b.jpg" intentionally has no ground-truth entry.

    const auto result = runDetectionBatchEvaluation(tempDir.string(), &groundTruth, [](const cv::Mat&) {
        std::vector<Detection> dets;
        dets.push_back(Detection{cv::Rect(1, 1, 2, 2), 0, "cat", 0.9f});
        return dets;
    });

    CHECK(result.error.empty());
    CHECK(result.imagesFound == 2);
    CHECK(result.imagesWithGroundTruth == 1);
    CHECK(result.images.size() == 2);
    CHECK(result.images[0].imageFilename == "a.jpg");
    CHECK(result.images[0].hasGroundTruth == true);
    CHECK(result.images[0].groundTruthBoxes.size() == 1);
    CHECK(result.images[1].imageFilename == "b.jpg");
    CHECK(result.images[1].hasGroundTruth == false);
    CHECK(result.timing.count == 2);

    const auto evalItems = toDetectionEvaluationItems(result);
    CHECK(evalItems.size() == 1);  // only a.jpg has ground truth

    fs::remove_all(tempDir, ec);
}

void test_batchEvaluation_ignoresGroundTruthOfTheOtherTaskType() {
    // A mixed-type dataset: "detection_only.jpg" was only ever annotated
    // with a box, "classification_only.jpg" was only ever annotated with a
    // class choice. Running detection evaluation must not treat the
    // classification-only image as having (empty) ground truth, and
    // running classification evaluation must not treat the detection-only
    // image as having an (empty-string) ground-truth label -- either would
    // silently corrupt the resulting metrics with a phantom false
    // positive/negative for an image that was simply never labeled for
    // that task type.
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_mixed_types";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "detection_only.jpg").string(), img);
    cv::imwrite((tempDir / "classification_only.jpg").string(), img);

    LabelStudioImportResult groundTruth;
    ImageGroundTruth gtDetectionOnly;
    gtDetectionOnly.imageFilename = "detection_only.jpg";
    gtDetectionOnly.boxes.push_back(GroundTruthBox{cv::Rect(1, 1, 2, 2), "cat"});
    gtDetectionOnly.hasDetectionAnnotation = true;
    groundTruth.images.push_back(gtDetectionOnly);

    ImageGroundTruth gtClassificationOnly;
    gtClassificationOnly.imageFilename = "classification_only.jpg";
    gtClassificationOnly.classificationLabel = "cat";
    gtClassificationOnly.hasClassificationAnnotation = true;
    groundTruth.images.push_back(gtClassificationOnly);

    const auto detectionResult =
        runDetectionBatchEvaluation(tempDir.string(), &groundTruth, [](const cv::Mat&) {
            return std::vector<Detection>{Detection{cv::Rect(1, 1, 2, 2), 0, "cat", 0.9f}};
        });
    CHECK(detectionResult.imagesWithGroundTruth == 1);
    for (const auto& image : detectionResult.images) {
        CHECK(image.hasGroundTruth == (image.imageFilename == "detection_only.jpg"));
    }

    const auto classificationResult =
        runClassificationBatchEvaluation(tempDir.string(), &groundTruth, [](const cv::Mat&) {
            return std::vector<ClassPrediction>{ClassPrediction{0, "cat", 0.9f}};
        });
    CHECK(classificationResult.imagesWithGroundTruth == 1);
    for (const auto& image : classificationResult.images) {
        CHECK(image.hasGroundTruth == (image.imageFilename == "classification_only.jpg"));
    }

    fs::remove_all(tempDir, ec);
}

void test_sampleImageFiles_zeroSampleSizeReturnsAllUnchanged() {
    std::vector<std::filesystem::path> files = {"a.jpg", "b.jpg", "c.jpg"};
    const auto sampled = sampleImageFiles(files, 0, 42);
    CHECK(sampled.size() == 3);
    CHECK(sampled[0] == "a.jpg");
    CHECK(sampled[1] == "b.jpg");
    CHECK(sampled[2] == "c.jpg");
}

void test_sampleImageFiles_sampleSizeAtOrAboveCountReturnsAllUnchanged() {
    std::vector<std::filesystem::path> files = {"a.jpg", "b.jpg"};
    const auto sampled = sampleImageFiles(files, 5, 42);
    CHECK(sampled.size() == 2);
}

void test_sampleImageFiles_reducesToRequestedCountAndStaysSorted() {
    std::vector<std::filesystem::path> files = {"a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"};
    const auto sampled = sampleImageFiles(files, 2, 42);
    CHECK(sampled.size() == 2);
    CHECK(sampled[0] < sampled[1]);  // re-sorted, not left in shuffled order
    // Every sampled entry must have come from the original set.
    for (const auto& f : sampled) {
        CHECK(std::find(files.begin(), files.end(), f) != files.end());
    }
}

void test_sampleImageFiles_sameSeedIsDeterministic() {
    // Two models compared against the same batch run must see the exact
    // same random subset -- this is what makes that possible.
    std::vector<std::filesystem::path> files = {"a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"};
    const auto sampledA = sampleImageFiles(files, 3, 1234);
    const auto sampledB = sampleImageFiles(files, 3, 1234);
    CHECK(sampledA.size() == sampledB.size());
    for (size_t i = 0; i < sampledA.size(); ++i) {
        CHECK(sampledA[i] == sampledB[i]);
    }
}

void test_runDetectionBatchEvaluation_sampleSizeLimitsImagesEvaluated() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_sample";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < 10; ++i) {
        cv::imwrite((tempDir / (std::to_string(i) + ".jpg")).string(), img);
    }

    const auto result = runDetectionBatchEvaluation(
        tempDir.string(), nullptr, [](const cv::Mat&) { return std::vector<Detection>{}; },
        nullptr, nullptr, /*sampleSize=*/3, /*sampleSeed=*/7);

    CHECK(result.error.empty());
    CHECK(result.totalFilesInFolder == 10);
    CHECK(result.imagesFound == 3);
    CHECK(result.images.size() == 3);

    fs::remove_all(tempDir, ec);
}

void test_runDetectionBatchEvaluation_emptyFolderIsError() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_empty";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    const auto result = runDetectionBatchEvaluation(
        tempDir.string(), nullptr, [](const cv::Mat&) { return std::vector<Detection>{}; });

    CHECK(!result.error.empty());
    CHECK(result.images.empty());

    fs::remove_all(tempDir, ec);
}

void test_runDetectionBatchEvaluation_progressCallback() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_progress";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "a.jpg").string(), img);
    cv::imwrite((tempDir / "b.jpg").string(), img);
    cv::imwrite((tempDir / "c.jpg").string(), img);

    std::vector<std::pair<int, int>> progressCalls;
    const auto result = runDetectionBatchEvaluation(
        tempDir.string(), nullptr,
        [](const cv::Mat&) { return std::vector<Detection>{}; },
        [&](int completed, int total) { progressCalls.push_back({completed, total}); });

    CHECK(result.error.empty());
    CHECK(result.imagesFound == 3);
    CHECK(progressCalls.size() == 3);
    CHECK(progressCalls[0] == std::make_pair(1, 3));
    CHECK(progressCalls[1] == std::make_pair(2, 3));
    CHECK(progressCalls[2] == std::make_pair(3, 3));

    fs::remove_all(tempDir, ec);
}

void test_runDetectionBatchEvaluation_cancellation() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_cancel";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "a.jpg").string(), img);
    cv::imwrite((tempDir / "b.jpg").string(), img);
    cv::imwrite((tempDir / "c.jpg").string(), img);

    std::atomic<bool> cancelRequested{true};
    const auto result = runDetectionBatchEvaluation(
        tempDir.string(), nullptr,
        [](const cv::Mat&) { return std::vector<Detection>{}; },
        nullptr, &cancelRequested);

    CHECK(result.error.empty());
    CHECK(result.images.empty());
    CHECK(result.imagesFound == 0);

    fs::remove_all(tempDir, ec);
}

void test_runClassificationBatchEvaluation_progressCallback() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_cls_progress";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "a.jpg").string(), img);
    cv::imwrite((tempDir / "b.jpg").string(), img);

    std::vector<std::pair<int, int>> progressCalls;
    const auto result = runClassificationBatchEvaluation(
        tempDir.string(), nullptr,
        [](const cv::Mat&) { return std::vector<ClassPrediction>{}; },
        [&](int completed, int total) { progressCalls.push_back({completed, total}); });

    CHECK(result.error.empty());
    CHECK(result.imagesFound == 2);
    CHECK(progressCalls.size() == 2);
    CHECK(progressCalls[0] == std::make_pair(1, 2));
    CHECK(progressCalls[1] == std::make_pair(2, 2));

    fs::remove_all(tempDir, ec);
}

void test_runClassificationBatchEvaluation_cancellation() {
    namespace fs = std::filesystem;
    const fs::path tempDir = fs::temp_directory_path() / "vision_app_test_batch_eval_cls_cancel";
    std::error_code ec;
    fs::create_directories(tempDir, ec);

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imwrite((tempDir / "a.jpg").string(), img);
    cv::imwrite((tempDir / "b.jpg").string(), img);

    std::atomic<bool> cancelRequested{true};
    const auto result = runClassificationBatchEvaluation(
        tempDir.string(), nullptr,
        [](const cv::Mat&) { return std::vector<ClassPrediction>{}; },
        nullptr, &cancelRequested);

    CHECK(result.error.empty());
    CHECK(result.images.empty());
    CHECK(result.imagesFound == 0);

    fs::remove_all(tempDir, ec);
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
    test_applyChannelNormalization_noOpWithDefaultMeanStd();
    test_applyChannelNormalization_appliesPerChannelMeanAndStd();
    test_buildCategoryPrototypes_singleCategoryAveragesEmbeddings();
    test_buildCategoryPrototypes_emptyInputProducesEmptyResult();
    test_buildCategoryPrototypes_multipleCategoriesSortedByLabel();
    test_matchToNearestPrototype_picksClosestBySimilarity();
    test_matchToNearestPrototype_picksHigherSimilarityCandidate();
    test_runLabelAssistant_splitsSeedAndDraftsRemainder();
    test_runLabelAssistant_noSeedGroundTruthIsHardError();
    test_runLabelAssistant_emptyFolderIsError();
    test_runLabelAssistant_progressCallback();
    test_runLabelAssistant_cancellation();
    test_parseLabelStudioExport_detectionBox();
    test_parseLabelStudioExport_classification();
    test_parseLabelStudioExport_bothDetectionAndClassificationSameImage();
    test_parseLabelStudioExport_multipleBoxesSameImage();
    test_parseLabelStudioExport_rotatedBoxSkipped();
    test_parseLabelStudioExport_noRecognizableImageFieldSkipped();
    test_parseLabelStudioExport_nonArrayTopLevelIsHardError();
    test_buildLabelStudioPredictionsJson_singleDraft();
    test_buildLabelStudioPredictionsJson_multipleDraftsInOrder();
    test_buildLabelStudioPredictionsJson_emptyDraftsIsValidEmptyArray();
    test_buildLabelStudioPredictionsJson_prefixTrailingSlashNormalized();
    test_buildLabelStudioPredictionsJson_emptyPrefixNoExtraSlash();
    test_computeDetectionMetrics_onePositiveOneFalsePositive();
    test_computeDetectionMetrics_perfectDetectorIsApOne();
    test_computeDetectionMetrics_noOverlapIsApZero();
    test_computeDetectionMetrics_excludesClassWithNoGroundTruth();
    test_computeClassificationMetrics_accuracyAndConfusionMatrix();
    test_computeClassificationMetrics_emptyIsZeroAccuracy();
    test_computeTimingStats_meanMedianP95();
    test_runDetectionBatchEvaluation_scansAndCrossReferences();
    test_batchEvaluation_ignoresGroundTruthOfTheOtherTaskType();
    test_sampleImageFiles_zeroSampleSizeReturnsAllUnchanged();
    test_sampleImageFiles_sampleSizeAtOrAboveCountReturnsAllUnchanged();
    test_sampleImageFiles_reducesToRequestedCountAndStaysSorted();
    test_sampleImageFiles_sameSeedIsDeterministic();
    test_runDetectionBatchEvaluation_sampleSizeLimitsImagesEvaluated();
    test_runDetectionBatchEvaluation_emptyFolderIsError();
    test_runDetectionBatchEvaluation_progressCallback();
    test_runDetectionBatchEvaluation_cancellation();
    test_runClassificationBatchEvaluation_progressCallback();
    test_runClassificationBatchEvaluation_cancellation();
    test_extractAnomalyScoreHints_defaultsWhenAbsent();
    test_extractAnomalyScoreHints_vadStyle();
    test_extractAnomalyScoreHints_nonTrivialRange();
    test_extractAnomalyScoreHints_malformedVadParamsIgnored();
    test_extractTaskHint_present();
    test_extractTaskHint_absentReturnsEmpty();
    test_unletterboxMask_centeredPadding_cropsAndResizesBack();
    test_unletterboxMask_topLeftPadding_cropsFromOrigin();
    test_unletterboxMask_resizesBackToOriginalDimensions();
    test_scoreAnomalyHeatmap_maxReductionAndNormalization();
    test_scoreAnomalyHeatmap_belowThresholdIsNotAnomalous();
    test_scoreAnomalyHeatmap_nonTrivialRangeAndClamping();
    test_scoreAnomalyHeatmap_degenerateRangeClampsRawScoreDirectly();
    test_scoreAnomalyHeatmap_emptyHeatmapReturnsDefault();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

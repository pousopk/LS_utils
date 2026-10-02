#include "manager/benchmark_filters.hpp"

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

bool approxEqual(float a, float b, float tolerance = 0.001f) {
    return std::fabs(a - b) <= tolerance;
}

} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

Detection det(const char* className, cv::Rect box, float confidence = 0.9f, float rotationDegrees = 0.0f) {
    return Detection{box, 0, className, confidence, rotationDegrees};
}

GroundTruthBox gt(const char* className, cv::Rect box, float rotationDegrees = 0.0f) {
    return GroundTruthBox{box, className, rotationDegrees};
}

BenchmarkImageResult detImage(std::vector<Detection> detections, std::vector<GroundTruthBox> truth = {}, bool hasGroundTruth = true) {
    BenchmarkImageResult image;
    image.detections = std::move(detections);
    image.groundTruthBoxes = std::move(truth);
    image.hasGroundTruth = hasGroundTruth;
    return image;
}

// truth == nullptr -> no ground truth for this image.
// predicted == nullptr -> no prediction.
BenchmarkImageResult clsImage(const char* predicted, const char* truth, float probability = 0.9f) {
    BenchmarkImageResult image;
    if (predicted != nullptr) {
        image.predictions.push_back(ClassPrediction{0, predicted, probability});
    }
    if (truth != nullptr) {
        image.groundTruthLabel = truth;
        image.hasGroundTruth = true;
    }
    return image;
}

const cv::Rect kBoxA(0, 0, 10, 10);
const cv::Rect kBoxFar(100, 100, 10, 10);

void test_matchDetections_exactMatch() {
    const DetectionMatch m = matchDetections({det("cat", kBoxA)}, {gt("cat", kBoxA)});
    CHECK(m.predictionMatched.size() == 1 && m.predictionMatched[0]);
    CHECK(m.groundTruthMatched.size() == 1 && m.groundTruthMatched[0]);
}

void test_matchDetections_wrongClassIsFpAndMissed() {
    const DetectionMatch m = matchDetections({det("dog", kBoxA)}, {gt("cat", kBoxA)});
    CHECK(!m.predictionMatched[0]);
    CHECK(!m.groundTruthMatched[0]);
}

void test_matchDetections_iouAtThresholdMatches() {
    // (0,0,10,5) vs (0,0,10,10): intersection 50 / union 100 = exactly 0.5.
    const DetectionMatch m = matchDetections({det("cat", cv::Rect(0, 0, 10, 5))}, {gt("cat", kBoxA)});
    CHECK(m.predictionMatched[0]);
    CHECK(m.groundTruthMatched[0]);
}

void test_matchDetections_iouBelowThresholdDoesNotMatch() {
    // (0,0,10,4) vs (0,0,10,10): IoU 0.4.
    const DetectionMatch m = matchDetections({det("cat", cv::Rect(0, 0, 10, 4))}, {gt("cat", kBoxA)});
    CHECK(!m.predictionMatched[0]);
    CHECK(!m.groundTruthMatched[0]);
}

void test_matchDetections_secondPredictionOnSameBoxIsFp() {
    const DetectionMatch m = matchDetections({det("cat", kBoxA), det("cat", kBoxA)}, {gt("cat", kBoxA)});
    CHECK(m.predictionMatched[0]);
    CHECK(!m.predictionMatched[1]);
    CHECK(m.groundTruthMatched[0]);
}

void test_matchDetections_rotationAware() {
    // Same unrotated rect, prediction rotated 90 degrees: true IoU is 0.
    const DetectionMatch m =
        matchDetections({det("widget", cv::Rect(0, 0, 40, 10), 0.9f, 90.0f)}, {gt("widget", cv::Rect(0, 0, 40, 10))});
    CHECK(!m.predictionMatched[0]);
    CHECK(!m.groundTruthMatched[0]);
}

void test_matchDetections_emptyInputs() {
    const DetectionMatch none = matchDetections({}, {});
    CHECK(none.predictionMatched.empty() && none.groundTruthMatched.empty());
    const DetectionMatch onlyGt = matchDetections({}, {gt("cat", kBoxA)});
    CHECK(onlyGt.groundTruthMatched.size() == 1 && !onlyGt.groundTruthMatched[0]);
    const DetectionMatch onlyPred = matchDetections({det("cat", kBoxFar)}, {});
    CHECK(onlyPred.predictionMatched.size() == 1 && !onlyPred.predictionMatched[0]);
}

void test_confidence_detectionBasis() {
    const BenchmarkImageResult image = detImage({det("a", kBoxA, 0.2f), det("b", kBoxA, 0.6f), det("c", kBoxA, 1.0f)});
    const auto mode = ModelTask::Detection;
    CHECK(approxEqual(*benchmarkImageConfidence(mode, &image, BenchmarkConfidenceBasis::Mean), 0.6f));
    CHECK(approxEqual(*benchmarkImageConfidence(mode, &image, BenchmarkConfidenceBasis::Lowest), 0.2f));
    CHECK(approxEqual(*benchmarkImageConfidence(mode, &image, BenchmarkConfidenceBasis::Highest), 1.0f));
}

void test_confidence_noBoxesOrNullIsNullopt() {
    const BenchmarkImageResult empty = detImage({});
    CHECK(!benchmarkImageConfidence(ModelTask::Detection, &empty, BenchmarkConfidenceBasis::Mean));
    CHECK(!benchmarkImageConfidence(ModelTask::Detection, nullptr, BenchmarkConfidenceBasis::Mean));
}

void test_confidence_classificationIgnoresBasis() {
    const BenchmarkImageResult image = clsImage("cat", nullptr, 0.7f);
    CHECK(approxEqual(*benchmarkImageConfidence(ModelTask::Classification, &image, BenchmarkConfidenceBasis::Lowest), 0.7f));
    CHECK(approxEqual(*benchmarkImageConfidence(ModelTask::Classification, &image, BenchmarkConfidenceBasis::Highest), 0.7f));
}

void test_confidence_anomalyScore() {
    BenchmarkImageResult image;
    image.anomalyResult = BenchmarkAnomalyResult{3.0f, 0.4f, false};
    CHECK(approxEqual(*benchmarkImageConfidence(ModelTask::Anomaly, &image, BenchmarkConfidenceBasis::Mean), 0.4f));
}

void test_sortConfidence_twoSlotRule() {
    const auto mode = ModelTask::Detection;
    const auto basis = BenchmarkConfidenceBasis::Mean;
    const BenchmarkImageResult low = detImage({det("a", kBoxA, 0.3f)});
    const BenchmarkImageResult high = detImage({det("a", kBoxA, 0.8f)});
    const BenchmarkImageResult empty = detImage({});
    CHECK(approxEqual(benchmarkImageSortConfidence(mode, &low, &high, basis), 0.3f));
    CHECK(approxEqual(benchmarkImageSortConfidence(mode, &high, &low, basis), 0.3f));
    CHECK(approxEqual(benchmarkImageSortConfidence(mode, &high, nullptr, basis), 0.8f));
    CHECK(approxEqual(benchmarkImageSortConfidence(mode, &empty, &high, basis), 0.8f));
    CHECK(approxEqual(benchmarkImageSortConfidence(mode, &empty, nullptr, basis), 0.0f));
}

bool passes(ModelTask mode, const BenchmarkImageResult* a, const BenchmarkImageResult* b, const BenchmarkImageFilters& f, bool hasGroundTruth = true) {
    return benchmarkImagePassesFilters(mode, hasGroundTruth, a, b, f);
}

void test_filters_defaultsPassEverything() {
    const BenchmarkImageFilters f;
    const BenchmarkImageResult emptyDet = detImage({}, {}, false);
    const BenchmarkImageResult wrongDet = detImage({det("dog", kBoxA)}, {gt("cat", kBoxA)});
    const BenchmarkImageResult wrongCls = clsImage("dog", "cat");
    BenchmarkImageResult anomaly;
    anomaly.anomalyResult = BenchmarkAnomalyResult{1.0f, 0.9f, true};
    CHECK(passes(ModelTask::Detection, &emptyDet, nullptr, f));
    CHECK(passes(ModelTask::Detection, &wrongDet, &emptyDet, f));
    CHECK(passes(ModelTask::Classification, &wrongCls, nullptr, f));
    CHECK(passes(ModelTask::Anomaly, &anomaly, nullptr, f));
}

void test_filters_detectionErrorTypes() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult correct = detImage({det("cat", kBoxA)}, {gt("cat", kBoxA)});
    const BenchmarkImageResult fpOnly = detImage({det("cat", kBoxA), det("cat", kBoxFar)}, {gt("cat", kBoxA)});
    const BenchmarkImageResult missedOnly = detImage({}, {gt("cat", kBoxA)});

    BenchmarkImageFilters f;
    f.errorFilter = BenchmarkErrorFilter::AnyError;
    CHECK(!passes(mode, &correct, nullptr, f));
    CHECK(passes(mode, &fpOnly, nullptr, f));
    CHECK(passes(mode, &missedOnly, nullptr, f));

    f.errorFilter = BenchmarkErrorFilter::FalsePositives;
    CHECK(passes(mode, &fpOnly, nullptr, f));
    CHECK(!passes(mode, &missedOnly, nullptr, f));

    f.errorFilter = BenchmarkErrorFilter::Missed;
    CHECK(!passes(mode, &fpOnly, nullptr, f));
    CHECK(passes(mode, &missedOnly, nullptr, f));
}

void test_filters_errorEitherSlotAndNoGroundTruth() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult correct = detImage({det("cat", kBoxA)}, {gt("cat", kBoxA)});
    const BenchmarkImageResult missed = detImage({}, {gt("cat", kBoxA)});
    const BenchmarkImageResult noGt = detImage({det("cat", kBoxA)}, {}, false);

    BenchmarkImageFilters f;
    f.errorFilter = BenchmarkErrorFilter::AnyError;
    CHECK(passes(mode, &correct, &missed, f));   // B has the error
    CHECK(!passes(mode, &noGt, nullptr, f));     // a slot without GT never has an "error"
    CHECK(passes(mode, &noGt, nullptr, f, false));  // run without GT: filter ignored
}

void test_filters_classificationMisclassified() {
    const auto mode = ModelTask::Classification;
    const BenchmarkImageResult right = clsImage("cat", "cat");
    const BenchmarkImageResult wrong = clsImage("dog", "cat");
    const BenchmarkImageResult noPrediction = clsImage(nullptr, "cat");

    BenchmarkImageFilters f;
    f.errorFilter = BenchmarkErrorFilter::AnyError;
    CHECK(!passes(mode, &right, nullptr, f));
    CHECK(passes(mode, &wrong, nullptr, f));
    CHECK(!passes(mode, &noPrediction, nullptr, f));

    f.errorFilter = BenchmarkErrorFilter::Missed;  // treated as AnyError in classification
    CHECK(passes(mode, &wrong, nullptr, f));
}

void test_filters_confidenceFilterUsesBasis() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult image = detImage({det("a", kBoxA, 0.2f), det("b", kBoxA, 0.9f)});  // mean 0.55

    BenchmarkImageFilters f;
    f.confidence.mode = ThresholdMode::LessThan;
    f.confidence.threshold = 0.5f;
    CHECK(!passes(mode, &image, nullptr, f));  // mean 0.55 is not < 0.5
    f.confidenceBasis = BenchmarkConfidenceBasis::Lowest;
    CHECK(passes(mode, &image, nullptr, f));   // lowest 0.2 < 0.5

    f.confidence.mode = ThresholdMode::GreaterThan;
    f.confidence.threshold = 0.8f;
    CHECK(!passes(mode, &image, nullptr, f));  // lowest 0.2
    f.confidenceBasis = BenchmarkConfidenceBasis::Highest;
    CHECK(passes(mode, &image, nullptr, f));   // highest 0.9

    const BenchmarkImageResult empty = detImage({});
    CHECK(!passes(mode, &empty, nullptr, f));  // no value never passes a confidence filter
}

void test_filters_detectionPresence() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult some = detImage({det("a", kBoxA)});
    const BenchmarkImageResult none = detImage({});

    BenchmarkImageFilters f;
    f.detections.presence = Presence::Lacks;
    CHECK(passes(mode, &none, nullptr, f));
    CHECK(!passes(mode, &some, nullptr, f));  // null slot B must not count as "no detections"
    CHECK(passes(mode, &some, &none, f));

    f.detections.presence = Presence::Has;
    CHECK(passes(mode, &some, nullptr, f));
    CHECK(!passes(mode, &none, nullptr, f));

    const BenchmarkImageResult cls = clsImage("cat", nullptr);
    CHECK(passes(ModelTask::Classification, &cls, nullptr, f));  // ignored outside detection
}

void test_classFilter_aloneMatchesGroundTruthOrPrediction() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult gtOnly = detImage({}, {gt("scratch", kBoxA)});
    const BenchmarkImageResult predOnly = detImage({det("scratch", kBoxA)}, {}, false);
    const BenchmarkImageResult other = detImage({det("dent", kBoxA)}, {gt("dent", kBoxA)});

    BenchmarkImageFilters f;
    f.cls.className = "scratch";
    CHECK(passes(mode, &gtOnly, nullptr, f));
    CHECK(passes(mode, &predOnly, nullptr, f));
    CHECK(!passes(mode, &other, nullptr, f));
    CHECK(passes(mode, &other, &predOnly, f));  // either slot

    const BenchmarkImageResult clsGt = clsImage("dog", "cat");
    f.cls.className = "cat";
    CHECK(passes(ModelTask::Classification, &clsGt, nullptr, f));
    f.cls.className = "dog";
    CHECK(passes(ModelTask::Classification, &clsGt, nullptr, f));
    f.cls.className = "bird";
    CHECK(!passes(ModelTask::Classification, &clsGt, nullptr, f));
}

void test_classFilter_scopesDetectionErrors() {
    const auto mode = ModelTask::Detection;
    // scratch correctly found, dent missed, one false-positive "dent" far away.
    const BenchmarkImageResult image = detImage(
        {det("scratch", kBoxA), det("dent", kBoxFar)},
        {gt("scratch", kBoxA), gt("dent", cv::Rect(50, 50, 10, 10))});

    BenchmarkImageFilters f;
    f.cls.className = "scratch";
    f.errorFilter = BenchmarkErrorFilter::AnyError;
    CHECK(!passes(mode, &image, nullptr, f));  // no scratch errors
    f.cls.className = "dent";
    CHECK(passes(mode, &image, nullptr, f));
    f.errorFilter = BenchmarkErrorFilter::Missed;
    CHECK(passes(mode, &image, nullptr, f));
    f.errorFilter = BenchmarkErrorFilter::FalsePositives;
    CHECK(passes(mode, &image, nullptr, f));
    f.cls.className = "scratch";
    CHECK(!passes(mode, &image, nullptr, f));
}

void test_classFilter_scopesMisclassificationEitherDirection() {
    const auto mode = ModelTask::Classification;
    const BenchmarkImageResult catAsDog = clsImage("dog", "cat");
    BenchmarkImageFilters f;
    f.errorFilter = BenchmarkErrorFilter::AnyError;
    f.cls.className = "cat";
    CHECK(passes(mode, &catAsDog, nullptr, f));   // true label
    f.cls.className = "dog";
    CHECK(passes(mode, &catAsDog, nullptr, f));   // predicted label
    f.cls.className = "bird";
    CHECK(!passes(mode, &catAsDog, nullptr, f));
}

void test_collectClassNames() {
    BenchmarkResult a;
    a.images.push_back(detImage({det("dent", kBoxA)}, {gt("scratch", kBoxA)}));
    BenchmarkResult b;
    b.images.push_back(detImage({det("crack", kBoxA), det("dent", kBoxA)}));
    const std::vector<std::string> names = collectBenchmarkClassNames(ModelTask::Detection, a, b);
    CHECK((names == std::vector<std::string>{"crack", "dent", "scratch"}));

    BenchmarkResult c;
    c.images.push_back(clsImage("dog", "cat"));
    c.images.push_back(clsImage("bird", nullptr));
    const std::vector<std::string> clsNames =
        collectBenchmarkClassNames(ModelTask::Classification, c, BenchmarkResult{});
    CHECK((clsNames == std::vector<std::string>{"bird", "cat", "dog"}));

    CHECK(collectBenchmarkClassNames(ModelTask::Anomaly, a, b).empty());
}

void test_disagree_classification() {
    const auto mode = ModelTask::Classification;
    const BenchmarkImageResult cat = clsImage("cat", nullptr);
    const BenchmarkImageResult cat2 = clsImage("cat", nullptr, 0.4f);
    const BenchmarkImageResult dog = clsImage("dog", nullptr);
    const BenchmarkImageResult none = clsImage(nullptr, nullptr);

    BenchmarkImageFilters f;
    f.modelsDisagreeOnly = true;
    CHECK(!passes(mode, &cat, &cat2, f, false));  // no ground truth needed
    CHECK(passes(mode, &cat, &dog, f, false));
    CHECK(passes(mode, &cat, &none, f, false));   // one side predicted nothing
    CHECK(!passes(mode, &none, &none, f, false));
    CHECK(!passes(mode, &cat, nullptr, f, false)); // needs both slots
}

void test_disagree_detection() {
    const auto mode = ModelTask::Detection;
    const BenchmarkImageResult one = detImage({det("cat", kBoxA, 0.9f)}, {}, false);
    const BenchmarkImageResult oneLowConf = detImage({det("cat", kBoxA, 0.3f)}, {}, false);
    const BenchmarkImageResult extra = detImage({det("cat", kBoxA), det("cat", kBoxFar)}, {}, false);
    const BenchmarkImageResult otherClass = detImage({det("dog", kBoxA)}, {}, false);
    const BenchmarkImageResult empty = detImage({}, {}, false);

    BenchmarkImageFilters f;
    f.modelsDisagreeOnly = true;
    CHECK(!passes(mode, &one, &oneLowConf, f, false));  // confidence difference alone isn't disagreement
    CHECK(passes(mode, &extra, &one, f, false));        // extra box in A
    CHECK(passes(mode, &one, &extra, f, false));        // extra box in B
    CHECK(passes(mode, &one, &otherClass, f, false));
    CHECK(!passes(mode, &empty, &empty, f, false));
    CHECK(!passes(mode, &one, nullptr, f, false));
}

void test_confusionCell() {
    const auto mode = ModelTask::Classification;
    const BenchmarkImageResult catAsDog = clsImage("dog", "cat");
    const BenchmarkImageResult catAsCat = clsImage("cat", "cat");
    const BenchmarkImageResult noGt = clsImage("dog", nullptr);

    BenchmarkImageFilters f;
    f.confusionCell = BenchmarkConfusionCellFilter{"cat", "dog", 0};
    CHECK(passes(mode, &catAsDog, nullptr, f));
    CHECK(!passes(mode, &catAsCat, nullptr, f));
    CHECK(!passes(mode, &noGt, nullptr, f));
    CHECK(!passes(mode, &catAsCat, &catAsDog, f));  // only the clicked slot (A) counts

    f.confusionCell->slotIndex = 1;
    CHECK(passes(mode, &catAsCat, &catAsDog, f));
    CHECK(!passes(mode, &catAsDog, nullptr, f));    // slot B absent
}

void test_classPrecisionRecall() {
    ClassAveragePrecision m;
    m.truePositives = 3;
    m.falsePositives = 1;
    m.numGroundTruth = 6;
    CHECK(approxEqual(*classPrecision(m), 0.75f));
    CHECK(approxEqual(*classRecall(m), 0.5f));

    ClassAveragePrecision none;
    CHECK(!classPrecision(none));  // no predictions
    CHECK(!classRecall(none));     // no ground truth

    ClassAveragePrecision allWrong;
    allWrong.falsePositives = 2;
    allWrong.numGroundTruth = 1;
    CHECK(approxEqual(*classPrecision(allWrong), 0.0f));
    CHECK(approxEqual(*classRecall(allWrong), 0.0f));
}

} // namespace

int main() {
    test_matchDetections_exactMatch();
    test_matchDetections_wrongClassIsFpAndMissed();
    test_matchDetections_iouAtThresholdMatches();
    test_matchDetections_iouBelowThresholdDoesNotMatch();
    test_matchDetections_secondPredictionOnSameBoxIsFp();
    test_matchDetections_rotationAware();
    test_matchDetections_emptyInputs();
    test_confidence_detectionBasis();
    test_confidence_noBoxesOrNullIsNullopt();
    test_confidence_classificationIgnoresBasis();
    test_confidence_anomalyScore();
    test_sortConfidence_twoSlotRule();
    test_filters_defaultsPassEverything();
    test_filters_detectionErrorTypes();
    test_filters_errorEitherSlotAndNoGroundTruth();
    test_filters_classificationMisclassified();
    test_filters_confidenceFilterUsesBasis();
    test_filters_detectionPresence();
    test_classFilter_aloneMatchesGroundTruthOrPrediction();
    test_classFilter_scopesDetectionErrors();
    test_classFilter_scopesMisclassificationEitherDirection();
    test_collectClassNames();
    test_disagree_classification();
    test_disagree_detection();
    test_confusionCell();
    test_classPrecisionRecall();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

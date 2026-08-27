#pragma once

// Shared between the live/batch model-evaluation state modules and
// BatchEvalRunConfig -- broken out to its own header so batch_evaluation_worker.hpp
// doesn't need to depend on (and risk circularity with) either state module.
enum class ComparisonTaskMode {
    Detection,
    Classification,
    Anomaly,
};

#pragma once

// What kind of model a slot holds -- shared by Benchmark, Label Assistant
// and the benchmark worker's run config, broken out to its own header so
// benchmark_worker.hpp doesn't depend on (and risk circularity with)
// either state module.
enum class ModelTask {
    Detection,
    Classification,
    Anomaly,
};

// Display name ("Detection", "Classification", "Anomaly").
inline const char* modelTaskName(ModelTask task) {
    switch (task) {
        case ModelTask::Detection:
            return "Detection";
        case ModelTask::Classification:
            return "Classification";
        case ModelTask::Anomaly:
            return "Anomaly";
    }
    return "";
}

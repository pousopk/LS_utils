#pragma once

#include "manager/model_task.hpp"
#include "manager/label_studio_client.hpp"

#include <string>
#include <vector>

struct LabelAssistantControlTagResolution {
    std::string fromName;
    std::string toName;
    std::string error;   // set if no control tag of the required type is found
};

// Pure function: finds the first control tag matching `mode`'s required
// type (RectangleLabels for Detection, Choices for Classification) in
// `controlTags` and returns its from_name/to_name -- "first one wins" if
// a project somehow has more than one, same convention this codebase's
// other control-tag lookups already use. Replaces the old
// fetchLabelStudioLabelingConfig-based auto-fetch this window used to do
// for the same information: that made its own network call and its own
// isDetection-driven guess; the project's controlTags are already known
// (from the shared project data every tab fetches once), so this is now
// a plain local lookup with no network call of its own. Sets `error`
// (leaving fromName/toName empty) if the project has no control tag of
// the type `mode` needs -- e.g. a Classification-only project selected
// while `mode` is Detection. `mode` is never ModelTask::Anomaly
// here -- this window's UI only offers Detection/Classification.
LabelAssistantControlTagResolution resolveLabelAssistantControlTag(
    const std::vector<LabelStudioControlTag>& controlTags, ModelTask mode);

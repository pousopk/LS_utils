#pragma once

#include "manager/label_studio_client.hpp"
#include "manager/label_studio_import.hpp"

#include <GLFW/glfw3.h>

#include <optional>
#include <string>
#include <vector>

struct BoxLabelEditorState {
    std::string fromName;
    std::string toName;
    std::vector<std::string> availableLabels;
    std::vector<DraftDetectionBox> boxes;
    int selectedBoxIndex = -1;   // -1 = none selected
    bool dirty = false;
};

struct ChoiceLabelEditorState {
    std::string fromName;
    std::string toName;
    std::vector<std::string> availableLabels;
    std::optional<std::string> selectedLabel;   // single-select for phase 1
    bool dirty = false;
};

enum class LabelingTaskLoadState {
    NotLoaded,
    Loading,
    Loaded,
    Failed,
};

enum class LabelingUnsavedPromptAction {
    None,
    SwitchTask,
    CloseWindow,
};

struct LabelingState {
    // Connection fields, own to this window -- matching the existing
    // per-window convention (Label Assistant, Timestamp Search, Batch
    // Eval each hold their own copy too); unifying this is explicitly
    // deferred, see the design doc's Non-goals.
    std::string labelStudioBaseUrl;
    int labelStudioProjectId = 0;
    std::string labelStudioApiToken;

    std::string lastAutoFetchKey;   // (baseUrl, projectId, apiToken), see updateLabelingState
    std::string configStatus;
    LabelStudioProjectConfig projectConfig;

    std::vector<LabelStudioTaskSummary> taskList;
    std::string taskListError;
    bool taskListLoading = false;

    int focusTaskId = -1;      // set by a cross-window "Label this" jump, consumed once
    int selectedTaskId = -1;
    LabelingTaskLoadState taskLoadState = LabelingTaskLoadState::NotLoaded;
    std::string taskLoadError;
    int imageWidth = 0;
    int imageHeight = 0;
    GLuint imageTexture = 0;
    std::optional<int> currentAnnotationId;   // set if the loaded task already had a real annotation

    std::optional<BoxLabelEditorState> boxEditor;
    std::optional<ChoiceLabelEditorState> choiceEditor;

    bool submitInProgress = false;
    std::string submitStatus;

    bool unsavedPromptOpen = false;
    LabelingUnsavedPromptAction unsavedPromptAction = LabelingUnsavedPromptAction::None;
    int unsavedPromptPendingTaskId = -1;   // valid when unsavedPromptAction == SwitchTask
};

// (Re)creates state.boxEditor/state.choiceEditor from state.projectConfig's
// control tags -- a RectangleLabels tag activates boxEditor, a Choices tag
// activates choiceEditor, both may be active at once (this is what makes
// phase 2 -- a project configured for both -- work with no extra code).
// Clears any existing editor state (fresh availableLabels, empty
// boxes/selectedLabel) -- callers that need to preserve in-progress edits
// across a config refresh must not call this while an editor is dirty.
void resetLabelingEditorsFromConfig(LabelingState& state);

// Seeds the active editor(s) from a task's existing annotation if
// present (also sets state.currentAnnotationId), else falls back to its
// prediction if present, else leaves the editors blank. Does not touch
// dirty flags left at their default (false) -- this represents freshly
// loaded, unedited state. Must be called after resetLabelingEditorsFromConfig
// has established which editors are active for the current project.
void applyTaskDetailToEditors(LabelingState& state, const LabelStudioTaskDetail& detail);

// True if either active editor has unsaved edits.
bool anyEditorDirty(const LabelingState& state);

// Builds the combined `result` array for submission by concatenating
// buildDetectionPredictionResult's result (if boxEditor is active and has
// at least one box) with buildClassificationPredictionResult's result (if
// choiceEditor is active and has a selection). An editor with nothing to
// contribute (no boxes / no selection) contributes nothing -- an empty
// combined result is valid (e.g. explicitly labeling an image as having
// zero objects).
nlohmann::json buildCombinedAnnotationResult(const LabelingState& state, int imageWidth, int imageHeight);

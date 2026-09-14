#pragma once

#include "manager/label_studio_client.hpp"
#include "manager/label_studio_import.hpp"
#include "manager/label_studio_session.hpp"
#include "manager/labeling_worker.hpp"

#include <GLFW/glfw3.h>

#include <opencv2/imgproc.hpp>

#include <array>
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

    // Class assigned to the next box drawn from empty space, chosen via
    // the label picker buttons or a number-key shortcut. Defaults to the
    // first available label (see resetLabelingEditorsFromConfig).
    std::string pendingNewBoxLabel;
};

struct LabelColor {
    unsigned char r = 200;
    unsigned char g = 200;
    unsigned char b = 200;
};

// Pure function: deterministically maps a class name to a distinct,
// visually stable color -- the same name always produces the same color
// (hash of the name -> hue, fixed saturation/value), with no per-project
// color configuration needed. Kept free of ImGui so it's testable in
// isolation; the widget layer converts LabelColor to whatever pixel
// format it needs (e.g. IM_COL32).
LabelColor colorForClassName(const std::string& className);

// Pure function: rotates `point` clockwise around `pivot` by `degrees`, in
// this codebase's image-space convention (x right, y down) -- clockwise as
// a viewer would see it on screen. The single primitive every other
// rotation-aware box function is built from.
cv::Point2f rotatePointClockwise(cv::Point2f point, cv::Point2f pivot, float degrees);

// Pure function: the 4 corners of `box` (top-left, top-right,
// bottom-right, bottom-left, in that order) after applying
// `rotationDegrees` around the box's own pivot (box.x, box.y) -- the
// top-left corner is always exactly (box.x, box.y) itself (rotating a
// point around itself is a no-op); the other 3 corners move. Used for
// drawing the rotated outline and for placing handles at their true
// screen positions.
std::array<cv::Point2f, 4> rotatedBoxCorners(const cv::Rect& box, float rotationDegrees);

// Pure function: true if `point` (image space) falls inside `box` once
// rotation is accounted for -- inverse-rotates `point` into the box's own
// unrotated frame, then does the ordinary axis-aligned contains-check
// there. Replaces a plain `box.contains(point)` hit-test for a box that
// may be rotated.
bool rotatedBoxContainsPoint(const cv::Rect& box, float rotationDegrees, cv::Point2f point);

// Pure function: the diagonally-opposite corner of `box` from
// `handleIndex` (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right --
// same convention as labeling_window.cpp's existing handle-index scheme),
// rotated into image space around the box's own pivot. This is the corner
// that must stay fixed on screen while `handleIndex` is dragged; callers
// capture it once at drag-start and hold it fixed for the whole resize.
cv::Point2f rotatedHandleAnchorPoint(int handleIndex, const cv::Rect& box, float rotationDegrees);

// Pure function: resizes a box by dragging one corner to follow
// `currentMouseImage`, given `anchorImage` (the screen-fixed opposite
// corner, from rotatedHandleAnchorPoint, captured once at drag-start) and
// `rotationDegrees` (fixed for the whole resize -- resizing never changes
// the angle). Computes the result in the box's own rotated coordinate
// frame (inverse-rotating the mouse position around the anchor, applying
// the same min/abs logic an unrotated editor would, then rotating the
// result back). Reduces exactly to the plain axis-aligned min/abs resize
// when rotationDegrees == 0.0f. rotationDegrees itself is unchanged by
// this function -- only x/y/width/height are returned.
cv::Rect resizeRotatedBox(float rotationDegrees, cv::Point2f anchorImage, cv::Point2f currentMouseImage);

struct RotatedBoxAngleDrag {
    cv::Rect box;              // updated x, y (width/height unchanged from originalBox)
    float rotationDegrees = 0.0f;
};

// Pure function: rotates `originalBox` (currently at `startRotationDegrees`)
// so its on-screen center stays fixed while its angle changes by
// (currentAngleRadians - startAngleRadians) -- both angles are
// atan2(dy, dx) from the box's own center to a point (the mouse), in
// image space, using this file's clockwise convention (consistent with
// rotatePointClockwise). x/y (Label Studio's own pivot) are recomputed so
// the center stays put -- only this function's caller decides what "the
// box's center" and the two angles are (see labeling_window.cpp's rotate
// drag); this function is pure algebra given them.
RotatedBoxAngleDrag rotateBoxAroundCenter(
    const cv::Rect& originalBox, float startRotationDegrees, float startAngleRadians, float currentAngleRadians);

struct ChoiceLabelEditorState {
    std::string fromName;
    std::string toName;
    std::vector<std::string> availableLabels;
    std::optional<std::string> selectedLabel;   // single-select for phase 1
    bool dirty = false;
};

struct BrushLabelEditorState {
    std::string fromName;
    std::string toName;
    std::vector<std::string> availableLabels;
    std::vector<DraftBrushRegion> regions;
    int selectedRegionIndex = -1;   // -1 = none selected
    bool dirty = false;

    // Class assigned to the next mask started from scratch, chosen via
    // the shared label picker buttons. Defaults to the first available
    // label (see resetLabelingEditorsFromConfig).
    std::string pendingNewMaskLabel;

    bool eraseMode = false;
    float brushRadius = 12.0f;
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
    LabelingWorker worker;
    std::string scratchFolderPath;   // set once by the window on first open, see Task 8

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
    std::string pendingLocalImagePath;   // set by updateLabelingState once FetchTaskDetail's image download completes
    std::string loadedLocalImagePath;    // the path currently uploaded into imageTexture
    cv::Mat baseImage;   // the currently loaded task's raw image (3-channel BGR), kept so mask edits can be
                         // recomposited without re-reading the file from disk (see drawLabelingWindow's
                         // maskChanged handling, in the widget layer, for the recomposite-and-reupload trigger)
    std::optional<int> currentAnnotationId;   // set if the loaded task already had a real annotation

    std::optional<BoxLabelEditorState> boxEditor;
    std::optional<ChoiceLabelEditorState> choiceEditor;
    std::optional<BrushLabelEditorState> maskEditor;

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

// Blends each region's mask onto a copy of `baseImage` (expected 3-channel
// BGR, matching cv::imread's default), tinted by colorForClassName, at a
// fixed alpha -- for visual display only, never written back to
// annotation data. Regions with an empty mask, or one whose size doesn't
// match baseImage's, are skipped (not an error -- a freshly-created
// region's mask matches the image by construction; a mismatch here would
// only happen from a bug elsewhere, and silently skipping is safer than
// crashing the display). Pure pixel manipulation, no GL/ImGui.
cv::Mat compositeMaskOverlay(const cv::Mat& baseImage, const std::vector<DraftBrushRegion>& regions);

// Called once per main-loop iteration while the Labeling window is open.
// Lazily (re)fetches state.projectConfig whenever
// (session.baseUrl, session.activeProjectId, session.apiToken) changes
// (same lastAutoFetchKey convention as Label Assistant/Timestamp Search),
// calling resetLabelingEditorsFromConfig on a successful fetch. Also polls
// state.worker for a finished job and applies its result: FetchTaskList
// fills state.taskList/taskListError; FetchTaskDetail fills the editors
// (via applyTaskDetailToEditors) and state.imageWidth/imageHeight/
// imageTexture; SubmitAnnotation clears both editors' dirty flags on
// success and sets state.submitStatus either way.
void updateLabelingState(LabelingState& state, const LabelStudioSessionState& session);

// Requests switching the selected task to `taskId`. If anyEditorDirty(state)
// is true, opens the unsaved-changes prompt instead of switching
// immediately (state.unsavedPromptAction = SwitchTask,
// state.unsavedPromptPendingTaskId = taskId); otherwise starts a
// FetchTaskDetail job right away, using session's connection info.
void requestSelectLabelingTask(LabelingState& state, const LabelStudioSessionState& session, int taskId);

// Pure function: finds state.selectedTaskId's position in state.taskList
// and returns the task id `direction` steps away (+1 = next, -1 =
// previous), or std::nullopt if the list is empty, the current task
// isn't in it, or stepping would go past either end. Used for keyboard
// task navigation (Ctrl+Left/Right) -- doesn't itself switch tasks or
// check for unsaved changes, see requestSelectLabelingTask for that.
std::optional<int> nextLabelingTaskId(const LabelingState& state, int direction);

// Starts a SubmitAnnotation job from buildCombinedAnnotationResult(state,
// state.imageWidth, state.imageHeight), passing state.currentAnnotationId
// through (present -> update, absent -> create), using session's
// connection info. Sets state.submitInProgress = true; updateLabelingState
// clears it once the job completes.
void beginSubmitLabelingAnnotation(LabelingState& state, const LabelStudioSessionState& session);

// Unsaved-changes prompt resolution: Discard closes the prompt and
// proceeds with whatever action was pending (switch task / close window)
// without saving.
void confirmDiscardAndSwitchTask(LabelingState& state, const LabelStudioSessionState& session);

// Unsaved-changes prompt resolution: Save submits the current edits first
// (beginSubmitLabelingAnnotation), then proceeds with the pending action
// once the submit completes. Implemented as: close the prompt, submit now,
// and let the caller (the window, Task 11) re-issue the pending
// switch/close after seeing submitInProgress go false -- LabelingState
// itself doesn't schedule follow-up actions across frames.
void confirmSaveAndSwitchTask(LabelingState& state, const LabelStudioSessionState& session);

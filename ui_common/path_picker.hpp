#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// One reusable folder/file picker modal shared by every window. Each window
// owns a PathPickerState plus its own "which field is this for" enum, calls
// openPathPicker from its Browse button, and dispatches on the path
// drawPathPicker returns.

enum class PathPickerMode { Folder, File };

struct PathPickerState {
    PathPickerMode mode = PathPickerMode::File;
    bool openRequested = false;
    std::string title;          // popup title, e.g. "Pick ONNX Model"
    std::string currentDir;     // browser location
    std::string originDir;      // location at open time ("Current" jump); empty if none
    std::string pathInput;      // editable path box
    std::string pathInputError; // inline error, e.g. "Path not found"
    std::string filter;
    std::string selectedFile;   // File mode only
    // Set when selectedFile changes from outside the list (path box, open);
    // the next draw scrolls that row into view and clears it.
    bool scrollToSelection = false;
    // Recents whose folder existed at open time -- filtered once, not per
    // frame, so a slow network mount in the list doesn't stall every frame.
    std::vector<std::filesystem::path> recentChoices;
    // Listing cache, refreshed when currentDir changes (not every frame).
    std::string cachedDir;
    std::vector<std::filesystem::path> cachedDirs;
    std::vector<std::filesystem::path> cachedFiles;
};

// Seeds currentDir/originDir from currentValue via initialPickerDirectory,
// sets title/mode, clears filter/selection/error, invalidates the listing
// cache, and flags the popup to open on the next drawPathPicker. In File
// mode an existing file currentValue is preselected.
void openPathPicker(
    PathPickerState& picker, PathPickerMode mode, std::string title, const std::string& currentValue);

// Draws the modal when open. Returns the chosen path on the confirming frame,
// std::nullopt otherwise. Confirming records the location in recents.
// popupId must be unique per window (e.g. "ModelEvalPicker"); it keeps the
// popup's ImGui ID stable while the title changes between targets.
std::optional<std::filesystem::path> drawPathPicker(PathPickerState& picker, const char* popupId);

// Records a confirmed location in the process-wide recents list (the one
// persisted to imgui.ini). drawPathPicker calls this on confirm.
void rememberPathPickerLocation(const std::filesystem::path& dir);

// Registers the imgui.ini handler that persists recents. Call once after
// ImGui::CreateContext and before the first NewFrame.
void registerPathPickerSettingsHandler();

// ---- State operations (no ImGui context needed) ----

// Moves to dir (cleaned), clearing filter, selection and input error, and
// syncing the path box to the new location.
void navigatePathPicker(PathPickerState& picker, const std::filesystem::path& dir);

// Selects file and syncs the path box to it. Does not change currentDir.
void selectPathPickerFile(PathPickerState& picker, const std::filesystem::path& file);

// A click on a file row: selects it, and returns true (confirm) only for a
// double-click on a file that was already selected before this click.
bool clickPathPickerFile(PathPickerState& picker, const std::filesystem::path& file, bool doubleClick);

// Applies the path box (Enter / Go): navigates, selects a file (navigating
// to its folder first), or sets pathInputError = "Path not found".
void applyPathPickerInput(PathPickerState& picker);

// Re-reads cachedDirs/cachedFiles when currentDir != cachedDir. Files are
// only listed in File mode.
void refreshPathPickerListing(PathPickerState& picker);

// ---- Pure helpers ----

// Makes a relative path absolute (against the working directory), then
// lexically_normal() plus trailing-separator removal (keeping "/"), so
// "/a/b/" and "/a/b" compare equal and parent_path() of either is "/a".
std::filesystem::path cleanPickerPath(const std::filesystem::path& path);

struct InitialPickerDirectory {
    std::filesystem::path dir;
    bool fromCurrentValue = false;
};
// Existing folder -> itself; existing file -> its folder; missing path whose
// parent folder exists -> that parent; empty or otherwise -> default root
// (fromCurrentValue = false).
InitialPickerDirectory initialPickerDirectory(const std::string& currentValue);

enum class PathInputAction { Navigate, SelectFile, NotFound };
struct PathInputResolution {
    PathInputAction action = PathInputAction::NotFound;
    std::filesystem::path path;
};
// Trims whitespace, expands a leading "~" to the default root, resolves
// relative paths against currentDir, cleans the result. Directory ->
// Navigate; regular file in File mode -> SelectFile; else NotFound.
PathInputResolution resolvePathInput(
    PathPickerMode mode, std::string_view text, const std::filesystem::path& currentDir);

inline constexpr std::size_t kMaxRecentLocations = 10;

// Moves dir (cleaned) to the front, deduped, truncated to cap.
void pushRecentLocation(
    std::vector<std::filesystem::path>& recents, const std::filesystem::path& dir,
    std::size_t cap = kMaxRecentLocations);

// ini read path: appends the parsed line at the back (file order is newest
// first), skipping duplicates and anything past kMaxRecentLocations.
void appendRecentFromIniLine(std::vector<std::filesystem::path>& recents, std::string_view line);

// imgui.ini line codec: "Path=<abs path>" <-> path. parse returns nullopt for
// unrecognized/empty lines and strips a trailing '\r'.
std::string formatRecentLine(const std::filesystem::path& dir);
std::optional<std::filesystem::path> parseRecentLine(std::string_view line);

// Recents whose folder still exists, order preserved.
std::vector<std::filesystem::path> existingRecentLocations(const std::vector<std::filesystem::path>& recents);

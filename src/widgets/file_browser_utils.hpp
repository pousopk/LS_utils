#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

std::filesystem::path defaultFileSelectorRoot();
std::filesystem::path normalizeDirectoryOrDefault(const std::filesystem::path& directory);
std::vector<std::filesystem::path> listDirectories(const std::filesystem::path& directory);
std::vector<std::filesystem::path> listFiles(const std::filesystem::path& directory);

// Case-insensitive substring match against the file's name, for the search/filter
// boxes drawn above file-picker lists (mirrors the GenICam feature search).
bool fileNameMatchesFilter(const std::filesystem::path& file, std::string_view filterText);

// Draws a folder explorer (current path, Up, filter box, scrollable subfolder
// list) inside the currently open ImGui window/popup. `explorerDirectory` is
// the browser's current location, normalized and mutated as the user
// navigates. `filterText` is a case-insensitive substring filter applied to
// the subfolder list -- owned by the caller so it persists across frames, and
// reusable by the caller to also filter a sibling file list in the same
// popup.
// If `selectedDirectory` is non-null, a "Use this folder" button is also shown;
// clicking it copies `explorerDirectory` into `*selectedDirectory` and this
// returns true for that frame (so callers can react, e.g. close a dialog).
// Pass nullptr to omit that button for pickers that only need navigation (e.g.
// a directory+file picker with its own file-selection action).
// `listChildId` must be unique within the enclosing window/popup.
bool drawDirectoryBrowser(
    std::string& explorerDirectory, std::string* selectedDirectory, const char* listChildId,
    std::string& filterText);

#include "ui_common/file_browser_utils.hpp"
#include "ui_common/path_picker.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

namespace fs = std::filesystem;

// Layout: <root>/models/best.onnx, <root>/models/classes.txt, <root>/models/runs/
// Created once by main(), removed at the end (same pattern as
// yolo_inference_tests' temp dirs).
fs::path g_root;

void createFixture() {
    g_root = fs::temp_directory_path() / "vision_ml_path_picker_tests";
    std::error_code ec;
    fs::remove_all(g_root, ec);
    fs::create_directories(g_root / "models" / "runs");
    std::ofstream(g_root / "models" / "best.onnx") << "x";
    std::ofstream(g_root / "models" / "classes.txt") << "a\n";
}

void removeFixture() {
    std::error_code ec;
    fs::remove_all(g_root, ec);
}

// ---- cleanPickerPath ----

void test_cleanPickerPath_stripsTrailingSlash() {
    CHECK(cleanPickerPath("/a/b/") == fs::path("/a/b"));
    CHECK(cleanPickerPath("/a/b/").parent_path() == fs::path("/a"));
}

void test_cleanPickerPath_keepsRoot() {
    CHECK(cleanPickerPath("/") == fs::path("/"));
}

void test_cleanPickerPath_collapsesDotSegments() {
    CHECK(cleanPickerPath("/a/./b/../c") == fs::path("/a/c"));
}

void test_cleanPickerPath_makesRelativeAbsolute() {
    const fs::path cleaned = cleanPickerPath("models/x");
    CHECK(cleaned.is_absolute());
    CHECK(cleaned == (fs::current_path() / "models" / "x").lexically_normal());
}

void test_cleanPickerPath_keepsEmptyEmpty() {
    CHECK(cleanPickerPath("").empty());
}

// ---- initialPickerDirectory ----

void test_initialPickerDirectory_relativeValueBecomesAbsolute() {
    const fs::path relative = fs::relative(g_root / "models" / "best.onnx", fs::current_path());
    const auto result = initialPickerDirectory(relative.string());
    CHECK(result.dir == g_root / "models");
    CHECK(result.fromCurrentValue);
}

void test_initialPickerDirectory_existingFile() {
    const auto result = initialPickerDirectory((g_root / "models" / "best.onnx").string());
    CHECK(result.dir == g_root / "models");
    CHECK(result.fromCurrentValue);
}

void test_initialPickerDirectory_existingFolder() {
    const auto result = initialPickerDirectory((g_root / "models").string() + "/");
    CHECK(result.dir == g_root / "models");
    CHECK(result.fromCurrentValue);
}

void test_initialPickerDirectory_empty() {
    const auto result = initialPickerDirectory("");
    CHECK(result.dir == defaultFileSelectorRoot());
    CHECK(!result.fromCurrentValue);
}

void test_initialPickerDirectory_missingFileExistingParent() {
    const auto result = initialPickerDirectory((g_root / "models" / "renamed.onnx").string());
    CHECK(result.dir == g_root / "models");
    CHECK(result.fromCurrentValue);
}

void test_initialPickerDirectory_missingEverything() {
    const auto result = initialPickerDirectory((g_root / "nope" / "deeper" / "x.onnx").string());
    CHECK(result.dir == defaultFileSelectorRoot());
    CHECK(!result.fromCurrentValue);
}

// ---- resolvePathInput ----

void test_resolvePathInput_directoryNavigatesInBothModes() {
    const std::string dir = (g_root / "models").string();
    CHECK(resolvePathInput(PathPickerMode::File, dir, "/").action == PathInputAction::Navigate);
    CHECK(resolvePathInput(PathPickerMode::Folder, dir, "/").action == PathInputAction::Navigate);
}

void test_resolvePathInput_fileSelectsInFileModeOnly() {
    const std::string file = (g_root / "models" / "best.onnx").string();
    const auto inFileMode = resolvePathInput(PathPickerMode::File, file, "/");
    CHECK(inFileMode.action == PathInputAction::SelectFile);
    CHECK(inFileMode.path == g_root / "models" / "best.onnx");
    CHECK(resolvePathInput(PathPickerMode::Folder, file, "/").action == PathInputAction::NotFound);
}

void test_resolvePathInput_missingIsNotFound() {
    CHECK(resolvePathInput(PathPickerMode::File, (g_root / "missing").string(), "/").action
          == PathInputAction::NotFound);
    CHECK(resolvePathInput(PathPickerMode::File, "", "/").action == PathInputAction::NotFound);
    CHECK(resolvePathInput(PathPickerMode::File, "   ", "/").action == PathInputAction::NotFound);
}

void test_resolvePathInput_trimsWhitespace() {
    const std::string padded = "  " + (g_root / "models").string() + "/\n";
    const auto result = resolvePathInput(PathPickerMode::Folder, padded, "/");
    CHECK(result.action == PathInputAction::Navigate);
    CHECK(result.path == g_root / "models");
}

void test_resolvePathInput_relative() {
    const auto result = resolvePathInput(PathPickerMode::File, "runs", g_root / "models");
    CHECK(result.action == PathInputAction::Navigate);
    CHECK(result.path == g_root / "models" / "runs");
    const auto up = resolvePathInput(PathPickerMode::File, "..", g_root / "models");
    CHECK(up.path == g_root);
}

void test_resolvePathInput_tildeAlone() {
    const auto result = resolvePathInput(PathPickerMode::Folder, "~", "/");
    CHECK(result.action == PathInputAction::Navigate);
    CHECK(result.path == cleanPickerPath(defaultFileSelectorRoot()));
}

void test_resolvePathInput_tildeSlash() {
    const auto result = resolvePathInput(PathPickerMode::Folder, "~/", "/");
    CHECK(result.path == cleanPickerPath(defaultFileSelectorRoot()));
}

// ---- state operations ----

void test_openPathPicker_seedsFromFolderValue() {
    PathPickerState picker;
    picker.filter = "stale";
    picker.selectedFile = "/stale";
    picker.pathInputError = "Path not found";
    picker.cachedDir = "/stale";
    openPathPicker(picker, PathPickerMode::Folder, "Pick Image Folder", (g_root / "models").string());
    CHECK(picker.openRequested);
    CHECK(picker.mode == PathPickerMode::Folder);
    CHECK(picker.title == "Pick Image Folder");
    CHECK(picker.currentDir == (g_root / "models").string());
    CHECK(picker.originDir == picker.currentDir);
    CHECK(picker.pathInput == picker.currentDir);
    CHECK(picker.filter.empty());
    CHECK(picker.selectedFile.empty());
    CHECK(picker.pathInputError.empty());
    CHECK(picker.cachedDir.empty());
}

void test_openPathPicker_emptyValueHasNoOrigin() {
    PathPickerState picker;
    openPathPicker(picker, PathPickerMode::File, "Pick ONNX Model", "");
    CHECK(picker.originDir.empty());
    CHECK(picker.currentDir == cleanPickerPath(defaultFileSelectorRoot()).string());
}

void test_openPathPicker_fileModePreselectsExistingFile() {
    PathPickerState picker;
    const std::string file = (g_root / "models" / "best.onnx").string();
    openPathPicker(picker, PathPickerMode::File, "Pick ONNX Model", file);
    CHECK(picker.currentDir == (g_root / "models").string());
    CHECK(picker.selectedFile == file);
    CHECK(picker.pathInput == file);
}

void test_openPathPicker_preselectRequestsScroll() {
    PathPickerState picker;
    openPathPicker(picker, PathPickerMode::File, "Pick ONNX Model", (g_root / "models" / "best.onnx").string());
    CHECK(picker.scrollToSelection);
}

void test_openPathPicker_snapshotsExistingRecents() {
    rememberPathPickerLocation(g_root / "gone");
    rememberPathPickerLocation(g_root / "models");
    PathPickerState picker;
    openPathPicker(picker, PathPickerMode::Folder, "Pick Image Folder", "");
    CHECK(picker.recentChoices.size() == 1);
    CHECK(picker.recentChoices.size() == 1 && picker.recentChoices[0] == g_root / "models");
}

void test_applyInput_fileSelectionRequestsScroll() {
    PathPickerState picker;
    picker.mode = PathPickerMode::File;
    picker.currentDir = "/";
    picker.pathInput = (g_root / "models" / "best.onnx").string();
    applyPathPickerInput(picker);
    CHECK(picker.scrollToSelection);
}

void test_clickFile_doesNotRequestScroll() {
    PathPickerState picker;
    clickPathPickerFile(picker, g_root / "models" / "best.onnx", false);
    CHECK(!picker.scrollToSelection);
}

void test_navigate_clearsFilterAndSelection() {
    PathPickerState picker;
    picker.filter = "be";
    picker.selectedFile = (g_root / "models" / "best.onnx").string();
    picker.pathInputError = "Path not found";
    navigatePathPicker(picker, g_root / "models" / "runs");
    CHECK(picker.currentDir == (g_root / "models" / "runs").string());
    CHECK(picker.pathInput == picker.currentDir);
    CHECK(picker.filter.empty());
    CHECK(picker.selectedFile.empty());
    CHECK(picker.pathInputError.empty());
}

void test_navigate_trailingSlashThenUp() {
    PathPickerState picker;
    navigatePathPicker(picker, (g_root / "models").string() + "/");
    CHECK(fs::path(picker.currentDir).parent_path() == g_root);
}

void test_applyInput_fileNavigatesThenSelects() {
    PathPickerState picker;
    picker.mode = PathPickerMode::File;
    picker.currentDir = "/";
    picker.pathInput = (g_root / "models" / "classes.txt").string();
    applyPathPickerInput(picker);
    CHECK(picker.currentDir == (g_root / "models").string());
    CHECK(picker.selectedFile == (g_root / "models" / "classes.txt").string());
    CHECK(picker.pathInput == picker.selectedFile);
    CHECK(picker.pathInputError.empty());
}

void test_applyInput_notFoundKeepsLocation() {
    PathPickerState picker;
    picker.currentDir = (g_root / "models").string();
    picker.pathInput = (g_root / "missing").string();
    applyPathPickerInput(picker);
    CHECK(picker.currentDir == (g_root / "models").string());
    CHECK(picker.pathInputError == "Path not found");
    CHECK(picker.pathInput == (g_root / "missing").string());
}

void test_refreshListing_fileModeListsBoth() {
    PathPickerState picker;
    picker.mode = PathPickerMode::File;
    picker.currentDir = (g_root / "models").string();
    refreshPathPickerListing(picker);
    CHECK(picker.cachedDir == picker.currentDir);
    CHECK(picker.cachedDirs.size() == 1);
    CHECK(picker.cachedFiles.size() == 2);
}

void test_refreshListing_folderModeListsNoFiles() {
    PathPickerState picker;
    picker.mode = PathPickerMode::Folder;
    picker.currentDir = (g_root / "models").string();
    refreshPathPickerListing(picker);
    CHECK(picker.cachedDirs.size() == 1);
    CHECK(picker.cachedFiles.empty());
}

void test_refreshListing_skipsWhenUnchanged() {
    PathPickerState picker;
    picker.mode = PathPickerMode::File;
    picker.currentDir = (g_root / "models").string();
    refreshPathPickerListing(picker);
    picker.cachedFiles.clear();  // sentinel: an unchanged dir must not re-read
    refreshPathPickerListing(picker);
    CHECK(picker.cachedFiles.empty());
}

void test_clickFile_doubleClickOnUnselectedOnlySelects() {
    // The second click of a double-click on a folder row lands on whatever
    // row is under the cursor after navigating; it must not confirm it.
    PathPickerState picker;
    picker.currentDir = (g_root / "models").string();
    const fs::path file = g_root / "models" / "best.onnx";
    CHECK(!clickPathPickerFile(picker, file, true));
    CHECK(picker.selectedFile == file.string());
}

void test_clickFile_doubleClickOnSelectedConfirms() {
    PathPickerState picker;
    const fs::path file = g_root / "models" / "best.onnx";
    CHECK(!clickPathPickerFile(picker, file, false));
    CHECK(clickPathPickerFile(picker, file, true));
}

void test_clickFile_singleClickNeverConfirms() {
    PathPickerState picker;
    const fs::path file = g_root / "models" / "best.onnx";
    selectPathPickerFile(picker, file);
    CHECK(!clickPathPickerFile(picker, file, false));
    CHECK(picker.selectedFile == file.string());
}

// ---- recents ----

void test_pushRecent_newAtFront() {
    std::vector<fs::path> recents{"/a"};
    pushRecentLocation(recents, "/b");
    CHECK(recents.size() == 2);
    CHECK(recents[0] == fs::path("/b"));
    CHECK(recents[1] == fs::path("/a"));
}

void test_pushRecent_duplicateMovesToFront() {
    std::vector<fs::path> recents{"/a", "/b", "/c"};
    pushRecentLocation(recents, "/c/");
    CHECK(recents.size() == 3);
    CHECK(recents[0] == fs::path("/c"));
    CHECK(recents[1] == fs::path("/a"));
    CHECK(recents[2] == fs::path("/b"));
}

void test_pushRecent_capDropsOldest() {
    std::vector<fs::path> recents;
    for (int i = 0; i < 12; ++i) {
        pushRecentLocation(recents, fs::path("/d" + std::to_string(i)));
    }
    CHECK(recents.size() == kMaxRecentLocations);
    CHECK(recents.front() == fs::path("/d11"));
    CHECK(recents.back() == fs::path("/d2"));
}

void test_recentLine_roundTrip() {
    const fs::path dir("/home/ag/my datasets/run 3");
    const auto parsed = parseRecentLine(formatRecentLine(dir));
    CHECK(parsed.has_value());
    CHECK(parsed.has_value() && *parsed == dir);
    CHECK(formatRecentLine(dir) == "Path=/home/ag/my datasets/run 3");
}

void test_recentLine_rejectsJunk() {
    CHECK(!parseRecentLine("").has_value());
    CHECK(!parseRecentLine("Path=").has_value());
    CHECK(!parseRecentLine("Pos=60,60").has_value());
}

void test_recentLine_stripsCarriageReturn() {
    const auto parsed = parseRecentLine("Path=/a/b\r");
    CHECK(parsed.has_value() && *parsed == fs::path("/a/b"));
}

void test_appendRecentFromIniLine_dedupesAndCaps() {
    std::vector<fs::path> recents;
    appendRecentFromIniLine(recents, "Path=/a");
    appendRecentFromIniLine(recents, "Path=/a/");
    appendRecentFromIniLine(recents, "Junk=1");
    CHECK(recents.size() == 1);
    for (int i = 0; i < 20; ++i) {
        appendRecentFromIniLine(recents, "Path=/x" + std::to_string(i));
    }
    CHECK(recents.size() == kMaxRecentLocations);
    CHECK(recents.front() == fs::path("/a"));  // file order preserved: first line is newest
}

void test_existingRecentLocations_skipsMissing() {
    const std::vector<fs::path> recents{g_root / "missing", g_root / "models", g_root};
    const auto existing = existingRecentLocations(recents);
    CHECK(existing.size() == 2);
    CHECK(existing.size() == 2 && existing[0] == g_root / "models");
}

} // namespace

int main() {
    createFixture();

    test_cleanPickerPath_stripsTrailingSlash();
    test_cleanPickerPath_keepsRoot();
    test_cleanPickerPath_collapsesDotSegments();
    test_cleanPickerPath_makesRelativeAbsolute();
    test_cleanPickerPath_keepsEmptyEmpty();
    test_initialPickerDirectory_relativeValueBecomesAbsolute();
    test_initialPickerDirectory_existingFile();
    test_initialPickerDirectory_existingFolder();
    test_initialPickerDirectory_empty();
    test_initialPickerDirectory_missingFileExistingParent();
    test_initialPickerDirectory_missingEverything();
    test_resolvePathInput_directoryNavigatesInBothModes();
    test_resolvePathInput_fileSelectsInFileModeOnly();
    test_resolvePathInput_missingIsNotFound();
    test_resolvePathInput_trimsWhitespace();
    test_resolvePathInput_relative();
    test_resolvePathInput_tildeAlone();
    test_resolvePathInput_tildeSlash();
    test_openPathPicker_seedsFromFolderValue();
    test_openPathPicker_emptyValueHasNoOrigin();
    test_openPathPicker_fileModePreselectsExistingFile();
    test_openPathPicker_preselectRequestsScroll();
    test_openPathPicker_snapshotsExistingRecents();
    test_applyInput_fileSelectionRequestsScroll();
    test_clickFile_doesNotRequestScroll();
    test_navigate_clearsFilterAndSelection();
    test_navigate_trailingSlashThenUp();
    test_applyInput_fileNavigatesThenSelects();
    test_applyInput_notFoundKeepsLocation();
    test_refreshListing_fileModeListsBoth();
    test_refreshListing_folderModeListsNoFiles();
    test_refreshListing_skipsWhenUnchanged();
    test_clickFile_doubleClickOnUnselectedOnlySelects();
    test_clickFile_doubleClickOnSelectedConfirms();
    test_clickFile_singleClickNeverConfirms();
    test_pushRecent_newAtFront();
    test_pushRecent_duplicateMovesToFront();
    test_pushRecent_capDropsOldest();
    test_recentLine_roundTrip();
    test_recentLine_rejectsJunk();
    test_recentLine_stripsCarriageReturn();
    test_appendRecentFromIniLine_dedupesAndCaps();
    test_existingRecentLocations_skipsMissing();

    removeFixture();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

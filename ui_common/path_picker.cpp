#include "ui_common/path_picker.hpp"

#include "ui_common/file_browser_utils.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cfloat>
#include <system_error>
#include <utility>

namespace {

constexpr std::string_view kRecentLinePrefix = "Path=";

std::string_view trimWhitespace(std::string_view text) {
    const auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!text.empty() && isSpace(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && isSpace(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

bool isExistingDirectory(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

bool isExistingRegularFile(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

// Process-wide so every picker shares one recent list, and so the ini
// settings handler (a plain function pointer) can reach it.
std::vector<std::filesystem::path>& recentLocations() {
    static std::vector<std::filesystem::path> recents;
    return recents;
}

void* recentSettingsReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name) {
    if (std::string_view(name) != "Recent") {
        return nullptr;
    }
    recentLocations().clear();
    return &recentLocations();
}

void recentSettingsReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line) {
    appendRecentFromIniLine(*static_cast<std::vector<std::filesystem::path>*>(entry), line);
}

void recentSettingsWriteAll(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* out) {
    const auto& recents = recentLocations();
    if (recents.empty()) {
        return;
    }
    out->appendf("[%s][Recent]\n", handler->TypeName);
    for (const auto& dir : recents) {
        out->appendf("%s\n", formatRecentLine(dir).c_str());
    }
    out->append("\n");
}

enum class PickerIcon { Folder, File, Up, Home, Current, Recent, Go };

// Simple vector icons drawn into the square [min, min + size] -- the app
// ships only ImGui's default font, which has no icon glyphs. Coordinates
// are fractions of size so icons follow font size/DPI.
void drawPickerIcon(ImDrawList* drawList, PickerIcon icon, ImVec2 min, float size, ImU32 color) {
    const auto at = [&](float fx, float fy) { return ImVec2(min.x + fx * size, min.y + fy * size); };
    const float thickness = std::max(1.0f, size * 0.09f);
    switch (icon) {
        case PickerIcon::Folder:
            drawList->AddRectFilled(at(0.06f, 0.16f), at(0.46f, 0.36f), color, size * 0.06f);
            drawList->AddRectFilled(at(0.06f, 0.28f), at(0.94f, 0.86f), color, size * 0.08f);
            break;
        case PickerIcon::File: {
            const ImVec2 page[] = {at(0.22f, 0.08f), at(0.60f, 0.08f), at(0.80f, 0.28f), at(0.80f, 0.92f),
                                   at(0.22f, 0.92f)};
            drawList->AddPolyline(page, 5, color, ImDrawFlags_Closed, thickness);
            const ImVec2 fold[] = {at(0.60f, 0.08f), at(0.60f, 0.28f), at(0.80f, 0.28f)};
            drawList->AddPolyline(fold, 3, color, ImDrawFlags_None, thickness);
            break;
        }
        case PickerIcon::Up:
            drawList->AddTriangleFilled(at(0.50f, 0.10f), at(0.88f, 0.50f), at(0.12f, 0.50f), color);
            drawList->AddRectFilled(at(0.38f, 0.48f), at(0.62f, 0.90f), color);
            break;
        case PickerIcon::Go:
            drawList->AddTriangleFilled(at(0.90f, 0.50f), at(0.50f, 0.12f), at(0.50f, 0.88f), color);
            drawList->AddRectFilled(at(0.10f, 0.38f), at(0.52f, 0.62f), color);
            break;
        case PickerIcon::Home:
            drawList->AddTriangleFilled(at(0.50f, 0.08f), at(0.94f, 0.50f), at(0.06f, 0.50f), color);
            drawList->AddRectFilled(at(0.20f, 0.48f), at(0.80f, 0.92f), color);
            break;
        case PickerIcon::Current:
            drawList->AddCircle(at(0.50f, 0.50f), size * 0.38f, color, 0, thickness);
            drawList->AddCircleFilled(at(0.50f, 0.50f), size * 0.14f, color);
            break;
        case PickerIcon::Recent:
            drawList->AddCircle(at(0.50f, 0.50f), size * 0.40f, color, 0, thickness);
            drawList->AddLine(at(0.50f, 0.50f), at(0.50f, 0.24f), color, thickness);
            drawList->AddLine(at(0.50f, 0.50f), at(0.70f, 0.50f), color, thickness);
            break;
    }
}

float iconButtonWidth(const char* label) {
    const ImGuiStyle& style = ImGui::GetStyle();
    return style.FramePadding.x * 2.0f + ImGui::GetTextLineHeight() + style.ItemInnerSpacing.x
        + ImGui::CalcTextSize(label).x;
}

// A Button with an icon before its text. id must start with "##".
bool iconButton(const char* id, PickerIcon icon, const char* label) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const bool pressed = ImGui::Button(id, ImVec2(iconButtonWidth(label), ImGui::GetFrameHeight()));
    // GetColorU32 applies style.Alpha, so icons fade with BeginDisabled.
    const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
    const float iconSize = ImGui::GetTextLineHeight();
    const ImVec2 iconMin(ImGui::GetItemRectMin().x + style.FramePadding.x, ImGui::GetItemRectMin().y + style.FramePadding.y);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawPickerIcon(drawList, icon, iconMin, iconSize, color);
    drawList->AddText(ImVec2(iconMin.x + iconSize + style.ItemInnerSpacing.x, iconMin.y), color, label);
    return pressed;
}

// A full-width Selectable row with an icon and a name. The name is drawn
// separately rather than used as the label, so "##" in a file name can't
// truncate it or collide IDs; callers PushID a unique key per row.
bool iconSelectable(PickerIcon icon, ImU32 iconColor, const std::string& name, bool selected,
                    ImGuiSelectableFlags flags = ImGuiSelectableFlags_None) {
    const bool clicked = ImGui::Selectable("##row", selected, flags);
    const float iconSize = ImGui::GetTextLineHeight();
    const ImVec2 min = ImGui::GetItemRectMin();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawPickerIcon(drawList, icon, min, iconSize, iconColor);
    drawList->AddText(ImVec2(min.x + iconSize + ImGui::GetStyle().ItemInnerSpacing.x, min.y),
                      ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
    return clicked;
}

} // namespace

std::filesystem::path cleanPickerPath(const std::filesystem::path& path) {
    std::filesystem::path absolute = path;
    if (!path.empty() && path.is_relative()) {
        std::error_code ec;
        absolute = std::filesystem::absolute(path, ec);
        if (ec) {
            absolute = path;
        }
    }
    std::string text = absolute.lexically_normal().string();
    while (text.size() > 1 && text.back() == '/') {
        text.pop_back();
    }
    return std::filesystem::path(text);
}

InitialPickerDirectory initialPickerDirectory(const std::string& currentValue) {
    namespace fs = std::filesystem;
    if (!currentValue.empty()) {
        const fs::path value = cleanPickerPath(fs::path(currentValue));
        if (isExistingDirectory(value)) {
            return {value, true};
        }
        // Covers both an existing file and a renamed/deleted one whose folder
        // is still there -- either way the user wants to start next to it.
        const fs::path parent = value.parent_path();
        if (!parent.empty() && isExistingDirectory(parent)) {
            return {parent, true};
        }
    }
    return {defaultFileSelectorRoot(), false};
}

PathInputResolution resolvePathInput(
    PathPickerMode mode, std::string_view text, const std::filesystem::path& currentDir) {
    namespace fs = std::filesystem;
    const std::string_view trimmed = trimWhitespace(text);
    if (trimmed.empty()) {
        return {PathInputAction::NotFound, {}};
    }

    fs::path path;
    if (trimmed == "~") {
        path = defaultFileSelectorRoot();
    } else if (trimmed.starts_with("~/")) {
        path = defaultFileSelectorRoot() / std::string(trimmed.substr(2));
    } else {
        path = fs::path(std::string(trimmed));
        if (path.is_relative()) {
            path = currentDir / path;
        }
    }
    path = cleanPickerPath(path);

    if (isExistingDirectory(path)) {
        return {PathInputAction::Navigate, path};
    }
    if (mode == PathPickerMode::File && isExistingRegularFile(path)) {
        return {PathInputAction::SelectFile, path};
    }
    return {PathInputAction::NotFound, path};
}

void openPathPicker(
    PathPickerState& picker, PathPickerMode mode, std::string title, const std::string& currentValue) {
    const InitialPickerDirectory initial = initialPickerDirectory(currentValue);
    picker.mode = mode;
    picker.title = std::move(title);
    navigatePathPicker(picker, initial.dir);
    picker.originDir = initial.fromCurrentValue ? picker.currentDir : std::string();
    if (mode == PathPickerMode::File && !currentValue.empty()) {
        const std::filesystem::path value = cleanPickerPath(std::filesystem::path(currentValue));
        if (isExistingRegularFile(value)) {
            selectPathPickerFile(picker, value);
            picker.scrollToSelection = true;
        }
    }
    picker.recentChoices = existingRecentLocations(recentLocations());
    picker.cachedDir.clear();
    picker.cachedDirs.clear();
    picker.cachedFiles.clear();
    picker.openRequested = true;
}

void navigatePathPicker(PathPickerState& picker, const std::filesystem::path& dir) {
    picker.currentDir = cleanPickerPath(dir).string();
    picker.pathInput = picker.currentDir;
    picker.pathInputError.clear();
    picker.filter.clear();
    picker.selectedFile.clear();
}

void selectPathPickerFile(PathPickerState& picker, const std::filesystem::path& file) {
    picker.selectedFile = file.string();
    picker.pathInput = picker.selectedFile;
    picker.pathInputError.clear();
}

bool clickPathPickerFile(PathPickerState& picker, const std::filesystem::path& file, bool doubleClick) {
    const bool wasSelected = picker.selectedFile == file.string();
    selectPathPickerFile(picker, file);
    return doubleClick && wasSelected;
}

void applyPathPickerInput(PathPickerState& picker) {
    const PathInputResolution resolved =
        resolvePathInput(picker.mode, picker.pathInput, std::filesystem::path(picker.currentDir));
    switch (resolved.action) {
        case PathInputAction::Navigate:
            navigatePathPicker(picker, resolved.path);
            break;
        case PathInputAction::SelectFile:
            navigatePathPicker(picker, resolved.path.parent_path());
            selectPathPickerFile(picker, resolved.path);
            picker.scrollToSelection = true;
            break;
        case PathInputAction::NotFound:
            picker.pathInputError = "Path not found";
            break;
    }
}

void refreshPathPickerListing(PathPickerState& picker) {
    if (picker.cachedDir == picker.currentDir) {
        return;
    }
    const std::filesystem::path dir(picker.currentDir);
    picker.cachedDir = picker.currentDir;
    picker.cachedDirs = listDirectories(dir);
    picker.cachedFiles = picker.mode == PathPickerMode::File ? listFiles(dir) : std::vector<std::filesystem::path>{};
}

void pushRecentLocation(
    std::vector<std::filesystem::path>& recents, const std::filesystem::path& dir, std::size_t cap) {
    const std::filesystem::path cleaned = cleanPickerPath(dir);
    if (cleaned.empty()) {
        return;
    }
    std::erase(recents, cleaned);
    recents.insert(recents.begin(), cleaned);
    if (recents.size() > cap) {
        recents.resize(cap);
    }
}

void appendRecentFromIniLine(std::vector<std::filesystem::path>& recents, std::string_view line) {
    if (recents.size() >= kMaxRecentLocations) {
        return;
    }
    const std::optional<std::filesystem::path> parsed = parseRecentLine(line);
    if (!parsed) {
        return;
    }
    const std::filesystem::path cleaned = cleanPickerPath(*parsed);
    if (std::find(recents.begin(), recents.end(), cleaned) == recents.end()) {
        recents.push_back(cleaned);
    }
}

std::string formatRecentLine(const std::filesystem::path& dir) {
    return std::string(kRecentLinePrefix) + dir.string();
}

std::optional<std::filesystem::path> parseRecentLine(std::string_view line) {
    // ImGui passes lines without '\n', but a hand-edited ini can carry '\r'.
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.remove_suffix(1);
    }
    if (!line.starts_with(kRecentLinePrefix)) {
        return std::nullopt;
    }
    line.remove_prefix(kRecentLinePrefix.size());
    if (line.empty()) {
        return std::nullopt;
    }
    return std::filesystem::path(std::string(line));
}

std::vector<std::filesystem::path> existingRecentLocations(const std::vector<std::filesystem::path>& recents) {
    std::vector<std::filesystem::path> existing;
    for (const auto& dir : recents) {
        if (isExistingDirectory(dir)) {
            existing.push_back(dir);
        }
    }
    return existing;
}

void rememberPathPickerLocation(const std::filesystem::path& dir) {
    pushRecentLocation(recentLocations(), dir);
}

void registerPathPickerSettingsHandler() {
    ImGuiSettingsHandler handler;
    handler.TypeName = "PathPicker";
    handler.TypeHash = ImHashStr("PathPicker");
    handler.ReadOpenFn = recentSettingsReadOpen;
    handler.ReadLineFn = recentSettingsReadLine;
    handler.WriteAllFn = recentSettingsWriteAll;
    ImGui::AddSettingsHandler(&handler);
}

std::optional<std::filesystem::path> drawPathPicker(PathPickerState& picker, const char* popupId) {
    namespace fs = std::filesystem;

    // "###" keeps the popup ID tied to popupId while the visible title
    // changes between targets ("Pick ONNX Model", "Pick Image Folder", ...).
    const std::string label = (picker.title.empty() ? std::string("Pick Path") : picker.title) + "###" + popupId;
    if (picker.openRequested) {
        ImGui::OpenPopup(label.c_str());
        picker.openRequested = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(label.c_str(), nullptr)) {
        return std::nullopt;
    }

    // The folder can vanish while the popup is open (or between opens).
    const fs::path existing = normalizeDirectoryOrDefault(fs::path(picker.currentDir));
    if (existing.string() != picker.currentDir) {
        navigatePathPicker(picker, existing);
    }
    refreshPathPickerListing(picker);

    // Sampled before any widget runs: InputText handles Escape (revert and
    // deactivate) inside its own call, so checking after the widgets would
    // see "no item active" and close the picker on that same first press.
    const bool escapeCloses = !ImGui::IsAnyItemActive() && ImGui::IsKeyPressed(ImGuiKey_Escape, false);

    const ImGuiStyle& style = ImGui::GetStyle();

    // Path row.
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Path:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-(iconButtonWidth("Go") + style.ItemSpacing.x));
    bool applyInput = ImGui::InputText("##PathInput", &picker.pathInput, ImGuiInputTextFlags_EnterReturnsTrue);
    if (ImGui::IsItemEdited()) {
        picker.pathInputError.clear();
    }
    ImGui::SameLine();
    applyInput |= iconButton("##Go", PickerIcon::Go, "Go");
    if (applyInput) {
        applyPathPickerInput(picker);
    }
    if (!picker.pathInputError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", picker.pathInputError.c_str());
    }

    // Quick jumps. Navigation is deferred to after the row so every button
    // in it sees the same frame state.
    std::optional<fs::path> navigateTarget;
    if (iconButton("##Up", PickerIcon::Up, "Up")) {
        const fs::path current(picker.currentDir);
        const fs::path parent = current.parent_path();
        if (!parent.empty() && parent != current) {
            navigateTarget = parent;
        }
    }
    ImGui::SameLine();
    if (iconButton("##Home", PickerIcon::Home, "Home")) {
        navigateTarget = defaultFileSelectorRoot();
    }
    if (!picker.originDir.empty()) {
        ImGui::SameLine();
        if (iconButton("##Current", PickerIcon::Current, "Current")) {
            navigateTarget = fs::path(picker.originDir);
        }
    }
    ImGui::SameLine();
    const std::vector<fs::path>& recents = picker.recentChoices;
    ImGui::BeginDisabled(recents.empty());
    ImGui::SetNextItemWidth(iconButtonWidth("Recent") + ImGui::GetFrameHeight());
    // CustomPreview lives in imgui_internal.h's private enum; cast to the
    // flags int to combine it with the public one.
    const ImGuiComboFlags recentComboFlags =
        ImGuiComboFlags_HeightLarge | static_cast<ImGuiComboFlags>(ImGuiComboFlags_CustomPreview);
    if (ImGui::BeginCombo("##Recent", "", recentComboFlags)) {
        const ImU32 folderColor = ImGui::GetColorU32(ImGuiCol_SliderGrab);
        for (const auto& dir : recents) {
            ImGui::PushID(dir.string().c_str());
            if (iconSelectable(PickerIcon::Folder, folderColor, dir.string(), false)) {
                navigateTarget = dir;
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    if (ImGui::BeginComboPreview()) {
        const float iconSize = ImGui::GetTextLineHeight();
        drawPickerIcon(ImGui::GetWindowDrawList(), PickerIcon::Recent, ImGui::GetCursorScreenPos(), iconSize,
                       ImGui::GetColorU32(ImGuiCol_Text));
        ImGui::Dummy(ImVec2(iconSize, iconSize));
        ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::TextUnformatted("Recent");
        ImGui::EndComboPreview();
    }
    ImGui::EndDisabled();

    // Filter row.
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Filter:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##Filter", "Search...", &picker.filter);

    // Combined list: folders first, then files (File mode only). Fills the
    // remaining height minus the button row.
    std::optional<fs::path> chosen;
    if (ImGui::BeginChild("##PathPickerList", ImVec2(0.0f, -ImGui::GetFrameHeightWithSpacing()),
                          ImGuiChildFlags_Borders)) {
        bool anyShown = false;
        const ImU32 folderColor = ImGui::GetColorU32(ImGuiCol_SliderGrab);
        const ImU32 fileColor = ImGui::GetColorU32(ImGuiCol_TextDisabled);
        for (const auto& dir : picker.cachedDirs) {
            if (!fileNameMatchesFilter(dir, picker.filter)) {
                continue;
            }
            anyShown = true;
            ImGui::PushID(dir.string().c_str());
            // Ignore the second click of a double-click: it lands on whatever
            // row moved under the cursor after the first click navigated.
            if (iconSelectable(PickerIcon::Folder, folderColor, dir.filename().string() + "/", false)
                && ImGui::GetIO().MouseClickedLastCount[0] < 2) {
                navigateTarget = dir;
            }
            ImGui::PopID();
        }
        for (const auto& file : picker.cachedFiles) {
            if (!fileNameMatchesFilter(file, picker.filter)) {
                continue;
            }
            anyShown = true;
            const bool isSelected = file.string() == picker.selectedFile;
            ImGui::PushID(file.string().c_str());
            if (iconSelectable(PickerIcon::File, fileColor, file.filename().string(), isSelected,
                               ImGuiSelectableFlags_AllowDoubleClick)) {
                if (clickPathPickerFile(picker, file, ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) {
                    chosen = file;
                }
            }
            if (isSelected && picker.scrollToSelection) {
                ImGui::SetScrollHereY(0.5f);
            }
            ImGui::PopID();
        }
        if (!anyShown) {
            const bool empty = picker.cachedDirs.empty() && picker.cachedFiles.empty();
            ImGui::TextDisabled(empty ? "Empty folder." : "Nothing matches filter.");
        }
    }
    ImGui::EndChild();
    // One-shot: cleared even when the filter hides the row, so it can't fire
    // later after the user has scrolled away.
    picker.scrollToSelection = false;

    if (navigateTarget) {
        navigatePathPicker(picker, *navigateTarget);
    }

    // Button row.
    if (picker.mode == PathPickerMode::File) {
        ImGui::BeginDisabled(picker.selectedFile.empty());
        if (ImGui::Button("Select")) {
            chosen = fs::path(picker.selectedFile);
        }
        ImGui::EndDisabled();
    } else if (ImGui::Button("Use this folder")) {
        chosen = fs::path(picker.currentDir);
    }
    ImGui::SameLine();
    // Escape while typing only leaves the text box; a second press closes.
    const bool cancel = ImGui::Button("Cancel") || escapeCloses;

    if (chosen) {
        rememberPathPickerLocation(picker.mode == PathPickerMode::File ? chosen->parent_path() : *chosen);
        ImGui::MarkIniSettingsDirty();
        ImGui::CloseCurrentPopup();
    } else if (cancel) {
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
    return chosen;
}

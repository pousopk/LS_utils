#pragma once

#include "manager/label_studio_profiles.hpp"
#include "manager/label_studio_session.hpp"

#include <filesystem>
#include <optional>
#include <string>

// The Connection tab's saved-profile picker: the profiles file, what was
// loaded from it, which profile is selected, and the last load/save
// message. Owned by MlAppUi. Every edit re-reads the file, applies on top
// of it (keeping profiles another instance saved) and writes it at once; on
// a failed write the store is reloaded from disk so the UI never shows
// unsaved profiles.
struct LabelStudioProfilesState {
    std::optional<std::filesystem::path> file;   // nullopt: no config dir; profiles unavailable
    LabelStudioProfileStore store;
    std::string selectedName;   // empty: "(none)" -- fields typed by hand
    std::string error;
    std::string warning;
    std::string saveAsName;     // "Save as..." popup buffer
};

// Loads `file` (nullopt => error, read-only). The no-arg overload uses
// defaultLabelStudioProfilesPath().
LabelStudioProfilesState loadLabelStudioProfilesState(const std::optional<std::filesystem::path>& file);
LabelStudioProfilesState loadLabelStudioProfilesState();

// True when there is a path and the file loaded cleanly.
bool canEditLabelStudioProfiles(const LabelStudioProfilesState& state);

// Selects `name` and copies its URL/token into the session fields. An empty
// name selects "(none)" and leaves the fields as they are. Never connects.
void selectLabelStudioProfile(
    LabelStudioProfilesState& state, const std::string& name, LabelStudioSessionState& session);

// True when a profile is selected and the session fields differ from it.
bool selectedLabelStudioProfileDiffers(const LabelStudioProfilesState& state, const LabelStudioSessionState& session);

// Each returns true when the file was written. On false, `state.error`
// says why (unless nothing was attempted because the store isn't editable
// or nothing is selected). Saving requires a non-empty base URL and token.
bool saveSelectedLabelStudioProfile(LabelStudioProfilesState& state, const LabelStudioSessionState& session);
// Trims `name`; an existing name is overwritten (the caller confirms).
// On success the saved profile becomes selected.
bool saveLabelStudioProfileAs(
    LabelStudioProfilesState& state, const std::string& name, const LabelStudioSessionState& session);
// On success the selection resets to "(none)"; session fields are untouched.
bool deleteSelectedLabelStudioProfile(LabelStudioProfilesState& state);

#include "manager/label_studio_profiles_state.hpp"

#include <utility>

namespace {

bool hasRequiredFields(const LabelStudioSessionState& session) {
    return !session.baseUrl.empty() && !session.apiToken.empty();
}

constexpr const char* kRequiredFieldsError = "Base URL and API token are required to save a profile.";

// Re-reads the file after a failed write, so the in-memory list always
// matches disk (including changes made outside the app). Drops the
// selection if that profile no longer exists.
void reloadFromDisk(LabelStudioProfilesState& state, const std::string& saveError) {
    LabelStudioProfilesLoadResult result = loadLabelStudioProfiles(*state.file);
    state.store = std::move(result.store);
    state.warning = std::move(result.warning);
    state.error = result.error.empty() ? saveError : saveError + " " + result.error;
    if (findLabelStudioProfile(state.store, state.selectedName) == nullptr) {
        state.selectedName.clear();
    }
}

// Reads the file as it is now, so an edit applies on top of profiles another
// vision_ml instance saved since this one loaded, instead of writing back a
// stale list over them. On a load error the state adopts the (read-only)
// result and the edit is abandoned.
bool loadCurrentStore(LabelStudioProfilesState& state, LabelStudioProfileStore& out) {
    LabelStudioProfilesLoadResult result = loadLabelStudioProfiles(*state.file);
    if (!result.error.empty()) {
        state.store = std::move(result.store);
        state.error = std::move(result.error);
        state.selectedName.clear();
        return false;
    }
    out = std::move(result.store);
    return true;
}

// Writes `updated`; adopts it on success, reloads from disk on failure.
bool commit(LabelStudioProfilesState& state, LabelStudioProfileStore updated) {
    const std::string error = saveLabelStudioProfiles(*state.file, updated);
    if (!error.empty()) {
        reloadFromDisk(state, error);
        return false;
    }
    state.store = std::move(updated);
    state.error.clear();
    return true;
}

} // namespace

LabelStudioProfilesState loadLabelStudioProfilesState(const std::optional<std::filesystem::path>& file) {
    LabelStudioProfilesState state;
    state.file = file;
    if (!file) {
        state.store.readOnly = true;
        state.error = "No config directory (HOME and XDG_CONFIG_HOME are unset); profiles can't be saved.";
        return state;
    }
    LabelStudioProfilesLoadResult result = loadLabelStudioProfiles(*file);
    state.store = std::move(result.store);
    state.error = std::move(result.error);
    state.warning = std::move(result.warning);
    return state;
}

LabelStudioProfilesState loadLabelStudioProfilesState() {
    return loadLabelStudioProfilesState(defaultLabelStudioProfilesPath());
}

bool canEditLabelStudioProfiles(const LabelStudioProfilesState& state) {
    return state.file.has_value() && !state.store.readOnly;
}

void selectLabelStudioProfile(
    LabelStudioProfilesState& state, const std::string& name, LabelStudioSessionState& session) {
    if (name.empty()) {
        state.selectedName.clear();
        return;
    }
    const LabelStudioProfile* profile = findLabelStudioProfile(state.store, name);
    if (profile == nullptr) {
        return;
    }
    state.selectedName = profile->name;
    session.baseUrl = profile->baseUrl;
    session.apiToken = profile->apiToken;
}

bool selectedLabelStudioProfileDiffers(const LabelStudioProfilesState& state, const LabelStudioSessionState& session) {
    const LabelStudioProfile* profile = findLabelStudioProfile(state.store, state.selectedName);
    if (profile == nullptr) {
        return false;
    }
    return profile->baseUrl != session.baseUrl || profile->apiToken != session.apiToken;
}

bool saveSelectedLabelStudioProfile(LabelStudioProfilesState& state, const LabelStudioSessionState& session) {
    if (!canEditLabelStudioProfiles(state) || state.selectedName.empty()) {
        return false;
    }
    if (!hasRequiredFields(session)) {
        state.error = kRequiredFieldsError;
        return false;
    }
    LabelStudioProfileStore updated;
    if (!loadCurrentStore(state, updated)) {
        return false;
    }
    upsertLabelStudioProfile(updated, {state.selectedName, session.baseUrl, session.apiToken});
    return commit(state, std::move(updated));
}

bool saveLabelStudioProfileAs(
    LabelStudioProfilesState& state, const std::string& name, const LabelStudioSessionState& session) {
    if (!canEditLabelStudioProfiles(state)) {
        return false;
    }
    const std::string trimmed = trimLabelStudioProfileName(name);
    if (trimmed.empty()) {
        state.error = "Profile name can't be empty.";
        return false;
    }
    if (!hasRequiredFields(session)) {
        state.error = kRequiredFieldsError;
        return false;
    }
    LabelStudioProfileStore updated;
    if (!loadCurrentStore(state, updated)) {
        return false;
    }
    upsertLabelStudioProfile(updated, {trimmed, session.baseUrl, session.apiToken});
    if (!commit(state, std::move(updated))) {
        return false;
    }
    state.selectedName = trimmed;
    return true;
}

bool deleteSelectedLabelStudioProfile(LabelStudioProfilesState& state) {
    if (!canEditLabelStudioProfiles(state) || state.selectedName.empty()) {
        return false;
    }
    LabelStudioProfileStore updated;
    if (!loadCurrentStore(state, updated)) {
        return false;
    }
    removeLabelStudioProfile(updated, state.selectedName);
    if (!commit(state, std::move(updated))) {
        return false;
    }
    state.selectedName.clear();
    return true;
}

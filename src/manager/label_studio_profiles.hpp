#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Saved Label Studio connection profiles, stored as plaintext JSON in a
// file only the owning user can read (0600, directory 0700) -- the
// ~/.netrc model: it keeps tokens out of the repo, the working directory
// and other local users' reach, not away from processes running as this
// user. Linux only.
//
// File format:
//   {"version": 1, "profiles": [{"name": ..., "baseUrl": ..., "apiToken": ...}]}

struct LabelStudioProfile {
    std::string name;
    std::string baseUrl;
    std::string apiToken;

    bool operator==(const LabelStudioProfile&) const = default;
};

struct LabelStudioProfileStore {
    std::vector<LabelStudioProfile> profiles;   // file order == display order
    // Set when the file exists but could not be loaded: saving is refused
    // so a file the user may still want to fix is never overwritten.
    bool readOnly = false;
};

struct LabelStudioProfilesLoadResult {
    LabelStudioProfileStore store;
    std::string error;     // non-empty => store.readOnly; never contains file content
    std::string warning;   // e.g. permissions were tightened to 0600
};

// $XDG_CONFIG_HOME/vision_ml/label_studio_profiles.json when XDG_CONFIG_HOME
// is an absolute path, else $HOME/.config/vision_ml/label_studio_profiles.json;
// std::nullopt when neither is usable. Takes the variables as parameters so
// tests don't touch the process environment.
std::optional<std::filesystem::path> defaultLabelStudioProfilesPath(const char* xdgConfigHome, const char* home);
std::optional<std::filesystem::path> defaultLabelStudioProfilesPath();

// Missing file => empty, writable store. A file readable by group/other is
// chmod-ed to 0600 with a warning. Invalid JSON, wrong field types, empty or
// duplicate names, or a version other than 1 => error + readOnly store.
LabelStudioProfilesLoadResult loadLabelStudioProfiles(const std::filesystem::path& file);

// Writes `store` atomically (0600 temp file in the same directory, fsync,
// rename), creating the parent directory with 0700 if missing. Returns an
// empty string on success, otherwise an error; on error the existing file
// is untouched. Refuses a readOnly store.
std::string saveLabelStudioProfiles(const std::filesystem::path& file, const LabelStudioProfileStore& store);

// In-memory edits; no I/O.
const LabelStudioProfile* findLabelStudioProfile(const LabelStudioProfileStore& store, const std::string& name);
// Replaces the profile with the same name in place, or appends it.
void upsertLabelStudioProfile(LabelStudioProfileStore& store, LabelStudioProfile profile);
// False when no profile has that name.
bool removeLabelStudioProfile(LabelStudioProfileStore& store, const std::string& name);

// Strips leading/trailing spaces, tabs and newlines from a typed name.
std::string trimLabelStudioProfileName(const std::string& name);

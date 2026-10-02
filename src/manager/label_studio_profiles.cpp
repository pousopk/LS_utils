#include "manager/label_studio_profiles.hpp"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr long long kProfilesFileVersion = 1;

std::string errnoText() {
    return std::strerror(errno);
}

bool isUsableEnvValue(const char* value) {
    return value != nullptr && value[0] != '\0';
}

// Error for a file that exists but failed validation. Built only from the
// path and a fixed reason (plus, at most, a profile name) -- never from file
// content -- so a token can't leak into the UI.
std::string invalidFileError(const fs::path& file, const std::string& reason) {
    return "Profiles file " + file.string() + " is invalid (" + reason + "); fix or remove it to save profiles.";
}

bool readStringField(const nlohmann::json& entry, const char* key, std::string& out) {
    const auto it = entry.find(key);
    if (it == entry.end() || !it->is_string()) {
        return false;
    }
    out = it->get<std::string>();
    return true;
}

bool writeAll(int fd, const std::string& data) {
    size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = ::write(fd, data.data() + written, data.size() - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

// Creates `dir` with mode 0700 if missing; its parents (e.g. ~/.config) get
// default permissions. An existing directory is left as it is -- the file
// inside is 0600 regardless.
std::string ensurePrivateDirectory(const fs::path& dir) {
    std::error_code ec;
    if (dir.empty() || fs::is_directory(dir, ec)) {
        return {};
    }
    if (dir.has_parent_path()) {
        fs::create_directories(dir.parent_path(), ec);
        if (ec) {
            return "Could not create " + dir.parent_path().string() + ": " + ec.message();
        }
    }
    if (::mkdir(dir.c_str(), S_IRWXU) != 0 && errno != EEXIST) {
        return "Could not create " + dir.string() + ": " + errnoText();
    }
    return {};
}

} // namespace

std::optional<fs::path> defaultLabelStudioProfilesPath(const char* xdgConfigHome, const char* home) {
    const fs::path relative = fs::path("vision_ml") / "label_studio_profiles.json";
    // The XDG spec says relative XDG_CONFIG_HOME values are to be ignored.
    if (isUsableEnvValue(xdgConfigHome) && fs::path(xdgConfigHome).is_absolute()) {
        return fs::path(xdgConfigHome) / relative;
    }
    if (isUsableEnvValue(home)) {
        return fs::path(home) / ".config" / relative;
    }
    return std::nullopt;
}

std::optional<fs::path> defaultLabelStudioProfilesPath() {
    return defaultLabelStudioProfilesPath(std::getenv("XDG_CONFIG_HOME"), std::getenv("HOME"));
}

LabelStudioProfilesLoadResult loadLabelStudioProfiles(const fs::path& file) {
    LabelStudioProfilesLoadResult result;
    auto fail = [&](const std::string& error) {
        result.store = {};
        result.store.readOnly = true;
        result.error = error;
        return result;
    };

    struct stat info{};
    if (::stat(file.c_str(), &info) != 0) {
        if (errno == ENOENT) {
            return result;
        }
        return fail("Could not read " + file.string() + ": " + errnoText());
    }
    if ((info.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        if (::chmod(file.c_str(), S_IRUSR | S_IWUSR) == 0) {
            result.warning =
                "Profiles file " + file.string() + " was readable by other users; permissions tightened to 0600.";
        } else {
            result.warning = "Profiles file " + file.string()
                + " is readable by other users and its permissions could not be tightened: " + errnoText();
        }
    }

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return fail("Could not read " + file.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();

    const nlohmann::json root = nlohmann::json::parse(buffer.str(), nullptr, false);
    if (root.is_discarded()) {
        return fail(invalidFileError(file, "not valid JSON"));
    }
    if (!root.is_object()) {
        return fail(invalidFileError(file, "top level is not an object"));
    }
    const auto version = root.find("version");
    if (version == root.end() || !version->is_number_integer()) {
        return fail(invalidFileError(file, "missing integer \"version\""));
    }
    if (version->get<long long>() != kProfilesFileVersion) {
        return fail(invalidFileError(file, "unsupported version " + std::to_string(version->get<long long>())));
    }
    const auto profiles = root.find("profiles");
    if (profiles == root.end() || !profiles->is_array()) {
        return fail(invalidFileError(file, "missing \"profiles\" array"));
    }

    for (size_t i = 0; i < profiles->size(); ++i) {
        const nlohmann::json& entry = (*profiles)[i];
        const std::string where = "profile " + std::to_string(i + 1);
        if (!entry.is_object()) {
            return fail(invalidFileError(file, where + " is not an object"));
        }
        LabelStudioProfile profile;
        if (!readStringField(entry, "name", profile.name) || !readStringField(entry, "baseUrl", profile.baseUrl)
            || !readStringField(entry, "apiToken", profile.apiToken)) {
            return fail(invalidFileError(file, where + " needs string \"name\", \"baseUrl\" and \"apiToken\""));
        }
        if (trimLabelStudioProfileName(profile.name).empty()) {
            return fail(invalidFileError(file, where + " has an empty name"));
        }
        if (findLabelStudioProfile(result.store, profile.name) != nullptr) {
            return fail(invalidFileError(file, "duplicate profile name \"" + profile.name + "\""));
        }
        result.store.profiles.push_back(std::move(profile));
    }
    return result;
}

std::string saveLabelStudioProfiles(const fs::path& file, const LabelStudioProfileStore& store) {
    if (store.readOnly) {
        return "Profiles file " + file.string() + " could not be loaded, so it is not being overwritten.";
    }
    if (const std::string error = ensurePrivateDirectory(file.parent_path()); !error.empty()) {
        return error;
    }

    nlohmann::json root;
    root["version"] = kProfilesFileVersion;
    root["profiles"] = nlohmann::json::array();
    for (const auto& profile : store.profiles) {
        root["profiles"].push_back(
            {{"name", profile.name}, {"baseUrl", profile.baseUrl}, {"apiToken", profile.apiToken}});
    }
    const std::string data = root.dump(2) + "\n";

    const fs::path temp = file.string() + ".tmp." + std::to_string(::getpid());
    ::unlink(temp.c_str());   // a leftover from a save that crashed under the same pid
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return "Could not write " + temp.string() + ": " + errnoText();
    }
    std::string error;
    if (!writeAll(fd, data) || ::fsync(fd) != 0) {
        error = "Could not write " + temp.string() + ": " + errnoText();
    }
    if (::close(fd) != 0 && error.empty()) {
        error = "Could not write " + temp.string() + ": " + errnoText();
    }
    if (error.empty() && ::rename(temp.c_str(), file.c_str()) != 0) {
        error = "Could not replace " + file.string() + ": " + errnoText();
    }
    if (!error.empty()) {
        ::unlink(temp.c_str());
    }
    return error;
}

const LabelStudioProfile* findLabelStudioProfile(const LabelStudioProfileStore& store, const std::string& name) {
    for (const auto& profile : store.profiles) {
        if (profile.name == name) {
            return &profile;
        }
    }
    return nullptr;
}

void upsertLabelStudioProfile(LabelStudioProfileStore& store, LabelStudioProfile profile) {
    for (auto& existing : store.profiles) {
        if (existing.name == profile.name) {
            existing = std::move(profile);
            return;
        }
    }
    store.profiles.push_back(std::move(profile));
}

bool removeLabelStudioProfile(LabelStudioProfileStore& store, const std::string& name) {
    for (auto it = store.profiles.begin(); it != store.profiles.end(); ++it) {
        if (it->name == name) {
            store.profiles.erase(it);
            return true;
        }
    }
    return false;
}

std::string trimLabelStudioProfileName(const std::string& name) {
    const char* whitespace = " \t\r\n";
    const size_t first = name.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = name.find_last_not_of(whitespace);
    return name.substr(first, last - first + 1);
}

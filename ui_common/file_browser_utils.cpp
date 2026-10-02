#include "ui_common/file_browser_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <system_error>

namespace {

std::string toLowerCopy(std::string_view text) {
    std::string lower(text.begin(), text.end());
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower;
}

} // namespace

std::filesystem::path defaultFileSelectorRoot() {
    namespace fs = std::filesystem;
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        const fs::path homePath(home);
        std::error_code ec;
        if (fs::exists(homePath, ec) && fs::is_directory(homePath, ec)) {
            return homePath;
        }
    }

    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec && fs::exists(cwd, ec) && fs::is_directory(cwd, ec)) {
        return cwd;
    }
    return fs::path("/");
}

std::filesystem::path normalizeDirectoryOrDefault(const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    const fs::path candidate = directory.empty() ? defaultFileSelectorRoot() : directory;
    std::error_code ec;
    if (fs::exists(candidate, ec) && fs::is_directory(candidate, ec)) {
        return candidate;
    }
    return defaultFileSelectorRoot();
}

std::vector<std::filesystem::path> listDirectories(const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    std::vector<fs::path> dirs;

    std::error_code ec;
    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        return dirs;
    }

    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_directory(ec) && !ec) {
            dirs.push_back(entry.path());
        }
    }

    std::sort(dirs.begin(), dirs.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });
    return dirs;
}

std::vector<std::filesystem::path> listFiles(const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;

    std::error_code ec;
    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        return files;
    }

    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_regular_file(ec) && !ec) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });
    return files;
}

bool fileNameMatchesFilter(const std::filesystem::path& file, std::string_view filterText) {
    if (filterText.empty()) {
        return true;
    }
    const std::string name = toLowerCopy(file.filename().string());
    return name.find(toLowerCopy(filterText)) != std::string::npos;
}

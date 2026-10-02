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

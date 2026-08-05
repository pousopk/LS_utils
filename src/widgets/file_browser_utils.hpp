#pragma once

#include <filesystem>
#include <vector>

std::filesystem::path defaultFileSelectorRoot();
std::filesystem::path normalizeDirectoryOrDefault(const std::filesystem::path& directory);
std::vector<std::filesystem::path> listDirectories(const std::filesystem::path& directory);
std::vector<std::filesystem::path> listFiles(const std::filesystem::path& directory);

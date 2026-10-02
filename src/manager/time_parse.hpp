#pragma once

#include <ctime>
#include <optional>
#include <string>

// Pure function: parses Label Studio's `created_at` timestamp format
// (ISO-8601 UTC, e.g. "2024-05-01T12:34:56.789012Z" -- fractional seconds
// and/or a trailing "Z" are both tolerated and ignored) into epoch
// seconds. Returns std::nullopt if the string doesn't match
// "%Y-%m-%dT%H:%M:%S" at minimum.
std::optional<std::time_t> parseIso8601Utc(const std::string& value);

// Parses a manually-typed factory timestamp in "YYYY/MM/DD HH:MM:SS"
// format (e.g. "2026/09/10 12:00:00") as LOCAL time -- the timestamp
// engraved on a physical piece reflects wherever/whenever it was
// engraved, not UTC, and comparing it against Label Studio's UTC
// `created_at` requires converting it first. Uses mktime, so it respects
// the running machine's configured timezone and DST rules; this assumes
// the machine running vision_app is set to the same timezone as the
// factory. Returns std::nullopt on any format mismatch. Not a pure
// function (mktime depends on the process's timezone setting).
std::optional<std::time_t> parseTypedLocalTimestamp(const std::string& value);

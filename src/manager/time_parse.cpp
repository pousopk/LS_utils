#include "manager/time_parse.hpp"

#include <iomanip>
#include <regex>
#include <sstream>

namespace {

std::optional<std::tm> parseTmWithFormat(const std::string& value, const char* format) {
    std::tm tm{};
    std::istringstream iss(value);
    iss >> std::get_time(&tm, format);
    if (iss.fail()) {
        return std::nullopt;
    }
    return tm;
}

} // namespace

std::optional<std::time_t> parseIso8601Utc(const std::string& value) {
    std::optional<std::tm> tm = parseTmWithFormat(value, "%Y-%m-%dT%H:%M:%S");
    if (!tm) {
        return std::nullopt;
    }
    // timegm is POSIX (available on this app's only build target, Linux)
    // -- unlike mktime, it interprets `tm` as UTC instead of the local
    // timezone, matching Label Studio's `created_at` format exactly.
    const std::time_t result = timegm(&*tm);
    if (result == static_cast<std::time_t>(-1)) {
        return std::nullopt;
    }
    return result;
}

std::optional<std::time_t> parseTypedLocalTimestamp(const std::string& value) {
    // std::get_time reports success when the input simply runs out, so a
    // half-typed "2026/09/1" would otherwise parse -- require the whole
    // shape (surrounding whitespace aside) before handing it over.
    static const std::regex kShape(R"(\s*\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}\s*)");
    if (!std::regex_match(value, kShape)) {
        return std::nullopt;
    }
    std::optional<std::tm> tm = parseTmWithFormat(value, "%Y/%m/%d %H:%M:%S");
    if (!tm) {
        return std::nullopt;
    }
    tm->tm_isdst = -1; // let mktime determine DST for this date from the system's timezone rules
    const std::time_t result = mktime(&*tm);
    if (result == static_cast<std::time_t>(-1)) {
        return std::nullopt;
    }
    return result;
}

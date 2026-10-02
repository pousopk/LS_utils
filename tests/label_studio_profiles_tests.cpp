#include "manager/label_studio_profiles.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {
int g_failures = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}
} // namespace

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

namespace {

// <root>/ is recreated by main() and removed at the end (same pattern as
// path_picker_tests).
fs::path g_root;

void createFixture() {
    g_root = fs::temp_directory_path() / "vision_ml_label_studio_profiles_tests";
    std::error_code ec;
    fs::remove_all(g_root, ec);
    fs::create_directories(g_root);
}

void removeFixture() {
    std::error_code ec;
    fs::remove_all(g_root, ec);
}

// A fresh, not-yet-existing profiles path under its own subdirectory.
fs::path freshFile(const std::string& testName) {
    return g_root / testName / "vision_ml" / "label_studio_profiles.json";
}

unsigned modeOf(const fs::path& path) {
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0) {
        return 0xFFFF;
    }
    return info.st_mode & 0777;
}

std::string readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void writeFile(const fs::path& path, const std::string& content, unsigned mode) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << content;
    ::chmod(path.c_str(), mode);
}

LabelStudioProfileStore twoProfiles() {
    LabelStudioProfileStore store;
    store.profiles = {
        {"prod", "https://ls.example.com", "token-prod"},
        {"local", "http://localhost:8080", "token-local"},
    };
    return store;
}

bool noTempFilesIn(const fs::path& dir) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
            return false;
        }
    }
    return true;
}

// ---- defaultLabelStudioProfilesPath ----

void test_path_usesXdgConfigHome() {
    const auto path = defaultLabelStudioProfilesPath("/xdg", "/home/u");
    CHECK(path.has_value() && *path == fs::path("/xdg/vision_ml/label_studio_profiles.json"));
}

void test_path_emptyXdgFallsBackToHome() {
    const auto path = defaultLabelStudioProfilesPath("", "/home/u");
    CHECK(path.has_value() && *path == fs::path("/home/u/.config/vision_ml/label_studio_profiles.json"));
}

void test_path_relativeXdgFallsBackToHome() {
    const auto path = defaultLabelStudioProfilesPath("relative/dir", "/home/u");
    CHECK(path.has_value() && *path == fs::path("/home/u/.config/vision_ml/label_studio_profiles.json"));
}

void test_path_neitherSetIsNullopt() {
    CHECK(!defaultLabelStudioProfilesPath(nullptr, nullptr).has_value());
    CHECK(!defaultLabelStudioProfilesPath("", "").has_value());
}

// ---- save ----

void test_save_createsPrivateDirAndFile() {
    const fs::path file = freshFile("save_creates");
    CHECK(saveLabelStudioProfiles(file, twoProfiles()).empty());
    CHECK(modeOf(file.parent_path()) == 0700);
    CHECK(modeOf(file) == 0600);
    CHECK(noTempFilesIn(file.parent_path()));
}

void test_save_overExistingLooseFileIsPrivate() {
    const fs::path file = freshFile("save_over_loose");
    writeFile(file, "{}", 0644);
    CHECK(saveLabelStudioProfiles(file, twoProfiles()).empty());
    CHECK(modeOf(file) == 0600);
}

void test_save_succeedsWithLeftoverTempFile() {
    const fs::path file = freshFile("save_leftover_tmp");
    const fs::path leftover = file.string() + ".tmp." + std::to_string(::getpid());
    writeFile(leftover, "partial", 0600);
    CHECK(saveLabelStudioProfiles(file, twoProfiles()).empty());
    CHECK(noTempFilesIn(file.parent_path()));
    CHECK(loadLabelStudioProfiles(file).store.profiles == twoProfiles().profiles);
}

void test_save_refusesReadOnlyStore() {
    const fs::path file = freshFile("save_readonly");
    LabelStudioProfileStore store = twoProfiles();
    store.readOnly = true;
    CHECK(!saveLabelStudioProfiles(file, store).empty());
    CHECK(!fs::exists(file));
}

// ---- load ----

void test_load_roundTripKeepsOrder() {
    const fs::path file = freshFile("roundtrip");
    CHECK(saveLabelStudioProfiles(file, twoProfiles()).empty());
    const auto result = loadLabelStudioProfiles(file);
    CHECK(result.error.empty());
    CHECK(result.warning.empty());
    CHECK(!result.store.readOnly);
    CHECK(result.store.profiles == twoProfiles().profiles);
}

void test_load_missingFileIsEmptyAndWritable() {
    const auto result = loadLabelStudioProfiles(freshFile("missing"));
    CHECK(result.error.empty());
    CHECK(result.store.profiles.empty());
    CHECK(!result.store.readOnly);
}

void test_load_looseFileIsTightenedWithWarning() {
    const fs::path file = freshFile("loose");
    CHECK(saveLabelStudioProfiles(file, twoProfiles()).empty());
    ::chmod(file.c_str(), 0644);
    const auto result = loadLabelStudioProfiles(file);
    CHECK(result.error.empty());
    CHECK(!result.warning.empty());
    CHECK(modeOf(file) == 0600);
    CHECK(result.store.profiles.size() == 2);
}

// Each of these must load as an error + readOnly store, and the error must
// never echo the token back.
void expectInvalid(const std::string& testName, const std::string& content) {
    const fs::path file = freshFile(testName);
    writeFile(file, content, 0600);
    const auto result = loadLabelStudioProfiles(file);
    CHECK(!result.error.empty());
    CHECK(result.store.readOnly);
    CHECK(result.store.profiles.empty());
    CHECK(result.error.find("SECRET123") == std::string::npos);
}

void test_load_invalidJson() {
    expectInvalid("invalid_json", R"({"version": 1, "profiles": [{"apiToken": "SECRET123")");
}

void test_load_wrongVersion() {
    expectInvalid("wrong_version", R"({"version": 2, "profiles": []})");
}

void test_load_missingVersion() {
    expectInvalid("missing_version", R"({"profiles": []})");
}

void test_load_profilesNotArray() {
    expectInvalid("profiles_not_array", R"({"version": 1, "profiles": {"name": "a"}})");
}

void test_load_wrongFieldType() {
    expectInvalid(
        "wrong_field_type",
        R"({"version": 1, "profiles": [{"name": "a", "baseUrl": 5, "apiToken": "SECRET123"}]})");
}

void test_load_nonObjectEntry() {
    expectInvalid(
        "non_object_entry",
        R"({"version": 1, "profiles": [{"name": "a", "baseUrl": "u", "apiToken": "SECRET123"}, 5]})");
}

void test_load_emptyName() {
    expectInvalid(
        "empty_name", R"({"version": 1, "profiles": [{"name": "  ", "baseUrl": "u", "apiToken": "SECRET123"}]})");
}

void test_load_duplicateNames() {
    expectInvalid(
        "duplicate_names",
        R"({"version": 1, "profiles": [)"
        R"({"name": "a", "baseUrl": "u", "apiToken": "SECRET123"},)"
        R"({"name": "a", "baseUrl": "v", "apiToken": "SECRET123"}]})");
}

void test_load_invalidFileIsNeverOverwritten() {
    const fs::path file = freshFile("invalid_not_overwritten");
    const std::string content = R"({"version": 2, "profiles": []})";
    writeFile(file, content, 0600);
    const auto result = loadLabelStudioProfiles(file);
    CHECK(!saveLabelStudioProfiles(file, result.store).empty());
    CHECK(readFile(file) == content);
}

// ---- in-memory edits ----

void test_upsert_appendsNewName() {
    LabelStudioProfileStore store = twoProfiles();
    upsertLabelStudioProfile(store, {"staging", "https://staging", "t"});
    CHECK(store.profiles.size() == 3);
    CHECK(store.profiles[2].name == "staging");
}

void test_upsert_replacesInPlace() {
    LabelStudioProfileStore store = twoProfiles();
    upsertLabelStudioProfile(store, {"prod", "https://new", "new-token"});
    CHECK(store.profiles.size() == 2);
    CHECK(store.profiles[0] == (LabelStudioProfile{"prod", "https://new", "new-token"}));
}

void test_remove_knownAndUnknown() {
    LabelStudioProfileStore store = twoProfiles();
    CHECK(removeLabelStudioProfile(store, "prod"));
    CHECK(store.profiles.size() == 1 && store.profiles[0].name == "local");
    CHECK(!removeLabelStudioProfile(store, "prod"));
}

void test_find() {
    const LabelStudioProfileStore store = twoProfiles();
    const LabelStudioProfile* found = findLabelStudioProfile(store, "local");
    CHECK(found != nullptr && found->apiToken == "token-local");
    CHECK(findLabelStudioProfile(store, "nope") == nullptr);
}

void test_trimName() {
    CHECK(trimLabelStudioProfileName("  prod \t\n") == "prod");
    CHECK(trimLabelStudioProfileName("   ").empty());
    CHECK(trimLabelStudioProfileName("a b") == "a b");
}

} // namespace

int main() {
    createFixture();

    test_path_usesXdgConfigHome();
    test_path_emptyXdgFallsBackToHome();
    test_path_relativeXdgFallsBackToHome();
    test_path_neitherSetIsNullopt();
    test_save_createsPrivateDirAndFile();
    test_save_overExistingLooseFileIsPrivate();
    test_save_succeedsWithLeftoverTempFile();
    test_save_refusesReadOnlyStore();
    test_load_roundTripKeepsOrder();
    test_load_missingFileIsEmptyAndWritable();
    test_load_looseFileIsTightenedWithWarning();
    test_load_invalidJson();
    test_load_wrongVersion();
    test_load_missingVersion();
    test_load_profilesNotArray();
    test_load_wrongFieldType();
    test_load_nonObjectEntry();
    test_load_emptyName();
    test_load_duplicateNames();
    test_load_invalidFileIsNeverOverwritten();
    test_upsert_appendsNewName();
    test_upsert_replacesInPlace();
    test_remove_knownAndUnknown();
    test_find();
    test_trimName();

    removeFixture();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

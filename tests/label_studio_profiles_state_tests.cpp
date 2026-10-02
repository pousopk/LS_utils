#include "manager/label_studio_profiles_state.hpp"

#include <cstdio>
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

fs::path g_root;

void createFixture() {
    g_root = fs::temp_directory_path() / "vision_ml_label_studio_profiles_state_tests";
    std::error_code ec;
    fs::remove_all(g_root, ec);
    fs::create_directories(g_root);
}

void removeFixture() {
    std::error_code ec;
    fs::permissions(g_root, fs::perms::owner_all, fs::perm_options::add, ec);
    fs::remove_all(g_root, ec);
}

fs::path freshFile(const std::string& testName) {
    return g_root / testName / "vision_ml" / "label_studio_profiles.json";
}

LabelStudioSessionState sessionWith(const std::string& baseUrl, const std::string& apiToken) {
    LabelStudioSessionState session;
    session.baseUrl = baseUrl;
    session.apiToken = apiToken;
    return session;
}

// A state backed by a file that already holds "prod".
LabelStudioProfilesState stateWithProd(const std::string& testName) {
    const fs::path file = freshFile(testName);
    LabelStudioProfileStore store;
    store.profiles = {{"prod", "https://ls.example.com", "token-prod"}};
    saveLabelStudioProfiles(file, store);
    return loadLabelStudioProfilesState(file);
}

void test_load_noPathIsReadOnlyWithError() {
    const auto state = loadLabelStudioProfilesState(std::nullopt);
    CHECK(!canEditLabelStudioProfiles(state));
    CHECK(!state.error.empty());
}

void test_load_missingFileIsEditable() {
    const auto state = loadLabelStudioProfilesState(freshFile("missing"));
    CHECK(canEditLabelStudioProfiles(state));
    CHECK(state.error.empty());
    CHECK(state.store.profiles.empty());
}

void test_select_copiesFieldsWithoutConnecting() {
    auto state = stateWithProd("select");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    CHECK(state.selectedName == "prod");
    CHECK(session.baseUrl == "https://ls.example.com");
    CHECK(session.apiToken == "token-prod");
    CHECK(session.status == LabelStudioSessionStatus::Disconnected);
}

void test_select_noneKeepsFields() {
    auto state = stateWithProd("select_none");
    auto session = sessionWith("https://typed", "typed-token");
    state.selectedName = "prod";
    selectLabelStudioProfile(state, "", session);
    CHECK(state.selectedName.empty());
    CHECK(session.baseUrl == "https://typed");
    CHECK(session.apiToken == "typed-token");
}

void test_differs() {
    auto state = stateWithProd("differs");
    auto session = sessionWith("", "");
    CHECK(!selectedLabelStudioProfileDiffers(state, session));   // nothing selected
    selectLabelStudioProfile(state, "prod", session);
    CHECK(!selectedLabelStudioProfileDiffers(state, session));
    session.apiToken = "edited";
    CHECK(selectedLabelStudioProfileDiffers(state, session));
}

void test_saveSelected_writesToDisk() {
    auto state = stateWithProd("save_selected");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    session.baseUrl = "https://new";
    CHECK(saveSelectedLabelStudioProfile(state, session));
    const auto reloaded = loadLabelStudioProfiles(*state.file);
    CHECK(reloaded.store.profiles.size() == 1);
    CHECK(reloaded.store.profiles[0].baseUrl == "https://new");
    CHECK(!selectedLabelStudioProfileDiffers(state, session));
}

void test_saveAs_trimsAndSelects() {
    auto state = loadLabelStudioProfilesState(freshFile("save_as_trim"));
    const auto session = sessionWith("https://ls", "t");
    CHECK(saveLabelStudioProfileAs(state, "  staging \t", session));
    CHECK(state.selectedName == "staging");
    const auto reloaded = loadLabelStudioProfiles(*state.file);
    CHECK(reloaded.store.profiles.size() == 1 && reloaded.store.profiles[0].name == "staging");
}

void test_saveAs_rejectsBlankName() {
    auto state = loadLabelStudioProfilesState(freshFile("save_as_blank"));
    CHECK(!saveLabelStudioProfileAs(state, "   ", sessionWith("https://ls", "t")));
    CHECK(!state.error.empty());
    CHECK(!fs::exists(*state.file));
}

void test_saveAs_requiresUrlAndToken() {
    auto state = loadLabelStudioProfilesState(freshFile("save_as_fields"));
    CHECK(!saveLabelStudioProfileAs(state, "a", sessionWith("", "t")));
    CHECK(!state.error.empty());
    state.error.clear();
    CHECK(!saveLabelStudioProfileAs(state, "a", sessionWith("https://ls", "")));
    CHECK(!state.error.empty());
    CHECK(!fs::exists(*state.file));
}

void test_saveSelected_requiresUrlAndToken() {
    auto state = stateWithProd("save_selected_fields");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    session.apiToken.clear();
    CHECK(!saveSelectedLabelStudioProfile(state, session));
    CHECK(!state.error.empty());
    CHECK(loadLabelStudioProfiles(*state.file).store.profiles[0].apiToken == "token-prod");
}

void test_saveAs_existingNameOverwrites() {
    auto state = stateWithProd("save_as_overwrite");
    CHECK(saveLabelStudioProfileAs(state, "prod", sessionWith("https://other", "other-token")));
    const auto reloaded = loadLabelStudioProfiles(*state.file);
    CHECK(reloaded.store.profiles.size() == 1);
    CHECK(reloaded.store.profiles[0].apiToken == "other-token");
}

void test_delete_removesAndClearsSelection() {
    auto state = stateWithProd("delete");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    CHECK(deleteSelectedLabelStudioProfile(state));
    CHECK(state.selectedName.empty());
    CHECK(state.store.profiles.empty());
    CHECK(loadLabelStudioProfiles(*state.file).store.profiles.empty());
    CHECK(session.baseUrl == "https://ls.example.com");   // fields untouched
}

void test_readOnlyStoreRefusesEdits() {
    const fs::path file = freshFile("read_only");
    fs::create_directories(file.parent_path());
    std::FILE* f = std::fopen(file.c_str(), "w");
    std::fputs("not json", f);
    std::fclose(f);
    auto state = loadLabelStudioProfilesState(file);
    CHECK(!canEditLabelStudioProfiles(state));
    CHECK(!state.error.empty());
    CHECK(!saveLabelStudioProfileAs(state, "a", sessionWith("https://ls", "t")));
}

void test_failedSaveRevertsToDisk() {
    if (::geteuid() == 0) {
        std::printf("skipping test_failedSaveRevertsToDisk: running as root\n");
        return;
    }
    auto state = stateWithProd("failed_save");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    const fs::path dir = state.file->parent_path();
    ::chmod(dir.c_str(), 0500);   // no new files => temp file can't be created
    const bool saved = saveLabelStudioProfileAs(state, "staging", sessionWith("https://s", "t"));
    ::chmod(dir.c_str(), 0700);
    CHECK(!saved);
    CHECK(!state.error.empty());
    CHECK(state.store.profiles.size() == 1 && state.store.profiles[0].name == "prod");
    CHECK(state.selectedName == "prod");
}

// Another vision_ml instance adds a profile after this state was loaded.
void addProfileBehindStatesBack(const LabelStudioProfilesState& state, const LabelStudioProfile& profile) {
    LabelStudioProfileStore onDisk = loadLabelStudioProfiles(*state.file).store;
    upsertLabelStudioProfile(onDisk, profile);
    saveLabelStudioProfiles(*state.file, onDisk);
}

void test_saveAs_keepsProfilesSavedByAnotherInstance() {
    auto state = stateWithProd("stale_save_as");
    addProfileBehindStatesBack(state, {"staging", "https://s", "token-staging"});
    CHECK(saveLabelStudioProfileAs(state, "local", sessionWith("http://localhost", "t")));
    const auto reloaded = loadLabelStudioProfiles(*state.file);
    CHECK(findLabelStudioProfile(reloaded.store, "staging") != nullptr);
    CHECK(findLabelStudioProfile(reloaded.store, "local") != nullptr);
    CHECK(findLabelStudioProfile(state.store, "staging") != nullptr);
}

void test_delete_keepsProfilesSavedByAnotherInstance() {
    auto state = stateWithProd("stale_delete");
    auto session = sessionWith("", "");
    selectLabelStudioProfile(state, "prod", session);
    addProfileBehindStatesBack(state, {"staging", "https://s", "token-staging"});
    CHECK(deleteSelectedLabelStudioProfile(state));
    const auto reloaded = loadLabelStudioProfiles(*state.file);
    CHECK(reloaded.store.profiles.size() == 1 && reloaded.store.profiles[0].name == "staging");
}

void test_save_refusesWhenFileBrokenByAnotherWriter() {
    auto state = stateWithProd("stale_broken");
    std::FILE* f = std::fopen(state.file->c_str(), "w");
    std::fputs("not json", f);
    std::fclose(f);
    CHECK(!saveLabelStudioProfileAs(state, "local", sessionWith("http://localhost", "t")));
    CHECK(!state.error.empty());
    CHECK(!canEditLabelStudioProfiles(state));
}

} // namespace

int main() {
    createFixture();

    test_load_noPathIsReadOnlyWithError();
    test_load_missingFileIsEditable();
    test_select_copiesFieldsWithoutConnecting();
    test_select_noneKeepsFields();
    test_differs();
    test_saveSelected_writesToDisk();
    test_saveAs_trimsAndSelects();
    test_saveAs_rejectsBlankName();
    test_saveAs_requiresUrlAndToken();
    test_saveSelected_requiresUrlAndToken();
    test_saveAs_existingNameOverwrites();
    test_delete_removesAndClearsSelection();
    test_readOnlyStoreRefusesEdits();
    test_failedSaveRevertsToDisk();
    test_saveAs_keepsProfilesSavedByAnotherInstance();
    test_delete_keepsProfilesSavedByAnotherInstance();
    test_save_refusesWhenFileBrokenByAnotherWriter();

    removeFixture();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

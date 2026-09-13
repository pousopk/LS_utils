#include "manager/label_studio_session.hpp"

#include <cstdio>

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

void test_applyProjectListResult_successPopulatesProjectsAndConnects() {
    LabelStudioSessionState session;
    session.status = LabelStudioSessionStatus::Connecting;

    LabelStudioProjectListResult result;
    result.projects = {{1, "Project One"}, {2, "Project Two"}};

    applyProjectListResult(session, result);

    CHECK(session.status == LabelStudioSessionStatus::Connected);
    CHECK(session.error.empty());
    CHECK(session.projects.size() == 2);
    CHECK(session.projects[0].id == 1);
    CHECK(session.projects[0].title == "Project One");
    CHECK(session.activeProjectId == 0);
    CHECK(session.activeProjectTitle.empty());
}

void test_applyProjectListResult_errorSetsErrorStatusAndClearsProjects() {
    LabelStudioSessionState session;
    session.projects = {{1, "Stale Project"}};
    session.activeProjectId = 1;
    session.activeProjectTitle = "Stale Project";

    LabelStudioProjectListResult result;
    result.error = "Label Studio returned HTTP 401: invalid token";

    applyProjectListResult(session, result);

    CHECK(session.status == LabelStudioSessionStatus::Error);
    CHECK(session.error == "Label Studio returned HTTP 401: invalid token");
    CHECK(session.projects.empty());
    CHECK(session.activeProjectId == 0);
    CHECK(session.activeProjectTitle.empty());
}

void test_applyProjectListResult_reconnectResetsPreviouslyActiveProject() {
    LabelStudioSessionState session;
    session.activeProjectId = 5;
    session.activeProjectTitle = "Old Project";

    LabelStudioProjectListResult result;
    result.projects = {{7, "New Project"}};

    applyProjectListResult(session, result);

    CHECK(session.activeProjectId == 0);
    CHECK(session.activeProjectTitle.empty());
    CHECK(session.projects.size() == 1);
    CHECK(session.projects[0].id == 7);
}

} // namespace

int main() {
    test_applyProjectListResult_successPopulatesProjectsAndConnects();
    test_applyProjectListResult_errorSetsErrorStatusAndClearsProjects();
    test_applyProjectListResult_reconnectResetsPreviouslyActiveProject();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}

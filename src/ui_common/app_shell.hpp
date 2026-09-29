#pragma once

#include <functional>
#include <string>

struct GLFWwindow;

struct AppShellConfig {
    std::string title = "Vision";
    int width = 1280;
    int height = 720;
    std::string iniFilename;   // ImGui layout file; empty disables saving
    int swapInterval = 1;      // 0 = uncapped (the camera loop polls sources every frame)
    bool hiddenWindow = false; // for headless loop benchmarks
};

// Owns the GLFW window + GL context and the ImGui context/backends, and
// tears them down in its destructor in the only safe order: ImGui
// backends, ImGui context, window, GLFW. Anything that owns GL objects
// or worker threads (the app's UI/session state) must be destroyed
// BEFORE this -- declare the AppShell first in main() so it is
// destroyed last. Not copyable.
class AppShell {
public:
    AppShell() = default;
    ~AppShell();
    AppShell(const AppShell&) = delete;
    AppShell& operator=(const AppShell&) = delete;

    // Returns false with `error` set if GLFW or the window can't be
    // created; the shell is then inert (run() returns immediately).
    bool init(const AppShellConfig& config, std::string& error);

    // Loops until the window is closed or requestClose() is called. Each
    // iteration: poll events, `update()` (no ImGui calls), then an ImGui
    // frame in which `draw()` issues all ImGui calls, then render + swap.
    void run(const std::function<void()>& update, const std::function<void()>& draw);

    void requestClose();

private:
    GLFWwindow* window_ = nullptr;
    bool imguiInitialized_ = false;
    std::string iniFilename_;   // ImGui keeps a pointer to this; must outlive the context
};

#include "ui_common/app_shell.hpp"

#include "ui_common/ui_theme.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

AppShell::~AppShell() {
    if (imguiInitialized_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
}

bool AppShell::init(const AppShellConfig& config, std::string& error) {
    if (glfwInit() == GLFW_FALSE) {
        error = "GLFW initialization failed";
        return false;
    }
    if (config.hiddenWindow) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }
    window_ = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        error = "GLFW window creation failed";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(config.swapInterval);

    // Matches ui_theme.cpp's kBgDarkest -- without this, any screen area not
    // painted by an opaque ImGui element shows the driver's undefined
    // default clear color instead of the theme's background.
    glClearColor(0.975f, 0.965f, 0.950f, 1.0f);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    iniFilename_ = config.iniFilename;
    ImGui::GetIO().IniFilename = iniFilename_.empty() ? nullptr : iniFilename_.c_str();
    applyProfessionalTheme();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    imguiInitialized_ = true;
    return true;
}

void AppShell::run(const std::function<void()>& update, const std::function<void()>& draw) {
    if (window_ == nullptr) {
        return;
    }
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        update();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw();
        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);
    }
}

void AppShell::requestClose() {
    if (window_ != nullptr) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
}

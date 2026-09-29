#include "ui_common/app_shell.hpp"
#include "widgets/ml_app_ui.hpp"

#include <iostream>
#include <string>

int main() {
    AppShell shell;   // declared first => destroyed last, after the UI's workers and textures (see AppShell)
    AppShellConfig config;
    config.title = "Vision ML";
    config.iniFilename = "vision_ml.ini";
    config.swapInterval = 1;   // nothing here needs an uncapped loop
    std::string error;
    if (!shell.init(config, error)) {
        std::cerr << error << std::endl;
        return 1;
    }

    {
        MlAppUi ui;
        shell.run([&ui] { ui.update(); }, [&ui] { ui.drawMainLayout(); });
    }
    return 0;
}

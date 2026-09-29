#pragma once

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

// Uploads `frame` (1, 3 or 4 channels, BGR order) into `texture` as RGB,
// reallocating storage only when the size changed since the last upload
// (tracked through uploadedWidth/uploadedHeight). A no-op for texture 0,
// an empty frame, or an unsupported channel count.
void uploadFrameToTexture(GLuint texture, const cv::Mat& frame, int& uploadedWidth, int& uploadedHeight);

// Owns one GL texture name: created on first get(), deleted in the
// destructor. Move-only, so a texture can't be deleted twice when its
// owner is moved (e.g. inside a std::vector). Must be destroyed while
// the GL context is still current -- AppShell's teardown order
// guarantees that for anything owned by the app's state. Existing
// raw-GLuint call sites are not converted yet; new code should use this.
class GlTexture {
public:
    GlTexture() = default;
    ~GlTexture();
    GlTexture(const GlTexture&) = delete;
    GlTexture& operator=(const GlTexture&) = delete;
    GlTexture(GlTexture&& other) noexcept;
    GlTexture& operator=(GlTexture&& other) noexcept;

    GLuint get();
    GLuint id() const { return id_; }

private:
    void reset();

    GLuint id_ = 0;
};

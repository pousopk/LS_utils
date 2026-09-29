#include "ui_common/gl_texture.hpp"

#include <opencv2/imgproc.hpp>

#include <utility>

void uploadFrameToTexture(GLuint texture, const cv::Mat& frame, int& uploadedWidth, int& uploadedHeight) {
    if (texture == 0 || frame.empty()) {
        return;
    }

    cv::Mat rgb;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
    } else {
        return;
    }

    const bool needsReallocate = (uploadedWidth != rgb.cols) || (uploadedHeight != rgb.rows);

    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (needsReallocate) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     rgb.cols, rgb.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        uploadedWidth = rgb.cols;
        uploadedHeight = rgb.rows;
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    rgb.cols, rgb.rows,
                    GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
}

GlTexture::~GlTexture() {
    reset();
}

GlTexture::GlTexture(GlTexture&& other) noexcept : id_(std::exchange(other.id_, 0)) {}

GlTexture& GlTexture::operator=(GlTexture&& other) noexcept {
    if (this != &other) {
        reset();
        id_ = std::exchange(other.id_, 0);
    }
    return *this;
}

GLuint GlTexture::get() {
    if (id_ == 0) {
        glGenTextures(1, &id_);
    }
    return id_;
}

void GlTexture::reset() {
    if (id_ != 0) {
        glDeleteTextures(1, &id_);
        id_ = 0;
    }
}

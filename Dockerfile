FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools, Python, and GLFW dependencies
RUN apt update -y && \
    apt install -y --no-install-recommends \
    build-essential \
    clang \
    clang-format \
    git \
    curl \
    unzip \
    wget \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    xorg-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Install Bazel
RUN BAZEL_ARCH=$([ $(uname -m) = "aarch64" ] && echo "arm64" || echo "amd64") && \
  curl \
    -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.24.0/bazelisk-linux-${BAZEL_ARCH}" \
    -o /usr/local/bin/bazel && \
  chmod +x /usr/local/bin/bazel

WORKDIR /app

ENV PYTHON_BIN_PATH=/usr/bin/python3.12

FROM mcr.microsoft.com/devcontainers/cpp:3-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        clang-22 \
        clang-format-22 \
        clang-tidy-22 \
        curl \
        doxygen \
        g++ \
        gdb \
        git \
        git-lfs \
        graphviz \
        lld-22 \
        lldb-22 \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        valgrind \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-22 100 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages "cmake>=4.3"

# The distro default g++ is below the 16.1 floor, so configuring in the
# container would fail the version gate. Same narrow sid pin as CI and the
# runtime image, plus alternatives so plain gcc/g++ resolve to 16.1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends debian-archive-keyring \
    && echo "deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian sid main" \
        > /etc/apt/sources.list.d/debian-sid.list \
    && printf 'Package: *\nPin: release o=Debian\nPin-Priority: 100\n\nPackage: gcc-16 gcc-16-* g++-16 g++-16-* cpp-16 cpp-16-* gcc-16-base libgcc-16-dev libstdc++-16-dev libstdc++6 libgcc-s1 libgomp1 libitm1 libatomic1 libquadmath0 libcc1-0 libasan8 libubsan1 liblsan0 libtsan2 libhwasan0\nPin: release o=Debian\nPin-Priority: 990\n' \
        > /etc/apt/preferences.d/debian \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc-16 g++-16 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-16 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-16 100 \
    && g++ --version \
    && rm -rf /var/lib/apt/lists/*

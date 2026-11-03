#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt >/dev/null 2>&1; then
    echo "this script currently supports apt-based environments only" >&2
    exit 1
fi

echo "removing and install latest cmake..."
sudo apt remove -y cmake || true
pip install cmake --upgrade

sudo apt install -y \
    build-essential \
    clang \
    clang-format \
    clang-tidy \
    doxygen \
    gdb \
    git-lfs \
    graphviz \
    ninja-build \
    valgrind

if command -v g++-16 >/dev/null 2>&1; then
    echo "g++-16 already present: $(g++-16 --version | head -n1)"
else
    echo
    echo "this project requires GCC 16.1, which no Ubuntu release packages yet."
    echo "the only source is Debian sid. this adds a sid apt source to your"
    echo "system, pinned to priority 100 so nothing is pulled from it except the"
    echo "gcc-16 packages listed below at priority 990."
    echo
    read -r -p "add the pinned Debian sid source? [y/N] " reply
    if [[ "${reply}" == "y" || "${reply}" == "Y" ]]; then
        sudo apt install -y --no-install-recommends debian-archive-keyring

        echo "deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian sid main" \
            | sudo tee /etc/apt/sources.list.d/debian-sid.list

        printf 'Package: *\nPin: release o=Debian\nPin-Priority: 100\n\nPackage: gcc-16 gcc-16-* g++-16 g++-16-* cpp-16 cpp-16-* gcc-16-base libgcc-16-dev libstdc++-16-dev libstdc++6 libgcc-s1 libgomp1 libitm1 libatomic1 libquadmath0 libcc1-0 libasan8 libubsan1 liblsan0 libtsan2 libhwasan0\nPin: release o=Debian\nPin-Priority: 990\n' \
            | sudo tee /etc/apt/preferences.d/debian

        sudo apt update
        sudo apt install -y --no-install-recommends gcc-16 g++-16
        g++-16 --version
    else
        echo "skipped. install GCC 16.1 yourself, or cmake configure will fail" >&2
        echo "the version gate in CMakeLists.txt." >&2
    fi
fi

if [[ -z "${VCPKG_ROOT:-}" ]]; then
    export VCPKG_ROOT="${HOME}/.local/vcpkg"
fi

if [[ ! -d "${VCPKG_ROOT}" ]]; then
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
fi

"${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics

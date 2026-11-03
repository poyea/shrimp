FROM ubuntu:26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# GCC 16.1.0 comes from Debian sid; ubuntu-toolchain-r/test only has
# pre-release 16.0.x snapshots. Debian is pinned to priority 100 overall and
# 990 for the gcc-16 packages alone, so nothing else is pulled from sid.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        debian-archive-keyring \
    && echo "deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian sid main" \
        > /etc/apt/sources.list.d/debian-sid.list \
    && printf 'Package: *\nPin: release o=Debian\nPin-Priority: 100\n\nPackage: gcc-16 gcc-16-* g++-16 g++-16-* cpp-16 cpp-16-* gcc-16-base libgcc-16-dev libstdc++-16-dev libstdc++6 libgcc-s1 libgomp1 libitm1 libatomic1 libquadmath0 libcc1-0 libasan8 libubsan1 liblsan0 libtsan2 libhwasan0\nPin: release o=Debian\nPin-Priority: 990\n' \
        > /etc/apt/preferences.d/debian \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc-16 \
        g++-16 \
        ninja-build \
        python3-pip \
    && pip3 install --break-system-packages cmake \
    && g++-16 --version \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-16 CXX=g++-16

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_TESTS=OFF \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS=OFF \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_DOCS=OFF \
    && cmake --build build --parallel

# Must track the builder: the binary links sid's libstdc++6 and libgcc-s1,
# which need a glibc no older than the one it was built against.
FROM ubuntu:26.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# A C++26 binary from GCC 16 needs that release's libstdc++ at run time, and
# stock 26.04 ships an older snapshot of it.
#
# The pin list is identical to the builder's on purpose. Trimming it to just
# the two libraries being installed is what broke this stage: sid's libstdc++6
# and libgcc-s1 declare Depends: gcc-16-base (= <same version>), and with
# gcc-16-base left outside the 990 pin apt held it at Ubuntu's snapshot and
# reported unmet dependencies. Only the install list needs to be narrow -- no
# compiler lands in the final image regardless of what the pin permits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        debian-archive-keyring \
    && echo "deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian sid main" \
        > /etc/apt/sources.list.d/debian-sid.list \
    && printf 'Package: *\nPin: release o=Debian\nPin-Priority: 100\n\nPackage: gcc-16 gcc-16-* g++-16 g++-16-* cpp-16 cpp-16-* gcc-16-base libgcc-16-dev libstdc++-16-dev libstdc++6 libgcc-s1 libgomp1 libitm1 libatomic1 libquadmath0 libcc1-0 libasan8 libubsan1 liblsan0 libtsan2 libhwasan0\nPin: release o=Debian\nPin-Priority: 990\n' \
        > /etc/apt/preferences.d/debian \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/src/cpp_package_boilerplate_cli /usr/local/bin/cpp_package_boilerplate

ENTRYPOINT ["/usr/local/bin/cpp_package_boilerplate"]

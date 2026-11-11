#!/usr/bin/env bash
# Sestaví libfreenect2 (ovladač Kinectu v2) a náš most kinect2_bridge.
# Vše se instaluje lokálně do third_party/ a build/, do systému se nic nekopíruje
# (výjimka: na Linuxu udev pravidlo, aby Kinect šel používat bez sudo).
#
#   ./scripts/install_kinect2.sh            # vše
#   ./scripts/install_kinect2.sh --bridge   # jen znovu sestavit most (po úpravě main.cpp)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/libfreenect2"
PREFIX="$ROOT/third_party/install"
BRIDGE_BUILD="$ROOT/build/kinect2_bridge"
OS="$(uname -s)"
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

build_bridge() {
  say "Sestavuji kinect2_bridge"
  cmake -S "$ROOT/native/kinect2_bridge" -B "$BRIDGE_BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX"
  cmake --build "$BRIDGE_BUILD" -j "$JOBS"
  say "Hotovo: $BRIDGE_BUILD/kinect2_bridge"
  echo "Test:  $BRIDGE_BUILD/kinect2_bridge --list    (vypíše sériové číslo Kinectu)"
}

if [[ "${1:-}" == "--bridge" ]]; then build_bridge; exit 0; fi

say "Závislosti"
EXTRA_CMAKE=()
if [[ "$OS" == "Darwin" ]]; then
  command -v brew >/dev/null || { echo "Chybí Homebrew (https://brew.sh)"; exit 1; }
  # Instalujeme jen chybějící balíčky a bez aktualizace ostatních (brew install je jinak upgraduje).
  missing=()
  for f in cmake pkg-config libusb glfw jpeg-turbo; do
    brew list --versions "$f" >/dev/null 2>&1 || missing+=("$f")
  done
  if ((${#missing[@]})); then
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1 brew install "${missing[@]}"
  fi
  JPEG="$(brew --prefix jpeg-turbo)"
  EXTRA_CMAKE+=(-DTurboJPEG_INCLUDE_DIRS="$JPEG/include"
                -DTurboJPEG_LIBRARIES="$JPEG/lib/libturbojpeg.dylib")
elif [[ "$OS" == "Linux" ]]; then
  if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake pkg-config git \
         libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev
  else
    echo "Nainstaluj ručně: cmake, pkg-config, libusb-1.0 (dev), turbojpeg (dev), glfw3 (dev)"
  fi
fi

if [[ ! -d "$SRC/.git" ]]; then
  say "Stahuji libfreenect2"
  git clone --depth 1 https://github.com/OpenKinect/libfreenect2.git "$SRC"
fi

say "Sestavuji libfreenect2"
# CMAKE_POLICY_VERSION_MINIMUM: libfreenect2 deklaruje velmi staré cmake_minimum_required,
# které CMake 4 jinak odmítne.
cmake -S "$SRC" -B "$SRC/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DENABLE_CUDA=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF \
      -DENABLE_CXX11=ON \
      -DBUILD_EXAMPLES=ON \
      "${EXTRA_CMAKE[@]+"${EXTRA_CMAKE[@]}"}"
cmake --build "$SRC/build" -j "$JOBS"
cmake --install "$SRC/build"

if [[ "$OS" == "Linux" ]]; then
  say "udev pravidlo pro Kinect v2 (vyžaduje sudo)"
  sudo cp "$SRC/platform/linux/udev/90-kinect2.rules" /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  echo "Odpoj a znovu zapoj Kinect."
fi

build_bridge

say "Ověření"
echo "Ukázkový program libfreenect2:  $SRC/build/bin/Protonect gl"
echo "Pískoviště s Kinectem:          .venv/bin/python -m sandbox run --sensor kinect2"

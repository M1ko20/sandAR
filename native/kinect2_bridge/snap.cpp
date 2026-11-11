// kinect2_snap — barevné snímky z Kinectu v2 srovnané s hloubkovou mřížkou (pro automatickou
// kalibraci projektoru: projektor promítá značky, barevná kamera je vidí, hloubka říká kde jsou).
//
// Protokol (stdin/stdout, binárně, little-endian):
//   start:  "K2SN" | u32 width=512 | u32 height=424 | f32 fx, fy, cx, cy   (hloubková kamera)
//   na každý řádek ze stdin (libovolný text) zahodí `skip` snímků (ať je obraz po změně
//   projekce čerstvý) a pošle:  "SNAP" | u8[w*h*3] BGR (registrovaná barva, 0 = bez dat)
//                               | f32[w*h] hloubka v mm (nezkreslená, stejná jako u kinect2_bridge)
//   EOF na stdin = konec.
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

static bool write_all(const void *data, size_t size) {
  return std::fwrite(data, 1, size, stdout) == size;
}

int main(int argc, char **argv) {
  int skip = 4;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--skip" && i + 1 < argc) skip = std::stoi(argv[++i]);
  }
#ifndef _WIN32
  const pid_t parent = getppid();  // stejný hlídač rodiče jako u kinect2_bridge
  std::thread([parent] {
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (getppid() != parent) std::_Exit(0);
    }
  }).detach();
#endif
  libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
  libfreenect2::Freenect2 freenect2;
  if (freenect2.enumerateDevices() == 0) {
    std::cerr << "kinect2_snap: Kinect v2 nenalezen\n";
    return 3;
  }
  libfreenect2::Freenect2Device *dev =
      freenect2.openDevice(freenect2.getDefaultDeviceSerialNumber(), new libfreenect2::CpuPacketPipeline());
  if (!dev) return 4;
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  if (!dev->start()) {
    std::cerr << "kinect2_snap: start streamu selhal\n";
    return 5;
  }
  auto ir = dev->getIrCameraParams();
  libfreenect2::Registration reg(ir, dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

  const uint32_t w = 512, h = 424;
  const float intr[4] = {ir.fx, ir.fy, ir.cx, ir.cy};
  write_all("K2SN", 4);
  write_all(&w, 4);
  write_all(&h, 4);
  write_all(intr, sizeof(intr));
  std::fflush(stdout);
  std::cerr << "kinect2_snap: pripraven\n";

  std::vector<uint8_t> bgr(w * h * 3);
  std::vector<float> depth(w * h);
  libfreenect2::FrameMap frames;
  std::string line;
  while (std::getline(std::cin, line)) {
    for (int i = 0; i <= skip; ++i) {
      if (!listener.waitForNewFrame(frames, 5000)) {
        std::cerr << "kinect2_snap: zadny snimek 5 s\n";
        std::_Exit(6);
      }
      if (i < skip) listener.release(frames);
    }
    reg.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth], &undistorted, &registered);
    const uint32_t *px = reinterpret_cast<const uint32_t *>(registered.data);
    const float *dp = reinterpret_cast<const float *>(undistorted.data);
    for (size_t i = 0; i < w * h; ++i) {
      uint32_t p = px[i];  // BGRX
      bgr[3 * i + 0] = p & 0xff;
      bgr[3 * i + 1] = (p >> 8) & 0xff;
      bgr[3 * i + 2] = (p >> 16) & 0xff;
      float d = dp[i];
      depth[i] = (std::isfinite(d) && d > 0.0f) ? d : 0.0f;
    }
    listener.release(frames);
    if (!write_all("SNAP", 4) || !write_all(bgr.data(), bgr.size()) ||
        !write_all(depth.data(), depth.size() * sizeof(float)))
      break;
    std::fflush(stdout);
  }
  std::thread([] {  // stop()/close() umí viset
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::_Exit(0);
  }).detach();
  dev->stop();
  dev->close();
  return 0;
}

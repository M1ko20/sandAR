// kinect2_bridge — tenký most mezi Kinectem v2 (libfreenect2) a Pythonem.
//
// Program otevře Kinect v2, spustí pouze hloubkový stream, odstraní zkreslení
// objektivu (Registration::undistortDepth) a posílá snímky na stdout v jednoduchém
// binárním formátu. Python (sandbox/sensor/kinect2.py) ho spouští jako podproces,
// takže chyba v ovladači nikdy neshodí hlavní aplikaci.
//
// Protokol (little-endian):
//   hlavička (jednou):  char[4] "K2BR", uint32 verze(=1), uint32 šířka, uint32 výška,
//                       float32 fx, fy, cx, cy   (parametry IR kamery PO undistortu)
//   každý snímek:       char[4] "FRME", uint32 pořadové číslo, float64 čas [s],
//                       float32[šířka*výška] hloubka v milimetrech (0 = neplatné)
//
// Diagnostika jde na stderr.

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

std::atomic<bool> g_running{true};

// Při ukončení: libfreenect2 umí zůstat viset ve stop()/close(), když Kinect neodpovídá.
// Proto po signálu spustíme pojistku, která proces po 3 s ukončí natvrdo.
void on_signal(int) {
  if (!g_running.exchange(false)) std::_Exit(130);  // druhý signál = okamžitě
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::_Exit(0);
  }).detach();
}

bool write_all(const void *data, size_t size) {
  const char *p = static_cast<const char *>(data);
  while (size > 0) {
    size_t n = std::fwrite(p, 1, size, stdout);
    if (n == 0) return false;  // čtenář (Python) skončil
    p += n;
    size -= n;
  }
  return true;
}

void usage() {
  std::cerr << "Pouziti: kinect2_bridge [--pipeline gl|cpu] [--serial SERIAL] [--list] [--frames N]\n";
}

libfreenect2::PacketPipeline *make_pipeline(const std::string &name) {
  if (name == "cpu") return new libfreenect2::CpuPacketPipeline();
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
  if (name == "gl") return new libfreenect2::OpenGLPacketPipeline();
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
  if (name == "cl") return new libfreenect2::OpenCLPacketPipeline();
#endif
  return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
  std::string pipeline_name = "gl";
  std::string serial;
  bool list_only = false;
  long max_frames = -1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--pipeline" && i + 1 < argc) pipeline_name = argv[++i];
    else if (a == "--serial" && i + 1 < argc) serial = argv[++i];
    else if (a == "--frames" && i + 1 < argc) max_frames = std::stol(argv[++i]);
    else if (a == "--list") list_only = true;
    else { usage(); return 2; }
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
#ifdef SIGPIPE
  std::signal(SIGPIPE, SIG_IGN);  // zápis do zavřené roury vrátí chybu místo pádu
#endif
#ifndef _WIN32
  // Hlídač rodiče: když Python skončí (i pádem nebo SIGKILL), most nesmí zůstat jako sirotek
  // a blokovat Kinect (LIBUSB_ERROR_BUSY). Zavřená roura nestačí – když Kinect zrovna
  // neposílá snímky, most na zápis vůbec nedojde.
  if (!list_only) {
    const pid_t parent = getppid();
    std::thread([parent] {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (getppid() != parent) std::_Exit(0);
      }
    }).detach();
  }
#endif

  libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));

  libfreenect2::Freenect2 freenect2;
  int count = freenect2.enumerateDevices();
  if (list_only) {
    for (int i = 0; i < count; ++i) std::cout << freenect2.getDeviceSerialNumber(i) << "\n";
    return 0;
  }
  if (count == 0) {
    std::cerr << "kinect2_bridge: Kinect v2 nenalezen (zkontroluj USB 3 a napajeni adapteru)\n";
    return 3;
  }
  if (serial.empty()) serial = freenect2.getDefaultDeviceSerialNumber();

  // Zkusíme požadovanou pipeline, při neúspěchu spadneme na CPU.
  libfreenect2::Freenect2Device *dev = nullptr;
  for (const std::string &name : {pipeline_name, std::string("cpu")}) {
    libfreenect2::PacketPipeline *pipeline = make_pipeline(name);
    if (!pipeline) {
      std::cerr << "kinect2_bridge: pipeline '" << name << "' neni v teto sestaveni dostupna\n";
      continue;
    }
    dev = freenect2.openDevice(serial, pipeline);  // dev přebírá vlastnictví pipeline
    if (dev) {
      std::cerr << "kinect2_bridge: zarizeni " << serial << " otevreno, pipeline=" << name << "\n";
      break;
    }
    std::cerr << "kinect2_bridge: otevreni s pipeline '" << name << "' selhalo\n";
  }
  if (!dev) return 4;

  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
  dev->setIrAndDepthFrameListener(&listener);
  if (!dev->startStreams(false, true)) {
    std::cerr << "kinect2_bridge: start streamu selhal\n";
    return 5;
  }
  std::cerr << "kinect2_bridge: stream hloubky spusten\n";

  libfreenect2::Freenect2Device::IrCameraParams ir = dev->getIrCameraParams();
  libfreenect2::Registration registration(ir, dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4);

  const uint32_t width = 512, height = 424, version = 1;
  const float intr[4] = {ir.fx, ir.fy, ir.cx, ir.cy};
  bool ok = write_all("K2BR", 4) && write_all(&version, 4) && write_all(&width, 4) &&
            write_all(&height, 4) && write_all(intr, sizeof(intr));
  std::fflush(stdout);

  std::vector<float> out(width * height);
  libfreenect2::FrameMap frames;
  uint32_t seq = 0;
  int stalls = 0;
  const auto t0 = std::chrono::steady_clock::now();

  while (ok && g_running && (max_frames < 0 || seq < max_frames)) {
    if (!listener.waitForNewFrame(frames, 2000)) {
      if (++stalls >= 5) {
        std::cerr << "kinect2_bridge: Kinect 10 s neposlal snimek - odpoj a znovu zapoj jeho napajeni\n";
        std::_Exit(6);  // stop()/close() by u zaseknutého Kinectu visel
      }
      std::cerr << "kinect2_bridge: zadny snimek 2 s, cekam dal...\n";
      continue;
    }
    stalls = 0;
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
    registration.undistortDepth(depth, &undistorted);

    const float *src = reinterpret_cast<const float *>(undistorted.data);
    for (size_t i = 0; i < out.size(); ++i) {
      float v = src[i];
      out[i] = (std::isfinite(v) && v > 0.0f) ? v : 0.0f;
    }
    listener.release(frames);

    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ok = write_all("FRME", 4) && write_all(&seq, 4) && write_all(&t, 8) &&
         write_all(out.data(), out.size() * sizeof(float));
    std::fflush(stdout);
    ++seq;
  }

  // Python skončil (zavřená roura) nebo došel limit snímků: stop()/close() umí viset,
  // a pak by most jako sirotek dál blokoval Kinect (LIBUSB_ERROR_BUSY). Stejná pojistka jako u signálu.
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::_Exit(0);
  }).detach();
  dev->stop();
  dev->close();
  delete dev;
  return 0;
}

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <cuda_gl_interop.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <arpa/inet.h>
#include <cstdio>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <jpeglib.h>
#include <linux/videodev2.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

#ifndef TRANSFORMIRROR_WIDTH
#define TRANSFORMIRROR_WIDTH 1024
#endif
#ifndef TRANSFORMIRROR_HEIGHT
#define TRANSFORMIRROR_HEIGHT 1024
#endif

constexpr int kWidth = TRANSFORMIRROR_WIDTH;
constexpr int kHeight = TRANSFORMIRROR_HEIGHT;
constexpr int kMaxDenoiseSteps = 8;
constexpr int kImageElems = 1 * 3 * kHeight * kWidth;
constexpr int kRgbElems = kHeight * kWidth * 3;
constexpr int kLatentWidth = kWidth / 8;
constexpr int kLatentHeight = kHeight / 8;
constexpr int kLatentElems = 1 * 4 * kLatentHeight * kLatentWidth;
constexpr int kPromptElems = 1 * 77 * 2048;
constexpr int kTextElems = 1 * 1280;
constexpr int kTimeElems = 1 * 6;
constexpr int kTimestepElems = 1;
constexpr int kParamElems = 4;  // sigma, inv_sigma_scale, scaling, inv_scaling
constexpr int kStepParamElems = kMaxDenoiseSteps * 3;  // sigma, next_sigma, inv_sigma_scale

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err__) + " at " +       \
                                     __FILE__ + ":" + std::to_string(__LINE__));\
        }                                                                       \
    } while (0)

std::atomic<bool> g_running{true};

void on_signal(int) {
    g_running.store(false);
}

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

struct Args {
    std::string engine_dir = "onnx";
    std::string asset_dir = "cpp_assets";
    std::string web_root = "web";
    std::string conditioning_backend = "worker";
    std::string conditioning_script = "conditioning_worker.py";
    std::string conditioning_socket;
    std::string camera_device = "/dev/video0";
    std::string capture_backend = "v4l2";
    std::string display_backend = "gl";
    std::string python = ".venv/bin/python";
    int capture_width = 1920;
    int capture_height = 1080;
    int camera_fps = 30;
    int http_port = 8080;
    int osc_port = 9000;
    int max_frames = 0;
    int main_core = -1;
    int capture_core = -1;
    int http_core = -1;
    int osc_core = -1;
    int reload_core = -1;
    int rt_priority = 0;
    bool lock_memory = false;
    bool no_display = false;
    bool has_initial_prompt = false;
    bool has_initial_seed = false;
    bool has_initial_strength = false;
    bool has_initial_steps = false;
    bool has_initial_blend = false;
    bool has_initial_passthrough = false;
    bool has_initial_use_latest_frame = false;
    std::string initial_prompt;
    int initial_seed = 0;
    float initial_strength = 0.7f;
    int initial_steps = 2;
    float initial_blend = 1.0f;
    bool initial_passthrough = false;
    bool initial_use_latest_frame = true;
};

struct AppState {
    std::string prompt = "a cinematic mirror portrait, detailed face, luminous color, sharp focus";
    int seed = 0;
    float strength = 0.7f;
    int steps = 2;
    float blend = 1.0f;
    bool passthrough = false;
    bool use_latest_frame = true;
    int width = kWidth;
    int height = kHeight;
    int http_port = 8080;
    int osc_port = 9000;
    int camera_source_width = 0;
    int camera_source_height = 0;
    int camera_crop_width = kWidth;
    int camera_crop_height = kHeight;
    int display_width = kWidth;
    int display_height = kHeight;
    double camera_fps = 0.0;
    double fps = 0.0;
    double frame_ms = 0.0;
    double capture_ms = 0.0;
    double display_ms = 0.0;
    double loop_ms = 0.0;
    double conditioning_ms = 0.0;
    int queued_frames = 0;
    uint64_t dropped_frames = 0;
    std::string status = "starting";
    bool reload_requested = false;
    bool resolution_rebuild_requested = false;
    bool resolution_rebuild_active = false;
    int requested_width = kWidth;
    int requested_height = kHeight;
};

std::mutex g_state_mutex;
std::condition_variable g_reload_cv;
std::condition_variable g_resolution_cv;
AppState g_state;
std::atomic<bool> g_reexec_requested{false};
std::mutex g_reexec_mutex;
std::string g_reexec_binary;
std::string g_reexec_engine_dir;
std::string g_reexec_asset_dir;

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

std::string current_working_directory() {
    char buf[4096];
    if (!getcwd(buf, sizeof(buf))) return ".";
    return std::string(buf);
}

int normalized_resolution(int value) {
    int rounded = static_cast<int>(std::lround(static_cast<double>(value) / 32.0) * 32.0);
    return std::clamp(rounded, 256, 1280);
}

void set_close_on_exec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}

void warn_errno(const std::string& what) {
    std::cerr << what << " failed: " << std::strerror(errno) << "\n";
}

void set_current_thread_name(const char* name) {
    pthread_setname_np(pthread_self(), name);
}

void configure_current_thread(const char* name, int core, int rt_priority) {
    set_current_thread_name(name);
    if (core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        if (rc != 0) {
            std::cerr << "pin " << name << " to CPU " << core << " failed: "
                      << std::strerror(rc) << "\n";
        }
    }
    if (rt_priority > 0) {
        sched_param param{};
        param.sched_priority = rt_priority;
        int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (rc != 0) {
            std::cerr << "set SCHED_FIFO " << name << " priority " << rt_priority
                      << " failed: " << std::strerror(rc) << "\n";
        }
    }
}

void try_lock_process_memory() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) return;
    int first_errno = errno;
    if (mlockall(MCL_CURRENT) == 0) {
        std::cerr << "mlockall(MCL_CURRENT|MCL_FUTURE) failed: "
                  << std::strerror(first_errno)
                  << "; locked current mappings only\n";
        return;
    }
    warn_errno("mlockall");
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--engine-dir") args.engine_dir = val("--engine-dir");
        else if (key == "--asset-dir") args.asset_dir = val("--asset-dir");
        else if (key == "--web-root") args.web_root = val("--web-root");
        else if (key == "--conditioning-backend") args.conditioning_backend = val("--conditioning-backend");
        else if (key == "--conditioning-script") args.conditioning_script = val("--conditioning-script");
        else if (key == "--conditioning-socket") args.conditioning_socket = val("--conditioning-socket");
        else if (key == "--camera-device") args.camera_device = val("--camera-device");
        else if (key == "--capture-backend") args.capture_backend = val("--capture-backend");
        else if (key == "--display-backend") args.display_backend = val("--display-backend");
        else if (key == "--python") args.python = val("--python");
        else if (key == "--capture-width") args.capture_width = std::stoi(val("--capture-width"));
        else if (key == "--capture-height") args.capture_height = std::stoi(val("--capture-height"));
        else if (key == "--camera-fps") args.camera_fps = std::stoi(val("--camera-fps"));
        else if (key == "--http-port") args.http_port = std::stoi(val("--http-port"));
        else if (key == "--osc-port") args.osc_port = std::stoi(val("--osc-port"));
        else if (key == "--max-frames") args.max_frames = std::stoi(val("--max-frames"));
        else if (key == "--main-core") args.main_core = std::stoi(val("--main-core"));
        else if (key == "--capture-core") args.capture_core = std::stoi(val("--capture-core"));
        else if (key == "--http-core") args.http_core = std::stoi(val("--http-core"));
        else if (key == "--osc-core") args.osc_core = std::stoi(val("--osc-core"));
        else if (key == "--reload-core") args.reload_core = std::stoi(val("--reload-core"));
        else if (key == "--rt-priority") args.rt_priority = std::stoi(val("--rt-priority"));
        else if (key == "--initial-prompt") { args.initial_prompt = val("--initial-prompt"); args.has_initial_prompt = true; }
        else if (key == "--initial-seed") { args.initial_seed = std::stoi(val("--initial-seed")); args.has_initial_seed = true; }
        else if (key == "--initial-strength") { args.initial_strength = std::stof(val("--initial-strength")); args.has_initial_strength = true; }
        else if (key == "--initial-steps") { args.initial_steps = std::stoi(val("--initial-steps")); args.has_initial_steps = true; }
        else if (key == "--initial-blend") { args.initial_blend = std::stof(val("--initial-blend")); args.has_initial_blend = true; }
        else if (key == "--initial-passthrough") { args.initial_passthrough = std::stoi(val("--initial-passthrough")) != 0; args.has_initial_passthrough = true; }
        else if (key == "--initial-use-latest-frame") { args.initial_use_latest_frame = std::stoi(val("--initial-use-latest-frame")) != 0; args.has_initial_use_latest_frame = true; }
        else if (key == "--lock-memory") args.lock_memory = true;
        else if (key == "--realtime") {
            args.lock_memory = true;
            args.rt_priority = 10;
        } else if (key == "--no-display") {
            args.no_display = true;
            args.display_backend = "none";
        }
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: transformirror_fast_app [--camera-device /dev/video0]\n"
                << "       [--engine-dir onnx] [--asset-dir cpp_assets] [--web-root web]\n"
                << "       [--conditioning-backend worker|script] [--conditioning-script conditioning_worker.py]\n"
                << "       [--capture-backend v4l2|ffmpeg] [--display-backend gl|ffplay|none]\n"
                << "       [--http-port 8080] [--osc-port 9000] [--no-display]\n"
                << "       [--realtime] [--lock-memory] [--rt-priority N]\n"
                << "       [--main-core N] [--capture-core N] [--http-core N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.capture_backend != "v4l2" && args.capture_backend != "ffmpeg") {
        throw std::runtime_error("--capture-backend must be v4l2 or ffmpeg");
    }
    if (args.display_backend != "gl" && args.display_backend != "ffplay" && args.display_backend != "none") {
        throw std::runtime_error("--display-backend must be gl, ffplay, or none");
    }
    if (args.conditioning_backend != "worker" && args.conditioning_backend != "script") {
        throw std::runtime_error("--conditioning-backend must be worker or script");
    }
    if (args.conditioning_socket.empty()) {
        args.conditioning_socket = "/tmp/transformirror_conditioning_" +
            std::to_string(getuid()) + "_" + std::to_string(args.http_port) + ".sock";
    }
    if (args.rt_priority < 0) args.rt_priority = 0;
    if (args.rt_priority > 95) args.rt_priority = 95;
    return args;
}

void apply_initial_state(const Args& args) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (args.has_initial_prompt) g_state.prompt = args.initial_prompt;
    if (args.has_initial_seed) g_state.seed = args.initial_seed;
    if (args.has_initial_strength) g_state.strength = std::clamp(args.initial_strength, 0.0f, 1.0f);
    if (args.has_initial_steps) g_state.steps = std::clamp(args.initial_steps, 2, 8);
    if (args.has_initial_blend) g_state.blend = std::clamp(args.initial_blend, 0.0f, 1.0f);
    if (args.has_initial_passthrough) g_state.passthrough = args.initial_passthrough;
    if (args.has_initial_use_latest_frame) g_state.use_latest_frame = args.initial_use_latest_frame;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("failed to open " + path);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size)) throw std::runtime_error("failed to read " + path);
    return data;
}

std::string read_text_file_or_empty(const std::string& path) {
    std::ifstream file(path);
    if (!file) return "";
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t elems = 0) { reset(elems); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void reset(size_t elems) {
        elems_ = elems;
        if (elems_) CHECK_CUDA(cudaMalloc(&ptr_, elems_ * sizeof(T)));
    }
    T* get() const { return ptr_; }
    size_t bytes() const { return elems_ * sizeof(T); }
  private:
    T* ptr_ = nullptr;
    size_t elems_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(size_t elems = 0) { reset(elems); }
    ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void reset(size_t elems) {
        elems_ = elems;
        if (elems_) CHECK_CUDA(cudaMallocHost(&ptr_, elems_ * sizeof(T)));
    }
    T* get() const { return ptr_; }
    size_t bytes() const { return elems_ * sizeof(T); }
  private:
    T* ptr_ = nullptr;
    size_t elems_ = 0;
};

struct AssetBlob {
    std::vector<char> noise;
    std::vector<char> prompt;
    std::vector<char> text;
    std::vector<char> time;
    std::vector<char> timestep;
    std::vector<char> params;
    std::vector<char> timesteps;
    std::vector<char> step_params;
    std::vector<char> step_count;
};

std::vector<char> read_exact_asset(const std::string& path, size_t bytes) {
    std::vector<char> data = read_file(path);
    if (data.size() != bytes) {
        throw std::runtime_error(path + " has " + std::to_string(data.size()) +
                                 " bytes, expected " + std::to_string(bytes));
    }
    return data;
}

bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

struct ResolutionEntry {
    int width = 0;
    int height = 0;
};

bool parse_resolution_key(const std::string& name, int& width, int& height) {
    size_t x = name.find('x');
    if (x == std::string::npos || x == 0 || x + 1 >= name.size()) return false;
    try {
        width = std::stoi(name.substr(0, x));
        height = std::stoi(name.substr(x + 1));
    } catch (...) {
        return false;
    }
    return width > 0 && height > 0;
}

bool cached_resolution_complete(const std::string& root, const std::string& key) {
    std::string engine_dir = join_path(join_path(root, "trt_engines"), key);
    std::string build_dir = join_path(join_path(root, "cpp"), "build_" + key);
    return file_exists(join_path(engine_dir, "taesdxl_encode.plan")) &&
           file_exists(join_path(engine_dir, "taesdxl_decode.plan")) &&
           file_exists(join_path(engine_dir, "sdxl_turbo_unet.plan")) &&
           file_exists(join_path(build_dir, "transformirror_fast_app"));
}

std::string cached_resolutions_json() {
    std::string root = current_working_directory();
    std::string engine_root = join_path(root, "trt_engines");
    std::vector<ResolutionEntry> entries;
    DIR* dir = opendir(engine_root.c_str());
    if (dir) {
        while (dirent* ent = readdir(dir)) {
            std::string name = ent->d_name;
            int width = 0;
            int height = 0;
            if (!parse_resolution_key(name, width, height)) continue;
            if (!cached_resolution_complete(root, name)) continue;
            entries.push_back({width, height});
        }
        closedir(dir);
    }
    std::sort(entries.begin(), entries.end(), [](const ResolutionEntry& a, const ResolutionEntry& b) {
        if (a.width * a.height != b.width * b.height) return a.width * a.height > b.width * b.height;
        if (a.width != b.width) return a.width > b.width;
        return a.height > b.height;
    });
    entries.erase(std::unique(entries.begin(), entries.end(), [](const ResolutionEntry& a, const ResolutionEntry& b) {
        return a.width == b.width && a.height == b.height;
    }), entries.end());

    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i) out << ",";
        out << "{\"width\":" << entries[i].width << ",\"height\":" << entries[i].height << "}";
    }
    out << "]";
    return out.str();
}

AssetBlob read_asset_blob(const std::string& asset_dir) {
    AssetBlob assets;
    assets.noise = read_exact_asset(join_path(asset_dir, "noise.fp16"), kLatentElems * sizeof(__half));
    assets.prompt = read_exact_asset(join_path(asset_dir, "prompt_embeds.fp16"), kPromptElems * sizeof(__half));
    assets.text = read_exact_asset(join_path(asset_dir, "text_embeds.fp16"), kTextElems * sizeof(__half));
    assets.time = read_exact_asset(join_path(asset_dir, "time_ids.fp16"), kTimeElems * sizeof(__half));
    assets.timestep = read_exact_asset(join_path(asset_dir, "timestep.f32"), kTimestepElems * sizeof(float));
    assets.params = read_exact_asset(join_path(asset_dir, "params.f32"), kParamElems * sizeof(float));
    std::string timesteps_path = join_path(asset_dir, "timesteps.f32");
    std::string step_params_path = join_path(asset_dir, "step_params.f32");
    std::string step_count_path = join_path(asset_dir, "step_count.i32");
    if (file_exists(timesteps_path) && file_exists(step_params_path) && file_exists(step_count_path)) {
        assets.timesteps = read_exact_asset(timesteps_path, kMaxDenoiseSteps * sizeof(float));
        assets.step_params = read_exact_asset(step_params_path, kStepParamElems * sizeof(float));
        assets.step_count = read_exact_asset(step_count_path, sizeof(int32_t));
    } else {
        assets.timesteps.assign(kMaxDenoiseSteps * sizeof(float), 0);
        assets.step_params.assign(kStepParamElems * sizeof(float), 0);
        assets.step_count.assign(sizeof(int32_t), 0);
        std::memcpy(assets.timesteps.data(), assets.timestep.data(), sizeof(float));
        float params[kParamElems] = {};
        std::memcpy(params, assets.params.data(), sizeof(params));
        float step_params[3] = {params[0], 0.0f, params[1]};
        std::memcpy(assets.step_params.data(), step_params, sizeof(step_params));
        int32_t count = 1;
        std::memcpy(assets.step_count.data(), &count, sizeof(count));
    }
    return assets;
}

class TrtEngine {
  public:
    TrtEngine(Logger& logger, const std::string& path) {
        auto plan = read_file(path);
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_) throw std::runtime_error("failed to create TensorRT runtime");
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) throw std::runtime_error("failed to deserialize " + path);
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("failed to create context " + path);
    }
    void bind(const char* name, void* ptr) {
        if (!context_->setTensorAddress(name, ptr)) {
            throw std::runtime_error(std::string("failed to bind ") + name);
        }
    }
    void enqueue(cudaStream_t stream) {
        if (!context_->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueue failed");
    }
  private:
    struct D { template <typename T> void operator()(T* ptr) const { delete ptr; } };
    std::unique_ptr<nvinfer1::IRuntime, D> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, D> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, D> context_;
};

__global__ void preprocess_rgb_kernel(const uint8_t* rgb, __half* image, int n_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;
    int src = idx * 3;
    image[idx] = __float2half_rn(static_cast<float>(rgb[src]) / 127.5f - 1.0f);
    image[n_pixels + idx] = __float2half_rn(static_cast<float>(rgb[src + 1]) / 127.5f - 1.0f);
    image[2 * n_pixels + idx] = __float2half_rn(static_cast<float>(rgb[src + 2]) / 127.5f - 1.0f);
}

__global__ void init_latents_kernel(
    const __half* encoded, const __half* noise, const float* step_params, const float* params,
    __half* latents, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sigma = step_params[0];
    float scaling = params[2];
    float latent = __half2float(encoded[idx]) * scaling + __half2float(noise[idx]) * sigma;
    latents[idx] = __float2half_rn(latent);
}

__global__ void prepare_model_input_kernel(
    const __half* latents, const float* step_params, __half* unet_input, int step, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_sigma_scale = step_params[step * 3 + 2];
    unet_input[idx] = __float2half_rn(__half2float(latents[idx]) * inv_sigma_scale);
}

__global__ void scheduler_step_kernel(
    __half* latents, const __half* noise_pred, const float* step_params, int step, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sigma = step_params[step * 3];
    float next_sigma = step_params[step * 3 + 1];
    float latent = __half2float(latents[idx]) + (next_sigma - sigma) * __half2float(noise_pred[idx]);
    latents[idx] = __float2half_rn(latent);
}

__global__ void prepare_decode_kernel(
    const __half* latents, const float* params,
    __half* decode_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_scaling = params[3];
    decode_input[idx] = __float2half_rn(__half2float(latents[idx]) * inv_scaling);
}

__global__ void compose_rgb_kernel(
    const uint8_t* src_rgb, const __half* decoded, uint8_t* dst_rgb,
    const float* blend_value, const uint8_t* passthrough_value, int n_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;
    float blend = passthrough_value[0] ? 0.0f : blend_value[0];
    blend = fminf(1.0f, fmaxf(0.0f, blend));
    int base = idx * 3;
    float source_r = static_cast<float>(src_rgb[base]) / 255.0f;
    float source_g = static_cast<float>(src_rgb[base + 1]) / 255.0f;
    float source_b = static_cast<float>(src_rgb[base + 2]) / 255.0f;
    float out_r = fminf(1.0f, fmaxf(0.0f, __half2float(decoded[idx]) * 0.5f + 0.5f));
    float out_g = fminf(1.0f, fmaxf(0.0f, __half2float(decoded[n_pixels + idx]) * 0.5f + 0.5f));
    float out_b = fminf(1.0f, fmaxf(0.0f, __half2float(decoded[2 * n_pixels + idx]) * 0.5f + 0.5f));
    dst_rgb[base] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, 255.0f * (source_r * (1.0f - blend) + out_r * blend))));
    dst_rgb[base + 1] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, 255.0f * (source_g * (1.0f - blend) + out_g * blend))));
    dst_rgb[base + 2] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, 255.0f * (source_b * (1.0f - blend) + out_b * blend))));
}

class FastPipeline {
  public:
    FastPipeline(const Args& args, Logger& logger)
        : args_(args),
          encode_(logger, join_path(args.engine_dir, "taesdxl_encode.plan")),
          unet_(logger, join_path(args.engine_dir, "sdxl_turbo_unet.plan")),
          decode_(logger, join_path(args.engine_dir, "taesdxl_decode.plan")),
          d_src_rgb_(kRgbElems), d_dst_rgb_(kRgbElems), d_image_(kImageElems),
          d_encoded_(kLatentElems), d_latents_(kLatentElems), d_unet_input_(kLatentElems),
          d_noise_pred_(kLatentElems), d_decode_input_(kLatentElems), d_decoded_(kImageElems),
          d_noise_(kLatentElems), d_prompt_(kPromptElems), d_text_(kTextElems), d_time_(kTimeElems),
          d_timestep_(kTimestepElems), d_timesteps_(kMaxDenoiseSteps),
          d_params_(kParamElems), d_step_params_(kStepParamElems), d_blend_(1), d_passthrough_(1),
          h_src_rgb_(kRgbElems), h_dst_rgb_(kRgbElems) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&start_event_));
        CHECK_CUDA(cudaEventCreate(&end_event_));
        bind_engines();
        reload_assets();
        update_blend(1.0f, false);
        capture_graph();
    }

    ~FastPipeline() {
        if (end_event_) cudaEventDestroy(end_event_);
        if (start_event_) cudaEventDestroy(start_event_);
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    uint8_t* host_input() const { return h_src_rgb_.get(); }
    uint8_t* host_output() const { return h_dst_rgb_.get(); }
    const uint8_t* device_output() const { return d_dst_rgb_.get(); }
    size_t rgb_bytes() const { return h_src_rgb_.bytes(); }

    void update_blend(float blend, bool passthrough) {
        uint8_t pass = passthrough ? 1 : 0;
        CHECK_CUDA(cudaMemcpyAsync(d_blend_.get(), &blend, sizeof(float), cudaMemcpyHostToDevice, stream_));
        CHECK_CUDA(cudaMemcpyAsync(d_passthrough_.get(), &pass, sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));
    }

    void reload_assets() {
        AssetBlob assets = read_asset_blob(args_.asset_dir);
        upload_assets(assets);
    }

    void upload_assets(const AssetBlob& assets) {
        std::lock_guard<std::mutex> lock(asset_mutex_);
        copy_asset(assets.noise, d_noise_.get(), d_noise_.bytes());
        copy_asset(assets.prompt, d_prompt_.get(), d_prompt_.bytes());
        copy_asset(assets.text, d_text_.get(), d_text_.bytes());
        copy_asset(assets.time, d_time_.get(), d_time_.bytes());
        copy_asset(assets.timestep, d_timestep_.get(), d_timestep_.bytes());
        copy_asset(assets.params, d_params_.get(), d_params_.bytes());
        copy_asset(assets.timesteps, d_timesteps_.get(), d_timesteps_.bytes());
        copy_asset(assets.step_params, d_step_params_.get(), d_step_params_.bytes());
        int32_t count = 1;
        if (assets.step_count.size() == sizeof(int32_t)) {
            std::memcpy(&count, assets.step_count.data(), sizeof(count));
        }
        step_count_ = std::clamp(static_cast<int>(count), 1, kMaxDenoiseSteps);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    float process_frame(bool need_host_output) {
        update_state_controls();
        CHECK_CUDA(cudaEventRecord(start_event_, stream_));
        CHECK_CUDA(cudaMemcpyAsync(d_src_rgb_.get(), h_src_rgb_.get(), h_src_rgb_.bytes(), cudaMemcpyHostToDevice, stream_));
        {
            std::lock_guard<std::mutex> lock(asset_mutex_);
            if (step_count_ == 1) {
                CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_));
            } else {
                enqueue_dynamic_body(step_count_);
            }
        }
        if (need_host_output) {
            CHECK_CUDA(cudaMemcpyAsync(h_dst_rgb_.get(), d_dst_rgb_.get(), h_dst_rgb_.bytes(), cudaMemcpyDeviceToHost, stream_));
        }
        CHECK_CUDA(cudaEventRecord(end_event_, stream_));
        CHECK_CUDA(cudaEventSynchronize(end_event_));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start_event_, end_event_));
        return elapsed;
    }

  private:
    void bind_engines() {
        encode_.bind("image", d_image_.get());
        encode_.bind("latents", d_encoded_.get());
        unet_.bind("sample", d_unet_input_.get());
        unet_.bind("timestep", d_timestep_.get());
        unet_.bind("encoder_hidden_states", d_prompt_.get());
        unet_.bind("text_embeds", d_text_.get());
        unet_.bind("time_ids", d_time_.get());
        unet_.bind("noise_pred", d_noise_pred_.get());
        decode_.bind("latents", d_decode_input_.get());
        decode_.bind("image", d_decoded_.get());
    }

    void copy_asset(const std::vector<char>& data, void* device_ptr, size_t bytes) {
        if (data.size() != bytes) {
            throw std::runtime_error("asset blob has " + std::to_string(data.size()) +
                                     " bytes, expected " + std::to_string(bytes));
        }
        CHECK_CUDA(cudaMemcpyAsync(device_ptr, data.data(), bytes, cudaMemcpyHostToDevice, stream_));
    }

    void enqueue_graph_body() {
        constexpr int block = 256;
        constexpr int pixels = kWidth * kHeight;
        constexpr int pixel_grid = (pixels + block - 1) / block;
        constexpr int latent_grid = (kLatentElems + block - 1) / block;
        preprocess_rgb_kernel<<<pixel_grid, block, 0, stream_>>>(d_src_rgb_.get(), d_image_.get(), pixels);
        encode_.enqueue(stream_);
        init_latents_kernel<<<latent_grid, block, 0, stream_>>>(
            d_encoded_.get(), d_noise_.get(), d_step_params_.get(), d_params_.get(), d_latents_.get(), kLatentElems);
        prepare_model_input_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_step_params_.get(), d_unet_input_.get(), 0, kLatentElems);
        CHECK_CUDA(cudaMemcpyAsync(d_timestep_.get(), d_timesteps_.get(), sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        unet_.enqueue(stream_);
        scheduler_step_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_noise_pred_.get(), d_step_params_.get(), 0, kLatentElems);
        prepare_decode_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_params_.get(), d_decode_input_.get(), kLatentElems);
        decode_.enqueue(stream_);
        compose_rgb_kernel<<<pixel_grid, block, 0, stream_>>>(
            d_src_rgb_.get(), d_decoded_.get(), d_dst_rgb_.get(), d_blend_.get(), d_passthrough_.get(), pixels);
    }

    void enqueue_dynamic_body(int step_count) {
        constexpr int block = 256;
        constexpr int pixels = kWidth * kHeight;
        constexpr int pixel_grid = (pixels + block - 1) / block;
        constexpr int latent_grid = (kLatentElems + block - 1) / block;
        preprocess_rgb_kernel<<<pixel_grid, block, 0, stream_>>>(d_src_rgb_.get(), d_image_.get(), pixels);
        encode_.enqueue(stream_);
        init_latents_kernel<<<latent_grid, block, 0, stream_>>>(
            d_encoded_.get(), d_noise_.get(), d_step_params_.get(), d_params_.get(), d_latents_.get(), kLatentElems);
        for (int step = 0; step < step_count; ++step) {
            prepare_model_input_kernel<<<latent_grid, block, 0, stream_>>>(
                d_latents_.get(), d_step_params_.get(), d_unet_input_.get(), step, kLatentElems);
            CHECK_CUDA(cudaMemcpyAsync(d_timestep_.get(), d_timesteps_.get() + step, sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            unet_.enqueue(stream_);
            scheduler_step_kernel<<<latent_grid, block, 0, stream_>>>(
                d_latents_.get(), d_noise_pred_.get(), d_step_params_.get(), step, kLatentElems);
        }
        prepare_decode_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_params_.get(), d_decode_input_.get(), kLatentElems);
        decode_.enqueue(stream_);
        compose_rgb_kernel<<<pixel_grid, block, 0, stream_>>>(
            d_src_rgb_.get(), d_decoded_.get(), d_dst_rgb_.get(), d_blend_.get(), d_passthrough_.get(), pixels);
    }

    void capture_graph() {
        for (int i = 0; i < 5; ++i) enqueue_graph_body();
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        cudaGraph_t graph = nullptr;
        CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
        enqueue_graph_body();
        CHECK_CUDA(cudaStreamEndCapture(stream_, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0));
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    void update_state_controls() {
        float blend = 1.0f;
        bool passthrough = false;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            blend = g_state.blend;
            passthrough = g_state.passthrough;
        }
        update_blend(blend, passthrough);
    }

    const Args& args_;
    TrtEngine encode_, unet_, decode_;
    DeviceBuffer<uint8_t> d_src_rgb_, d_dst_rgb_;
    DeviceBuffer<__half> d_image_, d_encoded_, d_latents_, d_unet_input_, d_noise_pred_, d_decode_input_, d_decoded_;
    DeviceBuffer<__half> d_noise_, d_prompt_, d_text_, d_time_;
    DeviceBuffer<float> d_timestep_, d_timesteps_, d_params_, d_step_params_, d_blend_;
    DeviceBuffer<uint8_t> d_passthrough_;
    PinnedBuffer<uint8_t> h_src_rgb_, h_dst_rgb_;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t end_event_ = nullptr;
    int step_count_ = 1;
    std::mutex asset_mutex_;
};

std::string ffmpeg_capture_cmd(const Args& args) {
    std::ostringstream cmd;
    cmd << "ffmpeg -hide_banner -loglevel error "
        << "-fflags nobuffer -flags low_delay "
        << "-f v4l2 -thread_queue_size 1 -input_format mjpeg -video_size " << args.capture_width << "x" << args.capture_height
        << " -framerate " << args.camera_fps
        << " -i " << shell_quote(args.camera_device)
        << " -vf 'crop=min(iw\\,ih*" << kWidth << "/" << kHeight << "):min(ih\\,iw*" << kHeight << "/" << kWidth
        << "):(iw-ow)/2:(ih-oh)/2,scale=" << kWidth << ":" << kHeight << "'"
        << " -pix_fmt rgb24 -f rawvideo -";
    return cmd.str();
}

std::string ffplay_display_cmd() {
    std::ostringstream cmd;
    cmd << "ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay "
        << "-f rawvideo -pixel_format rgb24 -video_size " << kWidth << "x" << kHeight
        << " -framerate 30 -i pipe:0 -fs -noborder -autoexit";
    return cmd.str();
}

class GlDisplay {
  public:
    explicit GlDisplay(size_t rgb_bytes) : rgb_bytes_(rgb_bytes) {
        open_display();
        create_window();
        create_gl_objects();
        register_cuda_pbo();
        std::cerr << "display: glx cuda-pbo " << window_width_ << "x" << window_height_ << "\n";
    }

    ~GlDisplay() {
        if (cuda_resource_) cudaGraphicsUnregisterResource(cuda_resource_);
        if (pbo_) glDeleteBuffers(1, &pbo_);
        if (texture_) glDeleteTextures(1, &texture_);
        if (context_) {
            glXMakeCurrent(display_, None, nullptr);
            glXDestroyContext(display_, context_);
        }
        if (window_) XDestroyWindow(display_, window_);
        if (colormap_) XFreeColormap(display_, colormap_);
        if (visual_) XFree(visual_);
        if (display_) XCloseDisplay(display_);
    }

    GlDisplay(const GlDisplay&) = delete;
    GlDisplay& operator=(const GlDisplay&) = delete;

    int width() const { return window_width_; }
    int height() const { return window_height_; }

    bool render(const uint8_t* device_rgb) {
        pump_events();
        if (!g_running.load()) return false;

        void* mapped = nullptr;
        size_t mapped_size = 0;
        CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_resource_, 0));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&mapped, &mapped_size, cuda_resource_));
        if (mapped_size < rgb_bytes_) throw std::runtime_error("mapped GL PBO is smaller than RGB frame");
        CHECK_CUDA(cudaMemcpy(mapped, device_rgb, rgb_bytes_, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_resource_, 0));

        int viewport_x = 0;
        int viewport_y = 0;
        int viewport_w = window_width_;
        int viewport_h = window_height_;
        const double target_aspect = static_cast<double>(kWidth) / static_cast<double>(kHeight);
        const double window_aspect = static_cast<double>(window_width_) / static_cast<double>(window_height_);
        if (window_aspect > target_aspect) {
            viewport_w = std::max(1, static_cast<int>(std::lround(window_height_ * target_aspect)));
            viewport_x = (window_width_ - viewport_w) / 2;
        } else if (window_aspect < target_aspect) {
            viewport_h = std::max(1, static_cast<int>(std::lround(window_width_ / target_aspect)));
            viewport_y = (window_height_ - viewport_h) / 2;
        }

        glViewport(0, 0, window_width_, window_height_);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(viewport_x, viewport_y, viewport_w, viewport_h);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_TRIANGLE_STRIP);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glEnd();

        glXSwapBuffers(display_, window_);
        return true;
    }

  private:
    using GlXSwapIntervalExtFn = void (*)(Display*, GLXDrawable, int);
    using GlXSwapIntervalMesaFn = int (*)(unsigned int);
    using GlXSwapIntervalSgiFn = int (*)(int);

    void open_display() {
        display_ = XOpenDisplay(nullptr);
        if (!display_) throw std::runtime_error("failed to open X display; set DISPLAY=:0 or use --display-backend ffplay");
        screen_ = DefaultScreen(display_);
        window_width_ = DisplayWidth(display_, screen_);
        window_height_ = DisplayHeight(display_, screen_);
    }

    void create_window() {
        int attrs[] = {
            GLX_RGBA,
            GLX_DOUBLEBUFFER,
            GLX_RED_SIZE, 8,
            GLX_GREEN_SIZE, 8,
            GLX_BLUE_SIZE, 8,
            GLX_DEPTH_SIZE, 0,
            None,
        };
        visual_ = glXChooseVisual(display_, screen_, attrs);
        if (!visual_) throw std::runtime_error("failed to choose GLX visual");

        colormap_ = XCreateColormap(display_, RootWindow(display_, screen_), visual_->visual, AllocNone);
        XSetWindowAttributes swa{};
        swa.colormap = colormap_;
        swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
        window_ = XCreateWindow(
            display_,
            RootWindow(display_, screen_),
            0,
            0,
            static_cast<unsigned int>(window_width_),
            static_cast<unsigned int>(window_height_),
            0,
            visual_->depth,
            InputOutput,
            visual_->visual,
            CWColormap | CWEventMask,
            &swa);
        if (!window_) throw std::runtime_error("failed to create X11 window");

        wm_delete_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &wm_delete_, 1);
        Atom wm_state = XInternAtom(display_, "_NET_WM_STATE", False);
        Atom fullscreen = XInternAtom(display_, "_NET_WM_STATE_FULLSCREEN", False);
        XChangeProperty(display_, window_, wm_state, XA_ATOM, 32, PropModeReplace, reinterpret_cast<unsigned char*>(&fullscreen), 1);
        XStoreName(display_, window_, "Transformirror Fast");
        XMapRaised(display_, window_);

        context_ = glXCreateContext(display_, visual_, nullptr, GL_TRUE);
        if (!context_) throw std::runtime_error("failed to create GLX context");
        if (!glXMakeCurrent(display_, window_, context_)) throw std::runtime_error("failed to make GLX context current");
        XFlush(display_);
    }

    void create_gl_objects() {
        glewExperimental = GL_TRUE;
        GLenum glew_status = glewInit();
        if (glew_status != GLEW_OK) {
            throw std::runtime_error(std::string("GLEW init failed: ") + reinterpret_cast<const char*>(glewGetErrorString(glew_status)));
        }
        disable_vsync();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, kWidth, kHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, rgb_bytes_, nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) throw std::runtime_error("OpenGL setup failed with error " + std::to_string(err));
    }

    void disable_vsync() {
        auto* ext = reinterpret_cast<GlXSwapIntervalExtFn>(glXGetProcAddressARB(reinterpret_cast<const GLubyte*>("glXSwapIntervalEXT")));
        if (ext) {
            ext(display_, window_, 0);
            return;
        }
        auto* mesa = reinterpret_cast<GlXSwapIntervalMesaFn>(glXGetProcAddressARB(reinterpret_cast<const GLubyte*>("glXSwapIntervalMESA")));
        if (mesa) {
            mesa(0);
            return;
        }
        auto* sgi = reinterpret_cast<GlXSwapIntervalSgiFn>(glXGetProcAddressARB(reinterpret_cast<const GLubyte*>("glXSwapIntervalSGI")));
        if (sgi) sgi(0);
    }

    void register_cuda_pbo() {
        CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cuda_resource_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard));
    }

    void pump_events() {
        while (XPending(display_) > 0) {
            XEvent event{};
            XNextEvent(display_, &event);
            if (event.type == ConfigureNotify) {
                window_width_ = std::max(1, event.xconfigure.width);
                window_height_ = std::max(1, event.xconfigure.height);
            } else if (event.type == ClientMessage &&
                       static_cast<Atom>(event.xclient.data.l[0]) == wm_delete_) {
                g_running.store(false);
            } else if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_Escape || key == XK_q) g_running.store(false);
            }
        }
    }

    size_t rgb_bytes_ = 0;
    Display* display_ = nullptr;
    int screen_ = 0;
    int window_width_ = kWidth;
    int window_height_ = kHeight;
    XVisualInfo* visual_ = nullptr;
    Colormap colormap_ = 0;
    Window window_ = 0;
    Atom wm_delete_ = 0;
    GLXContext context_ = nullptr;
    GLuint texture_ = 0;
    GLuint pbo_ = 0;
    cudaGraphicsResource* cuda_resource_ = nullptr;
};

bool read_exact(FILE* file, uint8_t* dst, size_t bytes) {
    size_t got = 0;
    while (got < bytes && g_running.load()) {
        size_t n = fread(dst + got, 1, bytes - got, file);
        if (n == 0) return false;
        got += n;
    }
    return got == bytes;
}

bool write_exact(FILE* file, const uint8_t* src, size_t bytes) {
    size_t sent = 0;
    while (sent < bytes && g_running.load()) {
        size_t n = fwrite(src + sent, 1, bytes - sent, file);
        if (n == 0) return false;
        sent += n;
    }
    fflush(file);
    return sent == bytes;
}

bool use_latest_frame_mode();
void update_capture_stats(int queued_frames, uint64_t dropped_frames);

class CaptureReader {
  public:
    CaptureReader(FILE* camera, size_t frame_bytes, int core, int rt_priority)
        : camera_(camera), frame_bytes_(frame_bytes), core_(core), rt_priority_(rt_priority),
          thread_(&CaptureReader::loop, this) {}

    ~CaptureReader() { stop(); }

    CaptureReader(const CaptureReader&) = delete;
    CaptureReader& operator=(const CaptureReader&) = delete;

    bool read_frame(uint8_t* dst) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || !running_.load() || !g_running.load(); });
        if (queue_.empty()) return false;

        if (use_latest_frame_mode()) {
            if (queue_.size() > 1) {
                dropped_frames_ += queue_.size() - 1;
                while (queue_.size() > 1) queue_.pop_front();
            }
            std::memcpy(dst, queue_.back().data(), frame_bytes_);
            queue_.clear();
        } else {
            std::memcpy(dst, queue_.front().data(), frame_bytes_);
            queue_.pop_front();
        }
        update_capture_stats(static_cast<int>(queue_.size()), dropped_frames_);
        return true;
    }

    void stop() {
        running_.store(false);
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

  private:
    void loop() {
        configure_current_thread("cap-ffmpeg", core_, rt_priority_);
        while (running_.load() && g_running.load()) {
            std::vector<uint8_t> frame(frame_bytes_);
            if (!read_exact(camera_, frame.data(), frame_bytes_)) break;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (use_latest_frame_mode()) {
                    dropped_frames_ += queue_.size();
                    queue_.clear();
                }
                queue_.push_back(std::move(frame));
                update_capture_stats(static_cast<int>(queue_.size()), dropped_frames_);
            }
            cv_.notify_one();
        }
        running_.store(false);
        cv_.notify_all();
    }

    FILE* camera_ = nullptr;
    size_t frame_bytes_ = 0;
    int core_ = -1;
    int rt_priority_ = 0;
    std::atomic<bool> running_{true};
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::vector<uint8_t>> queue_;
    uint64_t dropped_frames_ = 0;
};

int xioctl(int fd, unsigned long request, void* arg) {
    int rc;
    do {
        rc = ioctl(fd, request, arg);
    } while (rc == -1 && errno == EINTR);
    return rc;
}

struct JpegErrorManager {
    jpeg_error_mgr pub;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

void jpeg_error_exit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->message);
    longjmp(err->jump, 1);
}

bool decode_mjpeg_to_rgb(const uint8_t* data, size_t size, std::vector<uint8_t>& rgb, int& width, int& height) {
    jpeg_decompress_struct cinfo{};
    JpegErrorManager jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        std::cerr << "JPEG decode failed: " << jerr.message << "\n";
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<unsigned char*>(data), size);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    width = static_cast<int>(cinfo.output_width);
    height = static_cast<int>(cinfo.output_height);
    int row_stride = width * static_cast<int>(cinfo.output_components);
    rgb.resize(static_cast<size_t>(row_stride) * height);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = rgb.data() + static_cast<size_t>(cinfo.output_scanline) * row_stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}

void crop_resize_rgb_bilinear(const uint8_t* src, int src_w, int src_h, uint8_t* dst) {
    const double target_aspect = static_cast<double>(kWidth) / static_cast<double>(kHeight);
    const double source_aspect = static_cast<double>(src_w) / static_cast<double>(src_h);
    int crop_w = src_w;
    int crop_h = src_h;
    if (source_aspect > target_aspect) {
        crop_w = std::max(1, static_cast<int>(std::lround(src_h * target_aspect)));
    } else if (source_aspect < target_aspect) {
        crop_h = std::max(1, static_cast<int>(std::lround(src_w / target_aspect)));
    }
    crop_w = std::min(crop_w, src_w);
    crop_h = std::min(crop_h, src_h);
    const int x0 = (src_w - crop_w) / 2;
    const int y0 = (src_h - crop_h) / 2;
    const float scale_x = static_cast<float>(crop_w) / static_cast<float>(kWidth);
    const float scale_y = static_cast<float>(crop_h) / static_cast<float>(kHeight);
    for (int y = 0; y < kHeight; ++y) {
        float fy = y0 + (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y1 = std::clamp(static_cast<int>(std::floor(fy)), 0, src_h - 1);
        int y2 = std::min(y1 + 1, src_h - 1);
        float wy = fy - static_cast<float>(y1);
        for (int x = 0; x < kWidth; ++x) {
            float fx = x0 + (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x1 = std::clamp(static_cast<int>(std::floor(fx)), 0, src_w - 1);
            int x2 = std::min(x1 + 1, src_w - 1);
            float wx = fx - static_cast<float>(x1);
            const uint8_t* p11 = src + (static_cast<size_t>(y1) * src_w + x1) * 3;
            const uint8_t* p12 = src + (static_cast<size_t>(y1) * src_w + x2) * 3;
            const uint8_t* p21 = src + (static_cast<size_t>(y2) * src_w + x1) * 3;
            const uint8_t* p22 = src + (static_cast<size_t>(y2) * src_w + x2) * 3;
            uint8_t* out = dst + (static_cast<size_t>(y) * kWidth + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float top = p11[c] + (p12[c] - p11[c]) * wx;
                float bottom = p21[c] + (p22[c] - p21[c]) * wx;
                out[c] = static_cast<uint8_t>(std::clamp(top + (bottom - top) * wy, 0.0f, 255.0f));
            }
        }
    }
}

class V4L2CaptureReader {
  public:
    V4L2CaptureReader(const Args& args, size_t frame_bytes)
        : args_(args), frame_bytes_(frame_bytes) {
        setup_device();
        thread_ = std::thread(&V4L2CaptureReader::loop, this);
    }

    ~V4L2CaptureReader() {
        stop();
        close_device();
    }

    V4L2CaptureReader(const V4L2CaptureReader&) = delete;
    V4L2CaptureReader& operator=(const V4L2CaptureReader&) = delete;

    int source_width() const { return actual_width_; }
    int source_height() const { return actual_height_; }

    bool read_frame(uint8_t* dst) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || !running_.load() || !g_running.load(); });
        if (queue_.empty()) return false;
        if (use_latest_frame_mode()) {
            if (queue_.size() > 1) {
                dropped_frames_ += queue_.size() - 1;
                while (queue_.size() > 1) queue_.pop_front();
            }
            std::memcpy(dst, queue_.back().data(), frame_bytes_);
            queue_.clear();
        } else {
            std::memcpy(dst, queue_.front().data(), frame_bytes_);
            queue_.pop_front();
        }
        update_capture_stats(static_cast<int>(queue_.size()), dropped_frames_);
        return true;
    }

    void stop() {
        running_.store(false);
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

  private:
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };

    void setup_device() {
        fd_ = open(args_.camera_device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) throw std::runtime_error("failed to open " + args_.camera_device + ": " + std::strerror(errno));

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = args_.capture_width;
        fmt.fmt.pix.height = args_.capture_height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) throw std::runtime_error("VIDIOC_S_FMT failed: " + std::string(std::strerror(errno)));
        if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) throw std::runtime_error("V4L2 device did not accept MJPEG format");
        actual_width_ = static_cast<int>(fmt.fmt.pix.width);
        actual_height_ = static_cast<int>(fmt.fmt.pix.height);

        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = args_.camera_fps;
        if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) warn_errno("VIDIOC_S_PARM");

        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) throw std::runtime_error("VIDIOC_REQBUFS failed: " + std::string(std::strerror(errno)));
        if (req.count < 2) throw std::runtime_error("V4L2 returned too few mmap buffers");

        buffers_.resize(req.count);
        for (size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<uint32_t>(i);
            if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) throw std::runtime_error("VIDIOC_QUERYBUF failed: " + std::string(std::strerror(errno)));
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED) throw std::runtime_error("mmap V4L2 buffer failed: " + std::string(std::strerror(errno)));
            if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) throw std::runtime_error("VIDIOC_QBUF failed: " + std::string(std::strerror(errno)));
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) throw std::runtime_error("VIDIOC_STREAMON failed: " + std::string(std::strerror(errno)));
        streaming_ = true;
        std::cerr << "camera: v4l2 mmap MJPEG " << actual_width_ << "x" << actual_height_
                  << " @" << args_.camera_fps << " from " << args_.camera_device << "\n";
    }

    void close_device() {
        if (streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (auto& buffer : buffers_) {
            if (buffer.start && buffer.start != MAP_FAILED) munmap(buffer.start, buffer.length);
        }
        buffers_.clear();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void loop() {
        configure_current_thread("cap-v4l2", args_.capture_core, args_.rt_priority);
        while (running_.load() && g_running.load()) {
            pollfd pfd{};
            pfd.fd = fd_;
            pfd.events = POLLIN;
            int prc = poll(&pfd, 1, 250);
            if (prc < 0) {
                if (errno == EINTR) continue;
                warn_errno("poll V4L2");
                break;
            }
            if (prc == 0) continue;

            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
                if (errno == EAGAIN) continue;
                warn_errno("VIDIOC_DQBUF");
                break;
            }

            int jpeg_w = 0;
            int jpeg_h = 0;
            bool ok = decode_mjpeg_to_rgb(
                static_cast<const uint8_t*>(buffers_[buf.index].start),
                buf.bytesused,
                decoded_rgb_,
                jpeg_w,
                jpeg_h);
            if (ok) {
                std::vector<uint8_t> frame(frame_bytes_);
                crop_resize_rgb_bilinear(decoded_rgb_.data(), jpeg_w, jpeg_h, frame.data());
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (use_latest_frame_mode()) {
                        dropped_frames_ += queue_.size();
                        queue_.clear();
                    }
                    queue_.push_back(std::move(frame));
                    update_capture_stats(static_cast<int>(queue_.size()), dropped_frames_);
                }
                cv_.notify_one();
            }

            if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                warn_errno("VIDIOC_QBUF");
                break;
            }
        }
        running_.store(false);
        cv_.notify_all();
    }

    const Args& args_;
    size_t frame_bytes_ = 0;
    int fd_ = -1;
    int actual_width_ = 0;
    int actual_height_ = 0;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
    std::vector<uint8_t> decoded_rgb_;
    std::atomic<bool> running_{true};
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::vector<uint8_t>> queue_;
    uint64_t dropped_frames_ = 0;
};

std::string html_page() {
    return R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transformirror Fast</title>
<style>
body{font:14px system-ui;margin:24px;background:#111;color:#eee;max-width:760px}
label{display:block;margin:14px 0 6px;color:#aaa}.row{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center}
input,textarea,button{font:inherit;background:#1f1f1f;color:#fff;border:1px solid #444;border-radius:6px;padding:8px}
input[type=range]{width:100%}textarea{width:100%;height:88px}button{cursor:pointer;background:#2e5bff;border:0}
.status{padding:10px;background:#1b1b1b;border-radius:6px;margin-bottom:18px}
</style></head><body>
<h1>Transformirror Fast</h1><div id="status" class="status">loading</div>
<label>Prompt</label><textarea id="prompt"></textarea>
<label>Seed</label><input id="seed" type="number">
<label>Strength</label><div class="row"><input id="strength" type="range" min="0" max="1" step="0.01"><span id="strengthv"></span></div>
<label>Steps</label><input id="steps" type="number" min="1" max="8">
<label>Blend</label><div class="row"><input id="blend" type="range" min="0" max="1" step="0.01"><span id="blendv"></span></div>
<label><input id="passthrough" type="checkbox"> Passthrough</label>
<label>Frame handling</label><select id="frame_mode"><option value="latest">Always use newest frame</option><option value="fifo">Do not drop frames</option></select>
<label>Resolution</label><input id="resolution" value="1024x1024">
<p><button id="apply">Apply</button></p>
<script>
const ui={};
for (const id of ['status','prompt','seed','strength','strengthv','steps','blend','blendv','passthrough','frame_mode','resolution','apply']) ui[id]=document.getElementById(id);
async function getState(){const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);return await r.json()}
function statusLine(s){return `${s.status_text || s.status?.message || 'running'} | ${s.fps.toFixed(1)} fps | model ${s.frame_ms.toFixed(2)} ms | capture ${s.capture_ms.toFixed(2)} ms | display ${s.display_ms.toFixed(2)} ms | loop ${s.loop_ms.toFixed(2)} ms | queued ${s.queued_frames} | dropped ${s.dropped_frames}`}
function fill(s){ui.status.textContent=statusLine(s);
ui.prompt.value=s.prompt;ui.seed.value=s.seed;ui.strength.value=s.strength;ui.strengthv.textContent=s.strength.toFixed(2);
ui.steps.value=s.steps;ui.blend.value=s.blend;ui.blendv.textContent=s.blend.toFixed(2);ui.passthrough.checked=s.passthrough;
ui.frame_mode.value=s.use_latest_frame?'latest':'fifo';
ui.resolution.value=`${s.width}x${s.height}`}
async function refresh(){try{ui.status.textContent=statusLine(await getState())}catch(e){ui.status.textContent=`error: ${e.message}`}}
for (const id of ['strength','blend']) ui[id].oninput=()=>ui[id+'v'].textContent=(+ui[id].value).toFixed(2);
ui.apply.onclick=async()=>{let [w,h]=ui.resolution.value.split('x').map(Number);await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:ui.prompt.value,seed:+ui.seed.value,strength:+ui.strength.value,steps:+ui.steps.value,blend:+ui.blend.value,passthrough:ui.passthrough.checked,use_latest_frame:ui.frame_mode.value==='latest',width:w||1024,height:h||1024})});fill(await getState())}
setInterval(refresh,1000);
getState().then(fill).catch(e=>{ui.status.textContent=`error: ${e.message}`});
</script></body></html>)HTML";
}

std::string frontend_page(const std::string& web_root) {
    std::string disk = read_text_file_or_empty(join_path(web_root, "index.html"));
    if (!disk.empty()) return disk;
    return html_page();
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

std::string state_json() {
    const std::string cached_resolutions = cached_resolutions_json();
    std::lock_guard<std::mutex> lock(g_state_mutex);
    const std::string status_text = g_state.status;
    const bool has_error =
        status_text.find("failed") != std::string::npos ||
        status_text.find("error") != std::string::npos ||
        status_text.find("unsupported") != std::string::npos ||
        status_text.find("fixed") != std::string::npos ||
        status_text.find("expects") != std::string::npos;
    const bool model_ready =
        status_text.find("starting") == std::string::npos &&
        status_text.find("regenerating") == std::string::npos;
    const double diffusion_fps = g_state.frame_ms > 0.0 ? 1000.0 / g_state.frame_ms : g_state.fps;
    const double display_fps = g_state.fps;
    const double camera_fps = g_state.camera_fps > 0.0 ? g_state.camera_fps : g_state.fps;
    std::ostringstream out;
    out << "{"
        << "\"prompt\":\"" << json_escape(g_state.prompt) << "\","
        << "\"seed\":" << g_state.seed << ","
        << "\"strength\":" << g_state.strength << ","
        << "\"steps\":" << g_state.steps << ","
        << "\"blend\":" << g_state.blend << ","
        << "\"passthrough\":" << (g_state.passthrough ? "true" : "false") << ","
        << "\"use_latest_frame\":" << (g_state.use_latest_frame ? "true" : "false") << ","
        << "\"width\":" << g_state.width << ","
        << "\"height\":" << g_state.height << ","
        << "\"fps\":" << g_state.fps << ","
        << "\"frame_ms\":" << g_state.frame_ms << ","
        << "\"capture_ms\":" << g_state.capture_ms << ","
        << "\"display_ms\":" << g_state.display_ms << ","
        << "\"loop_ms\":" << g_state.loop_ms << ","
        << "\"conditioning_ms\":" << g_state.conditioning_ms << ","
        << "\"queued_frames\":" << g_state.queued_frames << ","
        << "\"dropped_frames\":" << g_state.dropped_frames << ","
        << "\"status_text\":\"" << json_escape(status_text) << "\","
        << "\"controls\":{"
        << "\"prompt\":\"" << json_escape(g_state.prompt) << "\","
        << "\"seed\":" << g_state.seed << ","
        << "\"strength\":" << g_state.strength << ","
        << "\"steps\":" << g_state.steps << ","
        << "\"blend\":" << g_state.blend
        << "},"
        << "\"config\":{"
        << "\"width\":" << g_state.width << ","
        << "\"height\":" << g_state.height << ","
        << "\"requested_width\":" << g_state.requested_width << ","
        << "\"requested_height\":" << g_state.requested_height << ","
        << "\"cached_resolutions\":" << cached_resolutions << ","
        << "\"http_port\":" << g_state.http_port << ","
        << "\"osc_port\":" << g_state.osc_port
        << "},"
        << "\"stats\":{"
        << "\"camera_source_width\":" << g_state.camera_source_width << ","
        << "\"camera_source_height\":" << g_state.camera_source_height << ","
        << "\"camera_crop_width\":" << g_state.camera_crop_width << ","
        << "\"camera_crop_height\":" << g_state.camera_crop_height << ","
        << "\"camera_fps\":" << camera_fps << ","
        << "\"display_width\":" << g_state.display_width << ","
        << "\"display_height\":" << g_state.display_height << ","
        << "\"display_fps\":" << display_fps << ","
        << "\"diffusion_ms\":" << g_state.frame_ms << ","
        << "\"diffusion_fps\":" << diffusion_fps << ","
        << "\"capture_ms\":" << g_state.capture_ms << ","
        << "\"display_ms\":" << g_state.display_ms << ","
        << "\"loop_ms\":" << g_state.loop_ms << ","
        << "\"conditioning_ms\":" << g_state.conditioning_ms << ","
        << "\"queued_frames\":" << g_state.queued_frames << ","
        << "\"dropped_frames\":" << g_state.dropped_frames
        << "},"
        << "\"status\":{"
        << "\"camera_ready\":" << (g_state.fps > 0.0 ? "true" : "false") << ","
        << "\"model_ready\":" << (model_ready ? "true" : "false") << ","
        << "\"resolution_changing\":" << (g_state.resolution_rebuild_active ? "true" : "false") << ","
        << "\"last_error\":\"" << (has_error ? json_escape(status_text) : "") << "\","
        << "\"message\":\"" << json_escape(status_text) << "\""
        << "}"
        << "}";
    return out.str();
}

bool find_string(const std::string& body, const std::string& key, std::string& value) {
    std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    p = body.find('"', p);
    if (p == std::string::npos) return false;
    std::string out;
    bool escaped = false;
    for (size_t i = p + 1; i < body.size(); ++i) {
        char c = body[i];
        if (escaped) {
            switch (c) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += c; break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            value = out;
            return true;
        } else {
            out += c;
        }
    }
    return false;
}

bool find_number(const std::string& body, const std::string& key, double& value) {
    std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
    size_t e = p;
    while (e < body.size() && (std::isdigit(static_cast<unsigned char>(body[e])) || body[e] == '.' || body[e] == '-')) ++e;
    if (e == p) return false;
    value = std::stod(body.substr(p, e - p));
    return true;
}

bool find_bool(const std::string& body, const std::string& key, bool& value) {
    std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
    if (body.compare(p, 4, "true") == 0) { value = true; return true; }
    if (body.compare(p, 5, "false") == 0) { value = false; return true; }
    return false;
}

void apply_json_state(const std::string& body) {
    std::string s;
    double n;
    bool b;
    bool reload = false;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        if (find_string(body, "prompt", s) && s != g_state.prompt) { g_state.prompt = s; reload = true; }
        if (find_number(body, "seed", n) && static_cast<int>(n) != g_state.seed) { g_state.seed = static_cast<int>(n); reload = true; }
        if (find_number(body, "strength", n) && std::fabs(static_cast<float>(n) - g_state.strength) > 1e-5f) { g_state.strength = std::clamp(static_cast<float>(n), 0.0f, 1.0f); reload = true; }
        if (find_number(body, "steps", n) && static_cast<int>(n) != g_state.steps) {
            g_state.steps = std::clamp(static_cast<int>(n), 2, 8);
            reload = true;
        }
        if (find_number(body, "blend", n)) g_state.blend = std::clamp(static_cast<float>(n), 0.0f, 1.0f);
        if (find_bool(body, "passthrough", b)) g_state.passthrough = b;
        if (find_bool(body, "use_latest_frame", b)) g_state.use_latest_frame = b;
        if (find_string(body, "frame_mode", s)) g_state.use_latest_frame = (s != "fifo" && s != "no_drop");
        bool has_width = find_number(body, "width", n);
        int requested_width = has_width ? normalized_resolution(static_cast<int>(n)) : g_state.width;
        bool has_height = find_number(body, "height", n);
        int requested_height = has_height ? normalized_resolution(static_cast<int>(n)) : g_state.height;
        if ((has_width || has_height) &&
            (requested_width != g_state.width || requested_height != g_state.height)) {
            if (requested_width == kWidth && requested_height == kHeight) {
                g_state.width = kWidth;
                g_state.height = kHeight;
                g_state.status = "already running requested resolution";
            } else {
                g_state.requested_width = requested_width;
                g_state.requested_height = requested_height;
                g_state.resolution_rebuild_requested = true;
                g_state.status = "queued resolution build for " +
                    std::to_string(requested_width) + "x" + std::to_string(requested_height);
            }
        }
        if (reload) {
            g_state.reload_requested = true;
            g_state.status = "conditioning reload queued";
        }
    }
    if (reload) g_reload_cv.notify_one();
    g_resolution_cv.notify_one();
}

bool use_latest_frame_mode() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_state.use_latest_frame;
}

void update_capture_stats(int queued_frames, uint64_t dropped_frames) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_state.queued_frames = queued_frames;
    g_state.dropped_frames = dropped_frames;
}

void send_http(int client, const std::string& code, const std::string& type, const std::string& body) {
    std::ostringstream res;
    res << "HTTP/1.1 " << code << "\r\nContent-Type: " << type
        << "\r\nContent-Length: " << body.size()
        << "\r\nCache-Control: no-store"
        << "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n" << body;
    std::string text = res.str();
    send(client, text.data(), text.size(), 0);
}

std::string local_mdns_name() {
    char host[256] = {};
    if (gethostname(host, sizeof(host) - 1) != 0 || host[0] == '\0') return "localhost";
    return std::string(host) + ".local";
}

int bind_tcp_any(int port) {
    int server = socket(AF_INET6, SOCK_STREAM, 0);
    if (server >= 0) {
        set_close_on_exec(server);
        int opt = 1;
        int dual_stack = 0;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        setsockopt(server, IPPROTO_IPV6, IPV6_V6ONLY, &dual_stack, sizeof(dual_stack));
        sockaddr_in6 addr{};
        addr.sin6_family = AF_INET6;
        addr.sin6_addr = in6addr_any;
        addr.sin6_port = htons(port);
        if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
            listen(server, 16) == 0) {
            return server;
        }
        close(server);
    }

    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return -1;
    set_close_on_exec(server);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(server, 16) != 0) {
        close(server);
        return -1;
    }
    return server;
}

int bind_udp_any(int port) {
    int sock = socket(AF_INET6, SOCK_DGRAM, 0);
    if (sock >= 0) {
        set_close_on_exec(sock);
        int opt = 1;
        int dual_stack = 0;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, &dual_stack, sizeof(dual_stack));
        sockaddr_in6 addr{};
        addr.sin6_family = AF_INET6;
        addr.sin6_addr = in6addr_any;
        addr.sin6_port = htons(port);
        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return sock;
        close(sock);
    }

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return -1;
    set_close_on_exec(sock);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(sock);
        return -1;
    }
    return sock;
}

size_t http_content_length(const std::string& req) {
    size_t p = req.find("Content-Length:");
    if (p == std::string::npos) p = req.find("content-length:");
    if (p == std::string::npos) return 0;
    p = req.find(':', p);
    if (p == std::string::npos) return 0;
    ++p;
    while (p < req.size() && std::isspace(static_cast<unsigned char>(req[p]))) ++p;
    size_t e = p;
    while (e < req.size() && std::isdigit(static_cast<unsigned char>(req[e]))) ++e;
    if (e == p) return 0;
    return static_cast<size_t>(std::stoul(req.substr(p, e - p)));
}

void http_thread(int port, int core, int rt_priority, std::string web_root) {
    configure_current_thread("http", core, rt_priority > 0 ? 1 : 0);
    int server = bind_tcp_any(port);
    if (server < 0) {
        std::cerr << "HTTP bind/listen failed on port " << port << "\n";
        return;
    }
    while (g_running.load()) {
        int client = accept(server, nullptr, nullptr);
        if (client < 0) continue;
        char buf[8192];
        ssize_t n = recv(client, buf, sizeof(buf) - 1, 0);
        if (n <= 0) { close(client); continue; }
        buf[n] = 0;
        std::string req(buf);
        bool post = req.rfind("POST ", 0) == 0;
        bool get = req.rfind("GET ", 0) == 0;
        std::string body;
        size_t body_pos = req.find("\r\n\r\n");
        if (body_pos != std::string::npos) {
            size_t header_bytes = body_pos + 4;
            size_t content_length = http_content_length(req);
            while (content_length > 0 && req.size() < header_bytes + content_length) {
                n = recv(client, buf, sizeof(buf) - 1, 0);
                if (n <= 0) break;
                req.append(buf, static_cast<size_t>(n));
            }
            body = req.substr(header_bytes, content_length > 0 ? content_length : std::string::npos);
        }
        if (get && req.find("GET /api/state ") == 0) send_http(client, "200 OK", "application/json", state_json());
        else if (post && req.find("POST /api/state ") == 0) { apply_json_state(body); send_http(client, "200 OK", "application/json", state_json()); }
        else if (get && (req.find("GET / ") == 0 || req.find("GET /index.html ") == 0)) {
            send_http(client, "200 OK", "text/html; charset=utf-8", frontend_page(web_root));
        }
        else send_http(client, "404 Not Found", "text/plain", "not found\n");
        close(client);
    }
    close(server);
}

uint32_t read_be32(const char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return ntohl(v);
}

float read_befloat(const char* p) {
    uint32_t v = read_be32(p);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

size_t osc_padded(size_t n) { return (n + 4) & ~size_t(3); }

void handle_osc(const char* data, size_t size) {
    if (size < 8 || data[0] != '/') return;
    std::string address(data);
    size_t pos = osc_padded(address.size() + 1);
    if (pos >= size || data[pos] != ',') return;
    std::string types(data + pos);
    pos += osc_padded(types.size() + 1);
    std::ostringstream json;
    json << "{";
    bool wrote = false;
    auto add_comma = [&] { if (wrote) json << ","; wrote = true; };
    auto name = address;
    const std::string prefix = "/transformirror";
    if (name.rfind(prefix, 0) == 0) name = name.substr(prefix.size());
    if (name == "/prompt" && types.find('s') != std::string::npos && pos < size) {
        add_comma(); json << "\"prompt\":\"" << json_escape(std::string(data + pos)) << "\"";
    } else if (name == "/frame_mode" && types.find('s') != std::string::npos && pos < size) {
        add_comma(); json << "\"frame_mode\":\"" << json_escape(std::string(data + pos)) << "\"";
    } else if ((name == "/seed" || name == "/steps" || name == "/width" || name == "/height") && types.find('i') != std::string::npos && pos + 4 <= size) {
        int value = static_cast<int>(read_be32(data + pos));
        add_comma();
        if (name == "/width") json << "\"width\":" << value;
        else if (name == "/height") json << "\"height\":" << value;
        else json << "\"" << name.substr(1) << "\":" << value;
    } else if ((name == "/strength" || name == "/blend") && types.find('f') != std::string::npos && pos + 4 <= size) {
        add_comma(); json << "\"" << name.substr(1) << "\":" << read_befloat(data + pos);
    } else if (name == "/passthrough" && pos + 4 <= size) {
        add_comma(); json << "\"passthrough\":" << (read_be32(data + pos) ? "true" : "false");
    } else if (name == "/use_latest_frame" && pos + 4 <= size) {
        add_comma(); json << "\"use_latest_frame\":" << (read_be32(data + pos) ? "true" : "false");
    }
    json << "}";
    if (wrote) apply_json_state(json.str());
}

void osc_thread(int port, int core, int rt_priority) {
    configure_current_thread("osc", core, rt_priority > 0 ? 2 : 0);
    int sock = bind_udp_any(port);
    if (sock < 0) {
        std::cerr << "OSC bind failed on port " << port << "\n";
        return;
    }
    char buf[4096];
    while (g_running.load()) {
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n > 0) handle_osc(buf, static_cast<size_t>(n));
    }
    close(sock);
}

bool send_all_fd(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string recv_line_fd(int fd) {
    std::string out;
    char buf[4096];
    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
        size_t newline = out.find('\n');
        if (newline != std::string::npos) {
            out.resize(newline);
            break;
        }
    }
    return out;
}

int connect_unix_socket(const std::string& path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        close(fd);
        errno = ENAMETOOLONG;
        return -1;
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

std::string conditioning_request_json(const AppState& state) {
    std::ostringstream req;
    req << "{"
        << "\"prompt\":\"" << json_escape(state.prompt) << "\","
        << "\"seed\":" << state.seed << ","
        << "\"strength\":" << state.strength << ","
        << "\"steps\":" << state.steps << ","
        << "\"width\":" << state.width << ","
        << "\"height\":" << state.height
        << "}\n";
    return req.str();
}

class ConditioningWorkerClient {
  public:
    explicit ConditioningWorkerClient(const Args& args) : args_(args) {}

    void start_async(bool force = false) {
        if (started_ && !force) return;
        started_ = true;
        unlink(args_.conditioning_socket.c_str());
        std::ostringstream cmd;
        cmd << "cd " << shell_quote(current_working_directory()) << " && "
            << shell_quote(args_.python) << " " << shell_quote(args_.conditioning_script)
            << " --socket " << shell_quote(args_.conditioning_socket)
            << " --out-dir " << shell_quote(args_.asset_dir)
            << " --width " << kWidth
            << " --height " << kHeight
            << " --parent-pid " << getpid()
            << " >/tmp/transformirror_conditioning_worker.log 2>&1 &";
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            throw std::runtime_error("failed to launch conditioning worker");
        }
    }

    bool wait_ready(double timeout_sec) {
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
        while (g_running.load() && std::chrono::steady_clock::now() < deadline) {
            if (access(args_.conditioning_socket.c_str(), F_OK) == 0) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    }

    bool request(const AppState& state, std::string& response, std::string& error) {
        start_async();
        int fd = connect_unix_socket(args_.conditioning_socket);
        if (fd < 0) {
            if (!wait_ready(2.0)) start_async(true);
            if (!wait_ready(20.0)) {
                error = "conditioning worker did not become ready; see /tmp/transformirror_conditioning_worker.log";
                return false;
            }
            fd = connect_unix_socket(args_.conditioning_socket);
            if (fd < 0) {
                start_async(true);
                if (!wait_ready(20.0)) {
                    error = "conditioning worker did not become ready after restart; see /tmp/transformirror_conditioning_worker.log";
                    return false;
                }
                fd = connect_unix_socket(args_.conditioning_socket);
                if (fd < 0) {
                    error = std::string("connect conditioning worker failed: ") + std::strerror(errno);
                    return false;
                }
            }
        }

        bool ok = send_all_fd(fd, conditioning_request_json(state));
        if (ok) response = recv_line_fd(fd);
        close(fd);
        if (!ok || response.empty()) {
            error = "conditioning worker returned an empty response";
            return false;
        }
        bool worker_ok = false;
        if (!find_bool(response, "ok", worker_ok) || !worker_ok) {
            std::string worker_error;
            find_string(response, "error", worker_error);
            error = worker_error.empty() ? response : worker_error;
            return false;
        }
        return true;
    }

  private:
    const Args& args_;
    bool started_ = false;
};

bool run_script_conditioning(const Args& args, const AppState& snapshot) {
    std::ostringstream cmd;
    cmd << shell_quote(args.python) << " export_cpp_assets.py"
        << " --out-dir " << shell_quote(args.asset_dir)
        << " --width " << kWidth
        << " --height " << kHeight
        << " --prompt " << shell_quote(snapshot.prompt)
        << " --seed " << snapshot.seed
        << " --strength " << snapshot.strength
        << " --steps " << snapshot.steps
        << " >/tmp/transformirror_fast_assets.log 2>&1";
    return std::system(cmd.str().c_str()) == 0;
}

void resolution_rebuild_thread(const Args& args) {
    configure_current_thread("resolution", args.reload_core, 0);
    while (g_running.load()) {
        int width = kWidth;
        int height = kHeight;
        AppState snapshot;
        {
            std::unique_lock<std::mutex> lock(g_state_mutex);
            g_resolution_cv.wait_for(lock, std::chrono::milliseconds(250), [] {
                return !g_running.load() || g_state.resolution_rebuild_requested;
            });
            if (!g_running.load()) break;
            if (!g_state.resolution_rebuild_requested) continue;
            g_state.resolution_rebuild_requested = false;
            g_state.resolution_rebuild_active = true;
            width = g_state.requested_width;
            height = g_state.requested_height;
            snapshot = g_state;
            g_state.status = "building TensorRT app for " + std::to_string(width) + "x" + std::to_string(height);
        }

        std::string root = current_working_directory();
        std::string key = std::to_string(width) + "x" + std::to_string(height);
        std::string build_log = "/tmp/transformirror_resolution_build_" + key + ".log";
        std::ostringstream cmd;
        cmd << "cd " << shell_quote(root) << " && "
            << "PYTHON_BIN=" << shell_quote(args.python) << " "
            << "TRANSFORMIRROR_PROMPT=" << shell_quote(snapshot.prompt) << " "
            << "TRANSFORMIRROR_SEED=" << snapshot.seed << " "
            << "TRANSFORMIRROR_STRENGTH=" << snapshot.strength << " "
            << "TRANSFORMIRROR_STEPS=" << snapshot.steps << " "
            << shell_quote(join_path(root, "scripts/build_resolution_app.sh"))
            << " " << width << " " << height
            << " >" << shell_quote(build_log) << " 2>&1";
        int rc = std::system(cmd.str().c_str());
        if (rc == 0) {
            {
                std::lock_guard<std::mutex> lock(g_reexec_mutex);
                g_reexec_binary = join_path(join_path(root, "cpp/build_" + key), "transformirror_fast_app");
                g_reexec_engine_dir = join_path(join_path(root, "trt_engines"), key);
                g_reexec_asset_dir = join_path(g_reexec_engine_dir, "assets");
            }
            {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.status = "switching to " + key + " build";
                g_state.resolution_rebuild_active = false;
            }
            g_reexec_requested.store(true);
            g_running.store(false);
            g_reload_cv.notify_all();
            g_resolution_cv.notify_all();
            break;
        } else {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.status = "resolution build failed for " + key + "; see " + build_log;
            g_state.resolution_rebuild_active = false;
        }
    }
}

void append_arg(std::vector<std::string>& args, const std::string& key, const std::string& value) {
    args.push_back(key);
    args.push_back(value);
}

bool reexec_if_requested(const Args& args) {
    if (!g_reexec_requested.load()) return false;
    std::string binary;
    std::string engine_dir;
    std::string asset_dir;
    {
        std::lock_guard<std::mutex> lock(g_reexec_mutex);
        binary = g_reexec_binary;
        engine_dir = g_reexec_engine_dir;
        asset_dir = g_reexec_asset_dir;
    }
    if (binary.empty()) return false;

    AppState state;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        state = g_state;
    }

    std::vector<std::string> argv;
    argv.push_back(binary);
    append_arg(argv, "--engine-dir", engine_dir);
    append_arg(argv, "--asset-dir", asset_dir);
    append_arg(argv, "--web-root", args.web_root);
    append_arg(argv, "--conditioning-backend", args.conditioning_backend);
    append_arg(argv, "--conditioning-script", args.conditioning_script);
    append_arg(argv, "--conditioning-socket", args.conditioning_socket);
    append_arg(argv, "--camera-device", args.camera_device);
    append_arg(argv, "--capture-backend", args.capture_backend);
    append_arg(argv, "--display-backend", args.display_backend);
    append_arg(argv, "--python", args.python);
    append_arg(argv, "--capture-width", std::to_string(args.capture_width));
    append_arg(argv, "--capture-height", std::to_string(args.capture_height));
    append_arg(argv, "--camera-fps", std::to_string(args.camera_fps));
    append_arg(argv, "--http-port", std::to_string(args.http_port));
    append_arg(argv, "--osc-port", std::to_string(args.osc_port));
    if (args.max_frames > 0) append_arg(argv, "--max-frames", std::to_string(args.max_frames));
    if (args.main_core >= 0) append_arg(argv, "--main-core", std::to_string(args.main_core));
    if (args.capture_core >= 0) append_arg(argv, "--capture-core", std::to_string(args.capture_core));
    if (args.http_core >= 0) append_arg(argv, "--http-core", std::to_string(args.http_core));
    if (args.osc_core >= 0) append_arg(argv, "--osc-core", std::to_string(args.osc_core));
    if (args.reload_core >= 0) append_arg(argv, "--reload-core", std::to_string(args.reload_core));
    if (args.rt_priority > 0) append_arg(argv, "--rt-priority", std::to_string(args.rt_priority));
    if (args.lock_memory) argv.push_back("--lock-memory");
    append_arg(argv, "--initial-prompt", state.prompt);
    append_arg(argv, "--initial-seed", std::to_string(state.seed));
    append_arg(argv, "--initial-strength", std::to_string(state.strength));
    append_arg(argv, "--initial-steps", std::to_string(state.steps));
    append_arg(argv, "--initial-blend", std::to_string(state.blend));
    append_arg(argv, "--initial-passthrough", state.passthrough ? "1" : "0");
    append_arg(argv, "--initial-use-latest-frame", state.use_latest_frame ? "1" : "0");

    std::vector<char*> exec_argv;
    exec_argv.reserve(argv.size() + 1);
    for (std::string& value : argv) exec_argv.push_back(value.data());
    exec_argv.push_back(nullptr);

    std::cerr << "exec: " << binary << " --engine-dir " << engine_dir
              << " --asset-dir " << asset_dir << "\n";
    execv(binary.c_str(), exec_argv.data());
    warn_errno("execv");
    return true;
}

void asset_reload_thread(const Args& args, FastPipeline* pipeline) {
    configure_current_thread("asset-reload", args.reload_core, 0);
    std::unique_ptr<ConditioningWorkerClient> worker;
    if (args.conditioning_backend == "worker") {
        worker.reset(new ConditioningWorkerClient(args));
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.status = "conditioning worker warming";
        }
        try {
            worker->start_async();
            if (worker->wait_ready(30.0)) {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                if (!g_state.reload_requested) g_state.status = "conditioning worker ready";
            } else {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.status = "conditioning worker warmup timed out";
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.status = std::string("conditioning worker launch failed: ") + e.what();
        }
    }
    while (g_running.load()) {
        bool reload = false;
        AppState snapshot;
        {
            std::unique_lock<std::mutex> lock(g_state_mutex);
            g_reload_cv.wait_for(lock, std::chrono::milliseconds(250), [] {
                return !g_running.load() || g_state.reload_requested;
            });
            if (g_state.reload_requested) {
                reload = true;
                g_state.reload_requested = false;
                g_state.status = "regenerating conditioning assets";
                snapshot = g_state;
            }
        }
        if (reload) {
            auto started = std::chrono::steady_clock::now();
            bool generated = false;
            std::string response;
            std::string error;
            if (worker) {
                generated = worker->request(snapshot, response, error);
            } else {
                generated = run_script_conditioning(args, snapshot);
                if (!generated) error = "asset regeneration failed; see /tmp/transformirror_fast_assets.log";
            }
            if (generated) {
                try {
                    AssetBlob assets = read_asset_blob(args.asset_dir);
                    pipeline->upload_assets(assets);
                    double total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    double worker_ms = total_ms;
                    find_number(response, "elapsed_ms", worker_ms);
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_state.conditioning_ms = total_ms;
                    g_state.status = "conditioning assets reloaded in " +
                        std::to_string(static_cast<int>(std::round(total_ms))) +
                        " ms (worker " +
                        std::to_string(static_cast<int>(std::round(worker_ms))) + " ms)";
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_state.status = std::string("asset reload failed: ") + e.what();
                }
            } else {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.status = error.empty() ? "asset regeneration failed" : error;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        apply_initial_state(args);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        configure_current_thread("main-frame", args.main_core, args.rt_priority);
        CHECK_CUDA(cudaSetDevice(0));
        Logger logger;
        initLibNvInferPlugins(&logger, "");

        FastPipeline pipeline(args, logger);
        if (args.lock_memory) try_lock_process_memory();
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.http_port = args.http_port;
            g_state.osc_port = args.osc_port;
            g_state.camera_source_width = args.capture_width;
            g_state.camera_source_height = args.capture_height;
            g_state.camera_fps = args.camera_fps;
            g_state.display_width = kWidth;
            g_state.display_height = kHeight;
            g_state.status = "running";
        }

        std::thread http(http_thread, args.http_port, args.http_core, args.rt_priority, args.web_root);
        http.detach();
        std::thread osc(osc_thread, args.osc_port, args.osc_core, args.rt_priority);
        osc.detach();
        std::string mdns = local_mdns_name();
        std::cerr << "HTTP: http://0.0.0.0:" << args.http_port
                  << "/ http://[::]:" << args.http_port
                  << "/ http://" << mdns << ":" << args.http_port << "/\n";
        std::cerr << "OSC: udp://0.0.0.0:" << args.osc_port
                  << " udp://[::]:" << args.osc_port
                  << " udp://" << mdns << ":" << args.osc_port << "\n";
        std::thread reloader(asset_reload_thread, std::cref(args), &pipeline);
        reloader.detach();
        std::thread resolver(resolution_rebuild_thread, std::cref(args));
        resolver.detach();

        FILE* camera = nullptr;
        std::unique_ptr<CaptureReader> ffmpeg_capture;
        std::unique_ptr<V4L2CaptureReader> v4l2_capture;
        if (args.capture_backend == "v4l2") {
            v4l2_capture.reset(new V4L2CaptureReader(args, pipeline.rgb_bytes()));
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.camera_source_width = v4l2_capture->source_width();
            g_state.camera_source_height = v4l2_capture->source_height();
        } else {
            std::string cap_cmd = ffmpeg_capture_cmd(args);
            std::cerr << "camera: " << cap_cmd << "\n";
            camera = popen(cap_cmd.c_str(), "r");
            if (!camera) throw std::runtime_error("failed to start ffmpeg camera");
            ffmpeg_capture.reset(new CaptureReader(camera, pipeline.rgb_bytes(), args.capture_core, args.rt_priority));
        }
        auto read_capture = [&](uint8_t* dst) -> bool {
            if (v4l2_capture) return v4l2_capture->read_frame(dst);
            return ffmpeg_capture->read_frame(dst);
        };

        FILE* display = nullptr;
        std::unique_ptr<GlDisplay> gl_display;
        if (args.display_backend == "ffplay") {
            std::string display_cmd = ffplay_display_cmd();
            std::cerr << "display: " << display_cmd << "\n";
            display = popen(display_cmd.c_str(), "w");
            if (!display) throw std::runtime_error("failed to start ffplay display");
        } else if (args.display_backend == "gl") {
            gl_display.reset(new GlDisplay(pipeline.rgb_bytes()));
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.display_width = gl_display->width();
            g_state.display_height = gl_display->height();
        }

        int frames = 0;
        int total_frames = 0;
        auto fps_start = std::chrono::steady_clock::now();
        while (g_running.load()) {
            if (args.max_frames > 0 && total_frames >= args.max_frames) break;
            auto loop_start = std::chrono::steady_clock::now();
            auto capture_start = std::chrono::steady_clock::now();
            if (!read_capture(pipeline.host_input())) break;
            auto capture_end = std::chrono::steady_clock::now();
            float frame_ms = pipeline.process_frame(display != nullptr);
            auto display_start = std::chrono::steady_clock::now();
            if (display && !write_exact(display, pipeline.host_output(), pipeline.rgb_bytes())) break;
            if (gl_display && !gl_display->render(pipeline.device_output())) break;
            auto display_end = std::chrono::steady_clock::now();
            ++frames;
            ++total_frames;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - fps_start).count();
            if (elapsed >= 0.5) {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.fps = frames / elapsed;
                g_state.frame_ms = frame_ms;
                g_state.capture_ms = std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
                g_state.display_ms = std::chrono::duration<double, std::milli>(display_end - display_start).count();
                g_state.loop_ms = std::chrono::duration<double, std::milli>(now - loop_start).count();
                if (gl_display) {
                    g_state.display_width = gl_display->width();
                    g_state.display_height = gl_display->height();
                }
                fps_start = now;
                frames = 0;
            }
        }
        g_running.store(false);
        if (v4l2_capture) v4l2_capture->stop();
        if (ffmpeg_capture) ffmpeg_capture->stop();
        gl_display.reset();
        v4l2_capture.reset();
        ffmpeg_capture.reset();
        if (camera) pclose(camera);
        if (display) pclose(display);
        if (reexec_if_requested(args)) return 1;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

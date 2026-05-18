#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr int kImageElems = 1 * 3 * kHeight * kWidth;
constexpr int kRgbElems = kHeight * kWidth * 3;
constexpr int kLatentElems = 1 * 4 * 128 * 128;
constexpr int kPromptElems = 1 * 77 * 2048;
constexpr int kTextElems = 1 * 1280;
constexpr int kTimeElems = 1 * 6;
constexpr int kTimestepElems = 1;
constexpr int kParamElems = 4;  // sigma, inv_sigma_scale, scaling, inv_scaling

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
    std::string camera_device = "/dev/video0";
    std::string python = ".venv/bin/python";
    int capture_width = 1920;
    int capture_height = 1080;
    int camera_fps = 30;
    int http_port = 8080;
    int osc_port = 9000;
    int max_frames = 0;
    bool no_display = false;
};

struct AppState {
    std::string prompt = "a cinematic mirror portrait, detailed face, luminous color, sharp focus";
    int seed = 0;
    float strength = 0.7f;
    int steps = 2;
    float blend = 1.0f;
    bool passthrough = false;
    int width = kWidth;
    int height = kHeight;
    double fps = 0.0;
    double frame_ms = 0.0;
    std::string status = "starting";
    bool reload_requested = false;
};

std::mutex g_state_mutex;
AppState g_state;

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
        else if (key == "--camera-device") args.camera_device = val("--camera-device");
        else if (key == "--python") args.python = val("--python");
        else if (key == "--capture-width") args.capture_width = std::stoi(val("--capture-width"));
        else if (key == "--capture-height") args.capture_height = std::stoi(val("--capture-height"));
        else if (key == "--camera-fps") args.camera_fps = std::stoi(val("--camera-fps"));
        else if (key == "--http-port") args.http_port = std::stoi(val("--http-port"));
        else if (key == "--osc-port") args.osc_port = std::stoi(val("--osc-port"));
        else if (key == "--max-frames") args.max_frames = std::stoi(val("--max-frames"));
        else if (key == "--no-display") args.no_display = true;
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: transformirror_fast_app [--camera-device /dev/video0]\n"
                << "       [--engine-dir onnx] [--asset-dir cpp_assets]\n"
                << "       [--http-port 8080] [--osc-port 9000] [--no-display]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
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

void load_device_file(const std::string& path, void* device_ptr, size_t bytes) {
    std::vector<char> data = read_file(path);
    if (data.size() != bytes) {
        throw std::runtime_error(path + " has " + std::to_string(data.size()) +
                                 " bytes, expected " + std::to_string(bytes));
    }
    CHECK_CUDA(cudaMemcpy(device_ptr, data.data(), bytes, cudaMemcpyHostToDevice));
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

__global__ void prepare_unet_kernel(
    const __half* encoded, const __half* noise, const float* params,
    __half* latents, __half* unet_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sigma = params[0];
    float inv_sigma_scale = params[1];
    float scaling = params[2];
    float latent = __half2float(encoded[idx]) * scaling + __half2float(noise[idx]) * sigma;
    latents[idx] = __float2half_rn(latent);
    unet_input[idx] = __float2half_rn(latent * inv_sigma_scale);
}

__global__ void prepare_decode_kernel(
    const __half* latents, const __half* noise_pred, const float* params,
    __half* decode_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sigma = params[0];
    float inv_scaling = params[3];
    float latent = __half2float(latents[idx]) - sigma * __half2float(noise_pred[idx]);
    decode_input[idx] = __float2half_rn(latent * inv_scaling);
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
          d_timestep_(kTimestepElems), d_params_(kParamElems), d_blend_(1), d_passthrough_(1),
          h_src_rgb_(kRgbElems), h_dst_rgb_(kRgbElems) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        bind_engines();
        reload_assets();
        update_blend(1.0f, false);
        capture_graph();
    }

    ~FastPipeline() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    uint8_t* host_input() const { return h_src_rgb_.get(); }
    uint8_t* host_output() const { return h_dst_rgb_.get(); }
    size_t rgb_bytes() const { return h_src_rgb_.bytes(); }

    void update_blend(float blend, bool passthrough) {
        uint8_t pass = passthrough ? 1 : 0;
        CHECK_CUDA(cudaMemcpyAsync(d_blend_.get(), &blend, sizeof(float), cudaMemcpyHostToDevice, stream_));
        CHECK_CUDA(cudaMemcpyAsync(d_passthrough_.get(), &pass, sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));
    }

    void reload_assets() {
        std::lock_guard<std::mutex> lock(asset_mutex_);
        load_device_file(join_path(args_.asset_dir, "noise.fp16"), d_noise_.get(), d_noise_.bytes());
        load_device_file(join_path(args_.asset_dir, "prompt_embeds.fp16"), d_prompt_.get(), d_prompt_.bytes());
        load_device_file(join_path(args_.asset_dir, "text_embeds.fp16"), d_text_.get(), d_text_.bytes());
        load_device_file(join_path(args_.asset_dir, "time_ids.fp16"), d_time_.get(), d_time_.bytes());
        load_device_file(join_path(args_.asset_dir, "timestep.f32"), d_timestep_.get(), d_timestep_.bytes());
        load_device_file(join_path(args_.asset_dir, "params.f32"), d_params_.get(), d_params_.bytes());
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    float process_frame() {
        update_state_controls();
        cudaEvent_t start, end;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&end));
        CHECK_CUDA(cudaEventRecord(start, stream_));
        CHECK_CUDA(cudaMemcpyAsync(d_src_rgb_.get(), h_src_rgb_.get(), h_src_rgb_.bytes(), cudaMemcpyHostToDevice, stream_));
        {
            std::lock_guard<std::mutex> lock(asset_mutex_);
            CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_));
        }
        CHECK_CUDA(cudaMemcpyAsync(h_dst_rgb_.get(), d_dst_rgb_.get(), h_dst_rgb_.bytes(), cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaEventRecord(end, stream_));
        CHECK_CUDA(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, end));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(end));
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

    void enqueue_graph_body() {
        constexpr int block = 256;
        constexpr int pixels = kWidth * kHeight;
        constexpr int pixel_grid = (pixels + block - 1) / block;
        constexpr int latent_grid = (kLatentElems + block - 1) / block;
        preprocess_rgb_kernel<<<pixel_grid, block, 0, stream_>>>(d_src_rgb_.get(), d_image_.get(), pixels);
        encode_.enqueue(stream_);
        prepare_unet_kernel<<<latent_grid, block, 0, stream_>>>(
            d_encoded_.get(), d_noise_.get(), d_params_.get(), d_latents_.get(), d_unet_input_.get(), kLatentElems);
        unet_.enqueue(stream_);
        prepare_decode_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_noise_pred_.get(), d_params_.get(), d_decode_input_.get(), kLatentElems);
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
    DeviceBuffer<float> d_timestep_, d_params_, d_blend_;
    DeviceBuffer<uint8_t> d_passthrough_;
    PinnedBuffer<uint8_t> h_src_rgb_, h_dst_rgb_;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::mutex asset_mutex_;
};

std::string ffmpeg_capture_cmd(const Args& args) {
    std::ostringstream cmd;
    cmd << "ffmpeg -hide_banner -loglevel error "
        << "-f v4l2 -input_format mjpeg -video_size " << args.capture_width << "x" << args.capture_height
        << " -framerate " << args.camera_fps
        << " -i " << shell_quote(args.camera_device)
        << " -vf 'crop=ih:ih:(iw-ih)/2:0,scale=" << kWidth << ":" << kHeight << "'"
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
<label>Resolution</label><input id="resolution" value="1024x1024">
<p><button id="apply">Apply</button></p>
<script>
async function state(){return await (await fetch('/api/state')).json()}
function fill(s){status.textContent=`${s.status} | ${s.fps.toFixed(1)} fps | ${s.frame_ms.toFixed(2)} ms`;
prompt.value=s.prompt;seed.value=s.seed;strength.value=s.strength;strengthv.textContent=s.strength.toFixed(2);
steps.value=s.steps;blend.value=s.blend;blendv.textContent=s.blend.toFixed(2);passthrough.checked=s.passthrough;
resolution.value=`${s.width}x${s.height}`}
for (const id of ['strength','blend']) document.getElementById(id).oninput=()=>document.getElementById(id+'v').textContent=(+document.getElementById(id).value).toFixed(2);
apply.onclick=async()=>{let [w,h]=resolution.value.split('x').map(Number);await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt.value,seed:+seed.value,strength:+strength.value,steps:+steps.value,blend:+blend.value,passthrough:passthrough.checked,width:w||1024,height:h||1024})});fill(await state())}
setInterval(async()=>{let s=await state();status.textContent=`${s.status} | ${s.fps.toFixed(1)} fps | ${s.frame_ms.toFixed(2)} ms`},1000);
state().then(fill);
</script></body></html>)HTML";
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
    std::lock_guard<std::mutex> lock(g_state_mutex);
    std::ostringstream out;
    out << "{"
        << "\"prompt\":\"" << json_escape(g_state.prompt) << "\","
        << "\"seed\":" << g_state.seed << ","
        << "\"strength\":" << g_state.strength << ","
        << "\"steps\":" << g_state.steps << ","
        << "\"blend\":" << g_state.blend << ","
        << "\"passthrough\":" << (g_state.passthrough ? "true" : "false") << ","
        << "\"width\":" << g_state.width << ","
        << "\"height\":" << g_state.height << ","
        << "\"fps\":" << g_state.fps << ","
        << "\"frame_ms\":" << g_state.frame_ms << ","
        << "\"status\":\"" << json_escape(g_state.status) << "\""
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
    size_t e = body.find('"', p + 1);
    if (e == std::string::npos) return false;
    value = body.substr(p + 1, e - p - 1);
    return true;
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
    std::lock_guard<std::mutex> lock(g_state_mutex);
    std::string s;
    double n;
    bool b;
    bool reload = false;
    if (find_string(body, "prompt", s) && s != g_state.prompt) { g_state.prompt = s; reload = true; }
    if (find_number(body, "seed", n) && static_cast<int>(n) != g_state.seed) { g_state.seed = static_cast<int>(n); reload = true; }
    if (find_number(body, "strength", n) && std::fabs(static_cast<float>(n) - g_state.strength) > 1e-5f) { g_state.strength = std::clamp(static_cast<float>(n), 0.0f, 1.0f); reload = true; }
    if (find_number(body, "steps", n) && static_cast<int>(n) != g_state.steps) { g_state.steps = std::clamp(static_cast<int>(n), 1, 8); reload = true; }
    if (find_number(body, "blend", n)) g_state.blend = std::clamp(static_cast<float>(n), 0.0f, 1.0f);
    if (find_bool(body, "passthrough", b)) g_state.passthrough = b;
    if (find_number(body, "width", n) && static_cast<int>(n) != kWidth) g_state.status = "resolution requires matching TensorRT engines; using 1024x1024";
    if (find_number(body, "height", n) && static_cast<int>(n) != kHeight) g_state.status = "resolution requires matching TensorRT engines; using 1024x1024";
    if (reload) {
        g_state.reload_requested = true;
        g_state.status = "conditioning reload queued";
    }
}

void send_http(int client, const std::string& code, const std::string& type, const std::string& body) {
    std::ostringstream res;
    res << "HTTP/1.1 " << code << "\r\nContent-Type: " << type
        << "\r\nContent-Length: " << body.size()
        << "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n" << body;
    std::string text = res.str();
    send(client, text.data(), text.size(), 0);
}

void http_thread(int port) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(server, 16) != 0) {
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
        if (body_pos != std::string::npos) body = req.substr(body_pos + 4);
        if (get && req.find("GET /api/state ") == 0) send_http(client, "200 OK", "application/json", state_json());
        else if (post && req.find("POST /api/state ") == 0) { apply_json_state(body); send_http(client, "200 OK", "application/json", state_json()); }
        else if (get && req.find("GET / ") == 0) send_http(client, "200 OK", "text/html; charset=utf-8", html_page());
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
    }
    json << "}";
    if (wrote) apply_json_state(json.str());
}

void osc_thread(int port) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
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

void asset_reload_thread(const Args& args, FastPipeline* pipeline) {
    while (g_running.load()) {
        bool reload = false;
        AppState snapshot;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            if (g_state.reload_requested) {
                reload = true;
                g_state.reload_requested = false;
                g_state.status = "regenerating conditioning assets";
                snapshot = g_state;
            }
        }
        if (reload) {
            std::ostringstream cmd;
            cmd << shell_quote(args.python) << " export_cpp_assets.py"
                << " --out-dir " << shell_quote(args.asset_dir)
                << " --prompt " << shell_quote(snapshot.prompt)
                << " --seed " << snapshot.seed
                << " --strength " << snapshot.strength
                << " --steps " << snapshot.steps
                << " >/tmp/transformirror_fast_assets.log 2>&1";
            int rc = std::system(cmd.str().c_str());
            if (rc == 0) {
                try {
                    pipeline->reload_assets();
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_state.status = "conditioning assets reloaded";
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_state.status = std::string("asset reload failed: ") + e.what();
                }
            } else {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.status = "asset regeneration failed; see /tmp/transformirror_fast_assets.log";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        CHECK_CUDA(cudaSetDevice(0));
        Logger logger;
        initLibNvInferPlugins(&logger, "");

        FastPipeline pipeline(args, logger);
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_state.status = "running";
        }

        std::thread http(http_thread, args.http_port);
        http.detach();
        std::thread osc(osc_thread, args.osc_port);
        osc.detach();
        std::thread reloader(asset_reload_thread, std::cref(args), &pipeline);
        reloader.detach();

        std::string cap_cmd = ffmpeg_capture_cmd(args);
        std::cerr << "camera: " << cap_cmd << "\n";
        FILE* camera = popen(cap_cmd.c_str(), "r");
        if (!camera) throw std::runtime_error("failed to start ffmpeg camera");

        FILE* display = nullptr;
        if (!args.no_display) {
            std::string display_cmd = ffplay_display_cmd();
            std::cerr << "display: " << display_cmd << "\n";
            display = popen(display_cmd.c_str(), "w");
            if (!display) throw std::runtime_error("failed to start ffplay display");
        }

        int frames = 0;
        int total_frames = 0;
        auto fps_start = std::chrono::steady_clock::now();
        while (g_running.load()) {
            if (args.max_frames > 0 && total_frames >= args.max_frames) break;
            if (!read_exact(camera, pipeline.host_input(), pipeline.rgb_bytes())) break;
            float frame_ms = pipeline.process_frame();
            if (display && !write_exact(display, pipeline.host_output(), pipeline.rgb_bytes())) break;
            ++frames;
            ++total_frames;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - fps_start).count();
            if (elapsed >= 0.5) {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_state.fps = frames / elapsed;
                g_state.frame_ms = frame_ms;
                fps_start = now;
                frames = 0;
            }
        }
        g_running.store(false);
        pclose(camera);
        if (display) pclose(display);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

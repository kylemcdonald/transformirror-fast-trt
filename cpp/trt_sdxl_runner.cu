#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr int kImageElems = 1 * 3 * kHeight * kWidth;
constexpr int kLatentElems = 1 * 4 * 128 * 128;
constexpr int kPromptElems = 1 * 77 * 2048;
constexpr int kTextElems = 1 * 1280;
constexpr int kTimeElems = 1 * 6;
constexpr int kTimestepElems = 1;

constexpr float kSigma = 1.6128870248794556f;
constexpr float kSigmaScale = 1.897736668586731f;
constexpr float kInvSigmaScale = 1.0f / kSigmaScale;
constexpr float kScalingFactor = 1.0f;
constexpr float kInvScalingFactor = 1.0f / kScalingFactor;

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err__) + " at " +       \
                                     __FILE__ + ":" + std::to_string(__LINE__));\
        }                                                                       \
    } while (0)

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
    int warmup = 20;
    int runs = 100;
    bool include_upload = false;
    bool include_download = false;
    bool no_graph = false;
    std::string save_output;
};

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (key == "--engine-dir") {
            args.engine_dir = require_value("--engine-dir");
        } else if (key == "--asset-dir") {
            args.asset_dir = require_value("--asset-dir");
        } else if (key == "--warmup") {
            args.warmup = std::stoi(require_value("--warmup"));
        } else if (key == "--runs") {
            args.runs = std::stoi(require_value("--runs"));
        } else if (key == "--include-upload") {
            args.include_upload = true;
        } else if (key == "--include-download") {
            args.include_download = true;
        } else if (key == "--no-graph") {
            args.no_graph = true;
        } else if (key == "--save-output") {
            args.save_output = require_value("--save-output");
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: trt_sdxl_runner [--engine-dir onnx] [--asset-dir cpp_assets]\n"
                << "                       [--warmup N] [--runs N]\n"
                << "                       [--include-upload] [--include-download]\n"
                << "                       [--no-graph] [--save-output output.fp16]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size)) {
        throw std::runtime_error("failed to read " + path);
    }
    return data;
}

void write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open output " + path);
    }
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error("failed to write output " + path);
    }
}

void load_device_file(const std::string& path, void* device_ptr, size_t expected_bytes) {
    std::vector<char> data = read_file(path);
    if (data.size() != expected_bytes) {
        throw std::runtime_error(path + " has " + std::to_string(data.size()) +
                                 " bytes, expected " + std::to_string(expected_bytes));
    }
    CHECK_CUDA(cudaMemcpy(device_ptr, data.data(), expected_bytes, cudaMemcpyHostToDevice));
}

class TrtEngine {
  public:
    TrtEngine(Logger& logger, const std::string& path) {
        std::vector<char> plan = read_file(path);
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_) {
            throw std::runtime_error("failed to create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) {
            throw std::runtime_error("failed to deserialize TensorRT plan " + path);
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("failed to create TensorRT execution context " + path);
        }
    }

    void bind(const char* name, void* ptr) {
        if (!context_->setTensorAddress(name, ptr)) {
            throw std::runtime_error(std::string("failed to bind TensorRT tensor ") + name);
        }
    }

    void enqueue(cudaStream_t stream) {
        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
    }

  private:
    struct RuntimeDeleter {
        void operator()(nvinfer1::IRuntime* ptr) const { delete ptr; }
    };
    struct EngineDeleter {
        void operator()(nvinfer1::ICudaEngine* ptr) const { delete ptr; }
    };
    struct ContextDeleter {
        void operator()(nvinfer1::IExecutionContext* ptr) const { delete ptr; }
    };

    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
};

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t elems = 0) { reset(elems); }
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reset(size_t elems) {
        elems_ = elems;
        if (elems_ == 0) {
            return;
        }
        CHECK_CUDA(cudaMalloc(&ptr_, elems_ * sizeof(T)));
    }

    T* get() const { return ptr_; }
    size_t bytes() const { return elems_ * sizeof(T); }
    size_t elems() const { return elems_; }

  private:
    T* ptr_ = nullptr;
    size_t elems_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(size_t elems = 0) { reset(elems); }
    ~PinnedBuffer() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void reset(size_t elems) {
        elems_ = elems;
        if (elems_ == 0) {
            return;
        }
        CHECK_CUDA(cudaMallocHost(&ptr_, elems_ * sizeof(T)));
    }

    T* get() const { return ptr_; }
    size_t bytes() const { return elems_ * sizeof(T); }

  private:
    T* ptr_ = nullptr;
    size_t elems_ = 0;
};

__global__ void prepare_unet_kernel(
    const __half* encoded,
    const __half* noise,
    __half* latents,
    __half* unet_input,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float latent = __half2float(encoded[idx]) * kScalingFactor +
                   __half2float(noise[idx]) * kSigma;
    latents[idx] = __float2half_rn(latent);
    unet_input[idx] = __float2half_rn(latent * kInvSigmaScale);
}

__global__ void prepare_decode_kernel(
    const __half* latents,
    const __half* noise_pred,
    __half* decode_input,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float latent = __half2float(latents[idx]) - kSigma * __half2float(noise_pred[idx]);
    decode_input[idx] = __float2half_rn(latent * kInvScalingFactor);
}

__global__ void postprocess_kernel(const __half* decoded, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = __half2float(decoded[idx]) * 0.5f + 0.5f;
    value = fminf(1.0f, fmaxf(0.0f, value));
    output[idx] = __float2half_rn(value);
}

struct Summary {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double p90 = 0.0;
};

Summary summarize(std::vector<double> values) {
    Summary s;
    if (values.empty()) {
        return s;
    }
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();
    s.median = values[values.size() / 2];
    s.p90 = values[static_cast<size_t>(0.9 * static_cast<double>(values.size() - 1))];
    return s;
}

class Runner {
  public:
    Runner(const Args& args, Logger& logger)
        : args_(args),
          encode_(logger, join_path(args.engine_dir, "taesdxl_encode.plan")),
          unet_(logger, join_path(args.engine_dir, "sdxl_turbo_unet.plan")),
          decode_(logger, join_path(args.engine_dir, "taesdxl_decode.plan")),
          d_image_(kImageElems),
          d_encoded_(kLatentElems),
          d_latents_(kLatentElems),
          d_unet_input_(kLatentElems),
          d_noise_pred_(kLatentElems),
          d_decode_input_(kLatentElems),
          d_decoded_(kImageElems),
          d_output_(kImageElems),
          d_noise_(kLatentElems),
          d_prompt_(kPromptElems),
          d_text_(kTextElems),
          d_time_(kTimeElems),
          d_timestep_(kTimestepElems),
          h_input_(kImageElems),
          h_output_(kImageElems) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        load_assets();
        bind_engines();
        if (!args_.no_graph) {
            capture_graph();
        }
    }

    ~Runner() {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void run_frame() {
        if (graph_exec_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_));
        } else {
            enqueue_frame();
        }
    }

    void timed_run(std::vector<double>& gpu_times, std::vector<double>& wall_times) {
        cudaEvent_t start;
        cudaEvent_t end;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&end));

        auto wall_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaEventRecord(start, stream_));
        if (args_.include_upload) {
            CHECK_CUDA(cudaMemcpyAsync(
                d_image_.get(), h_input_.get(), d_image_.bytes(), cudaMemcpyHostToDevice, stream_));
        }
        run_frame();
        if (args_.include_download) {
            CHECK_CUDA(cudaMemcpyAsync(
                h_output_.get(), d_output_.get(), d_output_.bytes(), cudaMemcpyDeviceToHost, stream_));
        }
        CHECK_CUDA(cudaEventRecord(end, stream_));
        CHECK_CUDA(cudaEventSynchronize(end));
        auto wall_end = std::chrono::steady_clock::now();

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, end));
        gpu_times.push_back(static_cast<double>(elapsed_ms));
        wall_times.push_back(
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count());

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(end));
    }

    void warmup() {
        for (int i = 0; i < args_.warmup; ++i) {
            if (args_.include_upload) {
                CHECK_CUDA(cudaMemcpyAsync(
                    d_image_.get(), h_input_.get(), d_image_.bytes(), cudaMemcpyHostToDevice, stream_));
            }
            run_frame();
            if (args_.include_download) {
                CHECK_CUDA(cudaMemcpyAsync(
                    h_output_.get(), d_output_.get(), d_output_.bytes(), cudaMemcpyDeviceToHost, stream_));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    void save_output() {
        if (args_.save_output.empty()) {
            return;
        }
        CHECK_CUDA(cudaMemcpy(
            h_output_.get(), d_output_.get(), d_output_.bytes(), cudaMemcpyDeviceToHost));
        write_file(args_.save_output, h_output_.get(), d_output_.bytes());
    }

  private:
    void load_assets() {
        const std::string asset_dir = args_.asset_dir;
        std::vector<char> input = read_file(join_path(asset_dir, "input_image.fp16"));
        if (input.size() != d_image_.bytes()) {
            throw std::runtime_error("input_image.fp16 size mismatch");
        }
        std::copy(input.begin(), input.end(), reinterpret_cast<char*>(h_input_.get()));
        CHECK_CUDA(cudaMemcpy(d_image_.get(), h_input_.get(), d_image_.bytes(), cudaMemcpyHostToDevice));

        load_device_file(join_path(asset_dir, "noise.fp16"), d_noise_.get(), d_noise_.bytes());
        load_device_file(join_path(asset_dir, "prompt_embeds.fp16"), d_prompt_.get(), d_prompt_.bytes());
        load_device_file(join_path(asset_dir, "text_embeds.fp16"), d_text_.get(), d_text_.bytes());
        load_device_file(join_path(asset_dir, "time_ids.fp16"), d_time_.get(), d_time_.bytes());
        load_device_file(join_path(asset_dir, "timestep.f32"), d_timestep_.get(), d_timestep_.bytes());
    }

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

    void enqueue_frame() {
        constexpr int block = 256;
        constexpr int latent_grid = (kLatentElems + block - 1) / block;
        constexpr int image_grid = (kImageElems + block - 1) / block;

        encode_.enqueue(stream_);
        prepare_unet_kernel<<<latent_grid, block, 0, stream_>>>(
            d_encoded_.get(), d_noise_.get(), d_latents_.get(), d_unet_input_.get(), kLatentElems);
        unet_.enqueue(stream_);
        prepare_decode_kernel<<<latent_grid, block, 0, stream_>>>(
            d_latents_.get(), d_noise_pred_.get(), d_decode_input_.get(), kLatentElems);
        decode_.enqueue(stream_);
        postprocess_kernel<<<image_grid, block, 0, stream_>>>(
            d_decoded_.get(), d_output_.get(), kImageElems);
    }

    void capture_graph() {
        for (int i = 0; i < 5; ++i) {
            enqueue_frame();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        cudaGraph_t graph = nullptr;
        CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
        enqueue_frame();
        CHECK_CUDA(cudaStreamEndCapture(stream_, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0));
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    const Args& args_;
    TrtEngine encode_;
    TrtEngine unet_;
    TrtEngine decode_;

    DeviceBuffer<__half> d_image_;
    DeviceBuffer<__half> d_encoded_;
    DeviceBuffer<__half> d_latents_;
    DeviceBuffer<__half> d_unet_input_;
    DeviceBuffer<__half> d_noise_pred_;
    DeviceBuffer<__half> d_decode_input_;
    DeviceBuffer<__half> d_decoded_;
    DeviceBuffer<__half> d_output_;
    DeviceBuffer<__half> d_noise_;
    DeviceBuffer<__half> d_prompt_;
    DeviceBuffer<__half> d_text_;
    DeviceBuffer<__half> d_time_;
    DeviceBuffer<float> d_timestep_;

    PinnedBuffer<__half> h_input_;
    PinnedBuffer<__half> h_output_;

    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        Logger logger;
        initLibNvInferPlugins(&logger, "");
        CHECK_CUDA(cudaSetDevice(0));

        Runner runner(args, logger);
        runner.warmup();

        std::vector<double> gpu_times;
        std::vector<double> wall_times;
        gpu_times.reserve(args.runs);
        wall_times.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i) {
            runner.timed_run(gpu_times, wall_times);
            std::cout << "run " << std::setw(3) << (i + 1) << "/" << args.runs
                      << ": gpu=" << std::fixed << std::setprecision(3) << gpu_times.back()
                      << " ms, wall=" << wall_times.back() << " ms\n";
        }
        runner.save_output();

        Summary gpu = summarize(gpu_times);
        Summary wall = summarize(wall_times);
        std::cout << "{\n"
                  << "  \"backend\": \"cpp_tensorrt_cuda_graph\",\n"
                  << "  \"runs\": " << args.runs << ",\n"
                  << "  \"include_upload\": " << (args.include_upload ? "true" : "false") << ",\n"
                  << "  \"include_download\": " << (args.include_download ? "true" : "false") << ",\n"
                  << "  \"cuda_graph\": " << (!args.no_graph ? "true" : "false") << ",\n"
                  << "  \"gpu_mean_ms\": " << gpu.mean << ",\n"
                  << "  \"gpu_median_ms\": " << gpu.median << ",\n"
                  << "  \"gpu_min_ms\": " << gpu.min << ",\n"
                  << "  \"gpu_p90_ms\": " << gpu.p90 << ",\n"
                  << "  \"gpu_max_ms\": " << gpu.max << ",\n"
                  << "  \"wall_mean_ms\": " << wall.mean << ",\n"
                  << "  \"wall_median_ms\": " << wall.median << ",\n"
                  << "  \"wall_min_ms\": " << wall.min << ",\n"
                  << "  \"wall_p90_ms\": " << wall.p90 << ",\n"
                  << "  \"wall_max_ms\": " << wall.max << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

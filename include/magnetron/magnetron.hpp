// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
// Magnetron Machine Learning Framework, public C++ 20 API.

#pragma once

#include "magnetron.h"

#include <string_view>
#include <thread>
#include <variant>

#include "magnetron_internal.h"

namespace magnetron {
    constexpr std::size_t k_default_chunk_size {MAG_DEFAULT_CHUNK_SIZE};
    constexpr std::size_t k_default_chunk_cap {MAG_DEFAULT_CHUNK_CAP};
    constexpr std::size_t k_max_dims {MAG_MAX_DIMS};
    constexpr std::size_t k_max_tensor_name_len {MAG_MAX_TENSOR_NAME_LEN};
    constexpr std::size_t k_max_input_tensors {MAG_MAX_INPUT_TENSORS};
    constexpr std::size_t k_max_op_params {MAG_MAX_OP_PARAMS};
    constexpr std::uint16_t k_version {MAG_VERSION};
    constexpr std::uint8_t k_version_major {mag_version_major(MAG_VERSION)};
    constexpr std::uint8_t k_version_minor {mag_version_minor(MAG_VERSION)};

    /**
     * Enumerates compute device types.
     */
    enum class compute_device : std::underlying_type_t<mag_compute_device_t> {
        cpu = MAG_COMPUTE_DEVICE_TYPE_CPU,
        gpu_cuda = MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA
    };

    /**
     * Get enumerator name of compute device.
     * @param device Compute device.
     * @return Enum name of the compute device.
     */
    [[nodiscard]] inline auto compute_device_name(compute_device device) noexcept -> std::string_view {
        return mag_device_type_get_name(static_cast<mag_compute_device_type_t>(device));
    }

    // Operator evaluation mode
    enum class exec_mode : std::underlying_type_t<mag_exec_mode_t> {
        /**
         * Execute operations eagerly (immediately and synchronously)
         */
        eager = MAG_EXEC_MODE_EAGER,

        /**
         * Execute operations deferred (lazily and asynchronously)
         */
        deferred = MAG_EXEC_MODE_DEFERRED
    };

    /**
     * Pseudo-random number generator (PRNG) algorithm
     */
    enum class prng_algorithm : std::underlying_type_t<mag_prng_algorithm_t> {
        /**
         * Merseene Twister 64
         */
        mersenne_twister = MAG_PRNG_MERSENNE_TWISTER,

        /**
         * Permuted Congruential Generator
         */
        pcg = MAG_PRNG_PCG
    };

    /**
     * Thread scheduling priority for CPU compute, higher priority means more CPU time
     */
    enum class thread_sched_prio : std::underlying_type_t<mag_thread_sched_prio_t> {
        normal = MAG_THREAD_SCHED_PRIO_NORMAL,
        medium = MAG_THREAD_SCHED_PRIO_MEDIUM,
        high = MAG_THREAD_SCHED_PRIO_HIGH,
        realtime = MAG_THREAD_SCHED_PRIO_REALTIME
    };

    /**
     * Desired color channels to load from image tensor
     */
    enum class color_channel : std::underlying_type_t<mag_color_channels_t> {
        automatic = MAG_COLOR_CHANNELS_AUTO,
        grayscale = MAG_COLOR_CHANNELS_GRAY,
        grayscale_alpha = MAG_COLOR_CHANNELS_GRAY_A,
        rgb = MAG_COLOR_CHANNELS_RGB,
        rgba = MAG_COLOR_CHANNELS_RGBA
    };

    /**
     * Function type used for all memory allocations in magnetron. Can be overwritten by user.
     * @param block Block of memory to reallocate or nullptr.
     * @param size Size to resize to or 0.
     * @return Allocated or reallocated block or nullptr.
     */
    using alloc_fn = auto(void* block, std::size_t size) -> void*;

    /**
     * Get the global allocator function.
     * @return Get the global allocator function.
     */
    [[nodiscard]] inline auto allocator() noexcept -> alloc_fn* {
        return mag_get_alloc_fn();
    }

    /**
     * Set the global allocator function.
     * @param alloc Allocator function to set.
     */
    inline auto set_allocator(alloc_fn* alloc) noexcept -> void {
        mag_set_alloc_fn(alloc);
    }

    /**
     * Enable or disable internal magnetron logging to stdout.
     * @param enable
     */
    inline auto enable_logging(bool enable) noexcept -> void {
        mag_set_log_mode(enable);
    }

    [[nodiscard]] inline auto pack_color_u8(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept -> std::uint32_t {
        return mag_pack_color_u8(r, g, b);
    }

    [[nodiscard]] inline auto pack_color_f32(float r, float g, float b) noexcept -> std::uint32_t {
        return mag_pack_color_f32(r, g, b);
    }

    /**
     * CPU compute device creation info.
     */
    struct cpu_device final {
        /**
         * Amount of threads to use for CPU parallel processing.
         */
        std::uint32_t thread_count {std::max(1u, std::thread::hardware_concurrency())};
    };

    /**
     * CUDA GPU device creation info.
     */
    struct cuda_device final {
        /**
         * GPU index to use. 0 = first GPU, 1 = second etc.
         */
        std::uint32_t device_index {};
    };

    using device_descriptor = std::variant<cpu_device, cuda_device>;

    /**
     * The context owns all tensors and runtime data structures. It must kept alive as long as any tensor is used.
     */
    class context final {
    public:
        explicit context(compute_device dvc) noexcept {
            m_ctx = mag_ctx_create(static_cast<mag_compute_device_type_t>(dvc));
        }

        explicit context(device_descriptor device) {
            mag_device_descriptor_t desc {};
            if (std::holds_alternative<cpu_device>(device)) {
                const auto& cpu = std::get<cpu_device>(device);
                desc.type = MAG_COMPUTE_DEVICE_TYPE_CPU;
                desc.thread_count = cpu.thread_count;
            } else if (std::holds_alternative<cuda_device>(device)) {
                const auto& cuda = std::get<cuda_device>(device);
                desc.type = MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA;
                desc.cuda_device_id = cuda.device_index;
            } else {
                throw std::invalid_argument("Invalid device type");
            }
        }

        context(context&&) = default;
        context& operator=(context&&) = default;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&) -> context& = delete;

        ~context() {
            mag_ctx_destroy(m_ctx);
        }

        [[nodiscard]] auto operator *() noexcept -> mag_ctx_t& { return *m_ctx; }
        [[nodiscard]] auto operator *() const noexcept -> const mag_ctx_t& { return *m_ctx; }
        [[nodiscard]] auto exec_mode() const noexcept -> exec_mode { return static_cast<enum exec_mode>(mag_ctx_get_exec_mode(m_ctx)); }
        auto exec_mode(enum exec_mode mode) noexcept -> void { mag_ctx_set_exec_mode(m_ctx, static_cast<mag_exec_mode_t>(mode)); }
        [[nodiscard]] auto prng_algorithm() const noexcept -> prng_algorithm { return static_cast<enum prng_algorithm>(mag_ctx_get_prng_algorithm(m_ctx)); }
        auto prng_algorithm(enum prng_algorithm algorithm, std::uint64_t seed) noexcept -> void { mag_ctx_set_prng_algorithm(m_ctx, static_cast<mag_prng_algorithm_t>(algorithm), seed); }
        [[nodiscard]] auto device_type() const noexcept -> compute_device { return static_cast<compute_device>(mag_ctx_get_compute_device_type(m_ctx)); }
        [[nodiscard]] auto device_name() const noexcept -> std::string_view { return mag_ctx_get_compute_device_name(m_ctx); }
        [[nodiscard]] auto os_name() const noexcept -> std::string_view { return mag_ctx_get_os_name(m_ctx); }
        [[nodiscard]] auto cpu_name() const noexcept -> std::string_view { return mag_ctx_get_cpu_name(m_ctx); }
        [[nodiscard]] auto cpu_virtual_cores() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_virtual_cores(m_ctx); }
        [[nodiscard]] auto cpu_physical_cores() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_physical_cores(m_ctx); }
        [[nodiscard]] auto cpu_sockets() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_sockets(m_ctx); }
        [[nodiscard]] auto physical_memory_total() const noexcept -> std::uint64_t { return mag_ctx_get_physical_memory_total(m_ctx); }
        [[nodiscard]] auto physical_memory_free() const noexcept -> std::uint64_t { return mag_ctx_get_physical_memory_free(m_ctx); }
        [[nodiscard]] auto is_numa_system() const noexcept -> bool { return mag_ctx_is_numa_system(m_ctx); }
        [[nodiscard]] auto total_tensors_created() const noexcept -> std::size_t { return mag_ctx_get_total_tensors_created(m_ctx); }
        auto start_profiler() noexcept -> void { mag_ctx_profiler_start(m_ctx); }
        auto end_profiler(const char* export_csv_file = nullptr) noexcept -> void { mag_ctx_profiler_end(m_ctx, export_csv_file); }
        [[nodiscard]] auto is_profiling() const noexcept -> bool { return mag_ctx_profiler_is_running(m_ctx); }
        auto start_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_start(m_ctx); }
        auto stop_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_stop(m_ctx); }
        [[nodiscard]] auto is_recording_gradients() const noexcept -> bool { return mag_ctx_grad_recorder_is_running(m_ctx); }

    private:
        mag_ctx_t* m_ctx {};
    };

    enum class dtype : std::underlying_type_t<mag_dtype_t> {
        e8m23 = MAG_DTYPE_E8M23,
        f32 = e8m23,
        e5m10 = MAG_DTYPE_E5M10,
        f16 = e5m10
    };

    [[nodiscard]] inline auto dtype_size(dtype t) noexcept -> std::size_t {
        return mag_dtype_meta_of(static_cast<mag_dtype_t>(t))->size;
    }

    [[nodiscard]] inline auto dtype_name(dtype t) noexcept -> std::string_view {
        return mag_dtype_meta_of(static_cast<mag_dtype_t>(t))->name;
    }

    /**
     * A 1-6 dimensional, reference counted tensor with a fixed size and data type.
     */
    class tensor final {
    public:
        tensor(context& ctx, dtype type, const std::span<std::int64_t> shape) {
            switch (shape.size()) {
                case 1: m_tensor = mag_tensor_create_1d(&*ctx, static_cast<mag_dtype_t>(type), shape[0]); break;
                case 2: m_tensor = mag_tensor_create_2d(&*ctx, static_cast<mag_dtype_t>(type), shape[0], shape[1]); break;
                case 3: m_tensor = mag_tensor_create_3d(&*ctx, static_cast<mag_dtype_t>(type), shape[0], shape[1], shape[2]); break;
                case 4: m_tensor = mag_tensor_create_4d(&*ctx, static_cast<mag_dtype_t>(type), shape[0], shape[1], shape[2], shape[3]); break;
                case 5: m_tensor = mag_tensor_create_5d(&*ctx, static_cast<mag_dtype_t>(type), shape[0], shape[1], shape[2], shape[3], shape[4]); break;
                case 6: m_tensor = mag_tensor_create_6d(&*ctx, static_cast<mag_dtype_t>(type), shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]); break;
                default: throw std::invalid_argument("Invalid tensor shape: " + std::to_string(shape.size()));
            }
        }

        tensor(const tensor& other) {
            mag_tensor_retain(other.m_tensor);
            m_tensor = other.m_tensor;
        }

        tensor(tensor&& other) {
            if (this != &other) {
                m_tensor = other.m_tensor;
                other.m_tensor = nullptr;
            }
        }

        auto operator = (const tensor& other) -> tensor& {
            if (this != &other) {
                mag_tensor_retain(other.m_tensor);
                mag_tensor_decref(m_tensor);
                m_tensor = other.m_tensor;
            }
            return *this;
        }

        auto operator = (tensor&& other) -> tensor& {
            if (this != &other) {
                mag_tensor_decref(m_tensor);
                m_tensor = other.m_tensor;
                other.m_tensor = nullptr;
            }
            return *this;
        }

        ~tensor() {
            if (m_tensor) {
                mag_tensor_decref(m_tensor);
            }
        }

        [[nodiscard]] auto operator * () noexcept -> mag_tensor_t& { return *m_tensor; }
        [[nodiscard]] auto operator * () const noexcept -> const mag_tensor_t& { return *m_tensor; }

    private:
        mag_tensor_t* m_tensor {};
    };
}

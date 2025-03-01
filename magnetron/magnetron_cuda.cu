/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include <array>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda.h>

#include  "magnetron_internal.h"

namespace mag::cuda {
    /* Driver result check. */
    #define mag_cu_chk_rdv(expr) \
        do { \
        if (auto rrr {(expr)}; rrr != CUDA_SUCCESS) [[unlikely]] { \
            const char* err_str = "?"; \
            cuGetErrorString(rrr, &err_str); \
            mag_panic(#expr, __func__, __FILE__, __LINE__, err_str); \
        } \
    } while (0)

    /* Runtime result check. */
    #define mag_cu_chk_rt(expr) \
        do { \
            if (auto rrr {(expr)}; rrr != cudaSuccess) [[unlikely]] { \
                mag_panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
            } \
        } while (0)

    constexpr std::size_t max_devices = 32;

    struct physical_device final {
        std::int32_t id {};             /* Device ID */
        std::array<char, 128> name {};  /* Device name */
        std::size_t vram {};            /* Video memory in bytes */
        std::uint32_t cl {};            /* Compute capability */
        std::uint32_t nsm {};           /* Number of SMs */
        std::uint32_t ntpb {};          /* Number of threads per block */
        std::size_t smpb {};            /* Shared memory per block */
        std::size_t smpb_opt {};        /* Shared memory per block opt-in */
        bool has_vmm {};                /* Has virtual memory management */
        std::size_t vmm_granularity {}; /* Virtual memory management granularity */
    };

    namespace kernels {
        template <typename T>
        static __global__ auto broadcast(std::int64_t numel, T* px, T x) -> void {
            for (std::int64_t i {blockIdx.x*blockDim.x + threadIdx.x}; i < numel; i += gridDim.x*blockDim.x)
                px[i] = x;
        }

        template <typename T>
        static __global__ auto add(std::int64_t numel, T* pr, const T* px, const T* py) -> void {
            for (std::int64_t i {blockIdx.x*blockDim.x + threadIdx.x}; i < numel; i += gridDim.x*blockDim.x)
                pr[i] = px[i] + py[i];
        }
    }

    static auto add(mag_tensor_t* r) -> void {
        mag_tensor_t* x {r->op_inputs[0]};
        mag_tensor_t* y {r->op_inputs[1]};
        mag_assert2(x->numel == y->numel);
        mag_assert2(x->numel == r->numel);
        std::int64_t block_size {256};
        std::int64_t num_blocks {(r->numel + block_size - 1)/block_size};
        auto* pr {reinterpret_cast<mag_e8m23_t*>(r->storage.base)};
        auto* px {reinterpret_cast<mag_e8m23_t*>(x->storage.base)};
        auto* py {reinterpret_cast<mag_e8m23_t*>(y->storage.base)};
        kernels::add<<<num_blocks, block_size>>>(r->numel, pr, px, py);
    }

    static constexpr auto (*const forward_kernels[MAG_OP__NUM])(mag_tensor_t*) -> void = {
        [MAG_OP_NOP] = nullptr,
        [MAG_OP_CLONE] = nullptr,
        [MAG_OP_VIEW] = nullptr,
        [MAG_OP_TRANSPOSE] = nullptr,
        [MAG_OP_PERMUTE] = nullptr,
        [MAG_OP_MEAN] = nullptr,
        [MAG_OP_MIN] = nullptr,
        [MAG_OP_MAX] = nullptr,
        [MAG_OP_SUM] = nullptr,
        [MAG_OP_ABS] = nullptr,
        [MAG_OP_NEG] = nullptr,
        [MAG_OP_LOG] = nullptr,
        [MAG_OP_SQR] = nullptr,
        [MAG_OP_SQRT] = nullptr,
        [MAG_OP_SIN] = nullptr,
        [MAG_OP_COS] = nullptr,
        [MAG_OP_STEP] = nullptr,
        [MAG_OP_EXP] = nullptr,
        [MAG_OP_SOFTMAX] = nullptr,
        [MAG_OP_SOFTMAX_DV] = nullptr,
        [MAG_OP_SIGMOID] = nullptr,
        [MAG_OP_SIGMOID_DV] = nullptr,
        [MAG_OP_HARD_SIGMOID] = nullptr,
        [MAG_OP_SILU] = nullptr,
        [MAG_OP_SILU_DV] = nullptr,
        [MAG_OP_TANH] = nullptr,
        [MAG_OP_TANH_DV] = nullptr,
        [MAG_OP_RELU] = nullptr,
        [MAG_OP_RELU_DV] = nullptr,
        [MAG_OP_GELU] = nullptr,
        [MAG_OP_GELU_DV] = nullptr,
        [MAG_OP_ADD] = &add,
        [MAG_OP_SUB] = nullptr,
        [MAG_OP_MUL] = nullptr,
        [MAG_OP_DIV] = nullptr,
        [MAG_OP_ADDS] = nullptr,
        [MAG_OP_SUBS] = nullptr,
        [MAG_OP_MULS] = nullptr,
        [MAG_OP_DIVS] = nullptr,
        [MAG_OP_POWS] = nullptr,
        [MAG_OP_MATMUL] = nullptr,
    };

    static auto broadcast_storage_buffer(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t stride) {
        mag_assert2(stride == 4);
        auto numel {static_cast<std::int64_t>(sto->size/4)};
        std::int64_t block_size {256};
        std::int64_t num_blocks {(numel + block_size - 1)/block_size};
        mag_e8m23_t* dst {reinterpret_cast<mag_e8m23_t*>(sto->base+offs)};
        mag_e8m23_t x {*reinterpret_cast<const mag_e8m23_t*>(src)};
        kernels::broadcast<<<num_blocks, block_size>>>(numel, dst+offs, x);
        cudaDeviceSynchronize();
    }

    static auto cpy_host_device(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t sz) -> void {
        mag_cu_chk_rt(cudaMemcpy(reinterpret_cast<void*>(sto->base+offs), src, sz, cudaMemcpyHostToDevice));
    }

    static auto cpy_device_host(mag_storage_buffer_t* sto, size_t offs, void* dst, size_t sz) -> void {
        mag_cu_chk_rt(cudaMemcpy(dst, reinterpret_cast<void*>(sto->base+offs), sz, cudaMemcpyDeviceToHost));
    }

    static auto alloc_storage_buffer(mag_compute_device_t* dvc, mag_storage_buffer_t* out, std::size_t sz) -> void {
        void* base;
        mag_cu_chk_rt(cudaMalloc(&base, sz));
        *out = mag_storage_buffer_t {
            .base = reinterpret_cast<std::uintptr_t>(base),
            .size = sz,
            .alignment = 256,
            .host = dvc,
            .broadcast = &broadcast_storage_buffer,
            .cpy_host_device = &cpy_host_device,
            .cpy_device_host = &cpy_device_host,
        };
    }

    static auto free_tensor_buffer(mag_compute_device_t* dvc, mag_storage_buffer_t* buf) -> void {
        mag_cu_chk_rt(cudaFree(reinterpret_cast<void*>(buf->base)));
    }

    static auto exec_fwd(mag_compute_device_t* dvc, mag_tensor_t* root) -> void {
        mag_assert2(root->op < MAG_OP__NUM);
        (*forward_kernels[root->op])(root);
        cudaDeviceSynchronize();
    }

    extern "C" auto mag_init_device_cuda([[maybe_unused]] mag_ctx_t* ctx) -> mag_compute_device_t* {
        std::int32_t requested_device_id {0};
        std::int32_t numgpus {};
        if (cudaGetDeviceCount(&numgpus) != cudaSuccess) [[unlikely]] return nullptr;
        if (numgpus < 1 || numgpus > max_devices) [[unlikely]] return nullptr;
        std::vector<physical_device> device_list {};
        device_list.reserve(numgpus);
        for (std::int32_t id {}; id < numgpus; ++id) { /* Iterate over devices */
            physical_device& dvc {device_list.emplace_back()};
            CUdevice cu_dvc {};
            std::int32_t vmm_support {};
            if (cuDeviceGet(&cu_dvc, id) != CUDA_SUCCESS) [[unlikely]] continue; /* Get device handle */
            if (cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc) != CUDA_SUCCESS) [[unlikely]] continue; /* Check VMM support */
            if (vmm_support) { /* Virtual memory management supported */
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = id;
                if (cuMemGetAllocationGranularity(&dvc.vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) [[unlikely]] continue; /* Get VMM granularity */
            }
            dvc.has_vmm = !!vmm_support;
            cudaDeviceProp props {};
            if (cudaGetDeviceProperties(&props, id) != cudaSuccess) [[unlikely]] continue; /* Get device properties */
            std::snprintf(dvc.name.data(), dvc.name.size(), "%s", props.name);
            dvc.id = id;
            dvc.nsm = props.multiProcessorCount;
            dvc.smpb = props.sharedMemPerBlock;
            dvc.smpb_opt = props.sharedMemPerBlockOptin;
            dvc.cl = 100*props.major + 10*props.minor;
            dvc.ntpb = props.maxThreadsPerBlock;
            dvc.vram = props.totalGlobalMem;
        }
        if (device_list.empty()) [[unlikely]] { /* No devices available or initialization failed, let runtime fallback to other compute device. */
            mag_log_error("No CUDA devices with id %d available, using CPU processing", requested_device_id);
            return nullptr;
        }
        auto& active_device {device_list[std::clamp(requested_device_id, 0, numgpus-1)]};
        auto* dvc {static_cast<mag_compute_device_t*>((*mag_alloc)(nullptr, sizeof(mag_compute_device_t)))};
        new (dvc) mag_compute_device_t {
            .name = "GPU",
            .impl = nullptr,
            .is_async = true,
            .type = MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA,
            .eager_exec_fwd = &exec_fwd,
            .eager_exec_bwd = nullptr,
            .alloc_storage = &alloc_storage_buffer,
            .free_storage = &free_tensor_buffer
        };
        double vram;
        const char* unit;
        mag_humanize_memory_size(active_device.vram, &vram, &unit);
        std::snprintf(dvc->name, sizeof(dvc->name), "%s - %s - %.03f %s VRAM", mag_device_type_get_name(dvc->type), active_device.name.data(), vram, unit);
        return dvc;
    }

    extern "C" auto mag_destroy_device_cuda(mag_compute_device_t* dvc) -> void {
        dvc->~mag_compute_device_t();
        (*mag_alloc)(dvc, 0);
    }
}

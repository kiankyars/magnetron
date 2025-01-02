/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "magnetron_cuda.cuh"

#include <bit>
#include <algorithm>
#include <cstdio>
#include <vector>

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

    #define mag_cu_assert(expr, msg, ...) \
        if (!(expr)) [[unlikely]] { \
            mag_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
        }
    #define mag_cu_assert2(expr) mag_cu_assert(expr, "")

    static constinit bool s_is_init {};
    static constinit std::int32_t active_device_id {};
    static std::vector<physical_device> s_devices {};

    auto mag_init_device_cuda([[maybe_unused]] mag_ctx_t* ctx) -> mag_compute_device_t* {
        std::int32_t active_device_id {0}; // TODO: Implement device selection.
        const physical_device* device {cuda_init(active_device_id)};
        if (!device) [[unlikely]] { /* No devices available or initialization failed, let runtime fallback to other compute device. */
            mag_log_error("No CUDA devices with id %d available, using CPU processing", active_device_id);
            return nullptr;
        }
        auto* dvc {static_cast<mag_compute_device_t*>((*mag_alloc)(nullptr, sizeof(mag_compute_device_t)))};
        new (dvc) mag_compute_device_t {
            .name = "GPU",
            .impl = nullptr,
            .is_async = true,
            .type = MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA,
            .eager_exec_fwd = nullptr,
            .eager_exec_bwd = nullptr,
            .alloc_storage = nullptr,
            .free_storage = nullptr
        };
        double vram;
        const char* unit;
        mag_humanize_memory_size(device->vram, &vram, &unit);
        std::snprintf(dvc->name, sizeof(dvc->name), "%s - %s - %.03f %s VRAM", mag_device_type_get_name(dvc->type), device->name.data(), vram, unit);
        return dvc;
    }

    void mag_destroy_device_cuda(mag_compute_device_t* dvc) {
        s_is_init = false;
        s_devices.clear();
        dvc->~mag_compute_device_t();
        (*mag_alloc)(dvc, 0);
    }

    /*
    ** Initialize CUDA runtime. Returns empty span if initialization failed or not devices are available.
    ** Normally, we panic when some CUDA runtime function fails, but in this case we just return an empty span,
    ** to allow the runtime to fall back to CPU processing, if no CUDA devices are available.
    */
    auto cuda_init(std::int32_t use_device) -> const physical_device* {
        if (s_is_init) return &s_devices[active_device_id];
        std::int32_t numgpus {};
        if (cudaGetDeviceCount(&numgpus) != cudaSuccess) [[unlikely]] return nullptr;
        if (numgpus < 1 || numgpus > max_devices) [[unlikely]] return nullptr;
        s_devices.reserve(numgpus);
        for (std::int32_t id {}; id < numgpus; ++id) { /* Iterate over devices */
            physical_device& dvc {s_devices.emplace_back()};
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
            dvc.id = id;
            dvc.name = std::bit_cast<decltype(dvc.name)>(props.name);
            dvc.nsm = props.multiProcessorCount;
            dvc.smpb = props.sharedMemPerBlock;
            dvc.smpb_opt = props.sharedMemPerBlockOptin;
            dvc.cl = 100*props.major + 10*props.minor;
            dvc.ntpb = props.maxThreadsPerBlock;
            dvc.vram = props.totalGlobalMem;
        }
        active_device_id = std::clamp(use_device, 0, numgpus-1);
        s_is_init = true;
        return &s_devices[active_device_id];
    }

    vm_pool::vm_pool(const physical_device& dvc) {
        m_granularity = dvc.vmm_granularity;
        m_dvc_id = dvc.id;
    }

    vm_pool::~vm_pool() {
        if (m_dvc_address) {
            mag_cu_chk_rdv(cuMemUnmap(m_dvc_address, m_cap));
            mag_cu_chk_rdv(cuMemAddressFree(m_dvc_address, max_size));
        }
    }

    auto vm_pool::alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* {
        sz = (sz+align-1)&-align; /* Overallocate to ensure alignment. */
        if (std::size_t free {m_cap-m_needle}; sz > free) {
            std::size_t reserve {sz-free};
            reserve = (reserve+m_granularity-1)&-m_granularity; /* Align to granularity. */
            mag_cu_assert2(m_cap+reserve <= max_size);
            CUmemAllocationProp prop {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = m_dvc_id;
            CUmemGenericAllocationHandle handle {};
            mag_cu_chk_rdv(cuMemCreate(&handle, reserve, &prop, 0));
            if (!m_dvc_address) /* Reserve virtual memory */
                mag_cu_chk_rdv(cuMemAddressReserve(&m_dvc_address, max_size, 0, 0, 0));
            mag_cu_chk_rdv(cuMemMap(m_dvc_address+m_cap, reserve, 0, handle, 0));
            mag_cu_chk_rdv(cuMemRelease(handle)); /* Release handle */
            CUmemAccessDesc access {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = m_dvc_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            mag_cu_chk_rdv(cuMemSetAccess(m_dvc_address+m_cap, reserve, &access, 1)); /* Set access */
            m_cap += reserve;
        }
        mag_cu_assert2(m_dvc_address);
        auto* p {reinterpret_cast<void*>(m_dvc_address+m_needle)};
        out_sz = sz;
        m_needle += sz;
        return p;
    }

    auto vm_pool::free(void* ptr, std::size_t sz) -> void {
        m_needle -= sz;
        mag_cu_assert2(ptr == reinterpret_cast<void*>(m_dvc_address + m_needle));
    }
}

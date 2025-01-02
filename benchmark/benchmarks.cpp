// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <functional>

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "magnetron_internal.h"

template <typename F, typename... Args>
static auto run_bench(
    std::string_view name,
    std::string_view unit,
    F&& callback,
    Args&&... args
) -> void;

auto main() -> int {
    mag_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); // Create context to print CPU and system info
    mag_ctx_destroy(ctx);
    mag_set_log_mode(false);

    volatile std::size_t alloc_n = 400;

    run_bench("Fixed Malloc Free", "malloc", [=](mag_ctx_t* ctx) -> void {
        void* p = std::malloc(alloc_n);
        ankerl::nanobench::doNotOptimizeAway(p);
        std::free(p);
    });

    mag_fixed_intrusive_pool cache {};
    mag_fixed_intrusive_pool_init(&cache, alloc_n, 8, 8192);
    run_bench("Fixed Cached Pool Alloc Free", "malloc", [&cache](mag_ctx_t* ctx) -> void {
        void* p = mag_fixed_intrusive_pool_malloc(&cache);
        ankerl::nanobench::doNotOptimizeAway(p);
        mag_fixed_intrusive_pool_free(&cache, p);
    });
    mag_fixed_intrusive_pool_destroy(&cache);

    run_bench("Tensor CPU Allocation", "alloc", [](mag_ctx_t* ctx) -> void {
        mag_tensor_t* A = mag_tensor_create_1d(ctx, MAG_DTYPE_F32, 8);
        mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 8, 8);
        mag_tensor_t* C = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 8, 8, 8);
        mag_tensor_t* D = mag_tensor_create_4d(ctx, MAG_DTYPE_F32, 8, 8, 8, 8);
        mag_tensor_t* E = mag_tensor_create_5d(ctx, MAG_DTYPE_F32, 8, 8, 8, 8, 8);
        mag_tensor_t* F = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, 8, 8, 8, 8, 8, 8);
        mag_tensor_decref(A);
        mag_tensor_decref(B);
        mag_tensor_decref(C);
        mag_tensor_decref(D);
        mag_tensor_decref(E);
        mag_tensor_decref(F);
    });

    run_bench("Tensor Add", "add", [](mag_ctx_t* ctx) -> void {
        constexpr std::int64_t N = 1024;

        mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, N, N);
        mag_tensor_fill(A, 1.0f);

        mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, N, N);
        mag_tensor_fill(A, 1.0f);

        mag_tensor_t* C = mag_add(A, B);

        mag_tensor_decref(A);
        mag_tensor_decref(B);
        mag_tensor_decref(C);
    });
}

template <typename F, typename... Args>
static auto run_bench(
    std::string_view name,
    std::string_view unit,
    F&& callback,
    Args&&... args
) -> void {
    static_assert(std::is_invocable_r_v<void, F, mag_ctx_t*, Args...>);
    ankerl::nanobench::Bench bench {};
    bench.title(name.data())
        .unit(unit.data())
        .minEpochIterations(10)
        .relative(true);
    bench.performanceCounters(true);
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    bench.run(name.data(), [&] {
        std::invoke(callback, ctx, std::forward<Args>(args)...);
    });
    ankerl::nanobench::doNotOptimizeAway(ctx);
    mag_ctx_destroy(ctx);
}

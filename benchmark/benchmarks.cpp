// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "magnetron_internal.h"

static auto bench_op(ankerl::nanobench::Bench& bench, std::int64_t numel_per_dim) -> void {
    mag_device_descriptor_t desc {};
    desc.type = MAG_COMPUTE_DEVICE_TYPE_CPU;
    mag_ctx_t* ctx = mag_ctx_create2(&desc);
    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, numel_per_dim, numel_per_dim);
    mag_tensor_fill(A, 1.0f);
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, numel_per_dim, numel_per_dim);
    mag_tensor_fill(A, 3.0f);
    bench.run("Parallel Elems = " + std::to_string(A->numel), [&] {
        mag_tensor_t* R = mag_add(A, B);
        ankerl::nanobench::doNotOptimizeAway(R);
        mag_tensor_decref(R);
    });

    ankerl::nanobench::doNotOptimizeAway(ctx);
    mag_tensor_decref(B);
    mag_tensor_decref(A);
    mag_ctx_destroy(ctx);
}

auto main() -> int {
    ankerl::nanobench::Bench bench {};
    bench.title("Parallel Big Tensor")
        .unit("MM")
        .warmup(100)
        .performanceCounters(true);
    bench_op(bench, 10000);
    bench_op(bench, 1000);
    bench_op(bench, 750);
    bench_op(bench, 500);
    bench_op(bench, 250);
    bench_op(bench, 100);
    bench_op(bench, 10);
    bench_op(bench, 4);
    return 0;
}

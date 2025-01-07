// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <functional>

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "magnetron_internal.h"

auto main() -> int {
    ankerl::nanobench::Bench bench {};
    bench.title("Threaded ADD")
        .unit("ADD")
        .relative(true);
    bench.performanceCounters(true);
    mag_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_set_log_mode(false);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 8192, 8192, 8);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* B = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 8192, 8192, 8);
    mag_tensor_fill_random_uniform(B, -1.0f, 1.0f);
    bench.run("Threaded ADD", [&] {
        mag_tensor_t* R = mag_add(A, B);
        ankerl::nanobench::doNotOptimizeAway(R);
    });
    ankerl::nanobench::doNotOptimizeAway(ctx);
    mag_ctx_destroy(ctx);
}

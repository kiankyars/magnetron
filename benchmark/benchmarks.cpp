// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: linux_prepare_perf.sh to setup the system for performance measurements.

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include <thread>

auto main() -> int {
    ankerl::nanobench::Bench bench {};
    bench.title("Parallel ADD")
        .unit("ADD")
        .minEpochIterations(1024)
        .warmup(100)
        .relative(true)
        .performanceCounters(true);

    auto exec_bench = [&](std::int64_t tensor_dim, std::uint32_t threads) {

        const mag_device_descriptor_t desc = {
            .type = MAG_COMPUTE_DEVICE_TYPE_CPU,
            .thread_count = threads,
        };
        mag_ctx_t* ctx = mag_ctx_create2(&desc);
        mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, tensor_dim, tensor_dim, tensor_dim);
        mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
        mag_tensor_t* B = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, tensor_dim, tensor_dim, tensor_dim);
        mag_tensor_fill_random_uniform(B, -1.0f, 1.0f);

        bench.run("Parallel ADD on " + std::to_string(threads) + " threads", [&] {
            mag_tensor_t* R = mag_add(A, B);
            ankerl::nanobench::doNotOptimizeAway(R);
            mag_tensor_decref(R);
        });

        ankerl::nanobench::doNotOptimizeAway(ctx);
        mag_tensor_decref(B);
        mag_tensor_decref(A);
        mag_ctx_destroy(ctx);
    };

    std::uint32_t num_threads = std::max(1u, std::thread::hardware_concurrency());

    for (std::uint32_t i=1; i <= num_threads; i <<= 1) {
        exec_bench(128, i);
    }
}

// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.

#include <magnetron/magnetron.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

using namespace magnetron;

static auto bench_op(ankerl::nanobench::Bench& bench, std::int64_t numel_per_dim) -> void {
    context ctx {compute_device::cpu};
    tensor x {ctx, dtype::e8m23, numel_per_dim, numel_per_dim};
    x.fill(1.0f);
    tensor y {ctx, dtype::e8m23, numel_per_dim, numel_per_dim};
    y.fill(3.0f);

    bench.run("Parallel Elems", [&] {
        tensor r {x + y};
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    ankerl::nanobench::doNotOptimizeAway(ctx);
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

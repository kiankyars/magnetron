// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.

#include <magnetron.h>
#include <algorithm>
#include <iostream>
#include <thread>

auto main() -> int {
    auto threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "Benchmarking Parallel MM on CPU with " << threads << " threads" << std::endl;
    mag_device_descriptor_t desc {};
    desc.type = MAG_COMPUTE_DEVICE_TYPE_CPU;
    desc.thread_count = threads;
    mag_ctx_t* ctx = mag_ctx_create2(&desc);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 8192, 8192, 2);
    mag_tensor_fill_random_normal(A, 0.0f, 1.0f);
    mag_tensor_t* B = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 8192, 8192, 2);
    mag_tensor_fill_random_normal(B, 0.0f, 1.0f);
    for (volatile std::uint32_t i=0; i < 100; ++i) {
        mag_tensor_t* R = mag_add(A, B);
        mag_tensor_decref(R);
    }
    mag_tensor_decref(B);
    mag_tensor_decref(A);
    mag_ctx_destroy(ctx);
    std::cout << "Benchmark finished" << std::endl;
    return 0;
}

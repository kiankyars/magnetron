// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#ifdef MAG_ENABLE_CUDA

#include "prelude.hpp"

TEST(cuda, simple_add) {
    mag_set_log_mode(true);
    mag_ctx_t* cuda_ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA);
    mag_tensor_t* gpu_a = mag_tensor_create_6d(cuda_ctx, MAG_DTYPE_F32, 8, 8, 8, 8, 8, 8);
    mag_tensor_fill(gpu_a, 1.0f);
    mag_tensor_t* gpu_b = mag_tensor_create_6d(cuda_ctx, MAG_DTYPE_F32, 8, 8, 8, 8, 8, 8);
    mag_tensor_fill(gpu_b, 1.0f);
    mag_tensor_t* gpu_r = mag_add(gpu_a, gpu_b);
    mag_storage_buffer_t& buf {gpu_r->storage};
    std::vector<float> result {};
    result.resize(gpu_r->numel);
    (*buf.cpy_device_host)(&buf, 0, result.data(), buf.size);
    for (auto x : result) {
        EXPECT_FLOAT_EQ(x, 2.0f);
    }
    mag_ctx_destroy(cuda_ctx);
    mag_set_log_mode(false);
}

#endif

// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(ctx, create_destroy_cpu) {
    mag_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    ASSERT_NE(ctx, nullptr);
    mag_ctx_destroy(ctx);
    mag_set_log_mode(false);
}

#ifdef MAG_ENABLE_CUDA
TEST(ctx, create_destroy_cuda) {
    mag_set_log_mode(true);
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA);
    ASSERT_NE(ctx, nullptr);
    mag_ctx_destroy(ctx);
    mag_set_log_mode(false);
}
#endif

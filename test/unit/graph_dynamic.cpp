// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(graph_dynamic, simple) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_set_exec_mode(ctx, MAG_EXEC_MODE_EAGER);

    // ((W * X) + B).relu()

    auto* W = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 2, 2);
    mag_tensor_fill(W, 0.6f);
    mag_tensor_set_name(W, "W");

    auto* X = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 2, 2);
    mag_tensor_fill(X, 2.11f);
    mag_tensor_set_name(X, "X");

    auto* WX = mag_mul(W, X);
    auto* buf = static_cast<float*>(mag_tensor_data_ptr(WX));
    for (std::int64_t i=0; i < mag_tensor_numel(WX); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.6f*2.11f);
    }

    auto* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 2, 2);
    mag_tensor_fill(B, 0.1f);
    mag_tensor_set_name(B, "B");

    auto* WXB = mag_add(WX, B);
    buf = static_cast<float*>(mag_tensor_data_ptr(WXB));
    for (std::int64_t i=0; i < mag_tensor_numel(WXB); ++i) { // op must already be executed
        ASSERT_EQ(buf[i], 0.6f*2.11f + 0.1f);
    }

    mag_tensor_decref(WXB);
    mag_tensor_decref(B);
    mag_tensor_decref(WX);
    mag_tensor_decref(W);
    mag_tensor_decref(X);

    mag_ctx_destroy(ctx);
}
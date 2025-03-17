#include "prelude.hpp"

TEST(unpack_data_cpu, unpack_e8m23) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr mag_e8m23_t val {3.1415f};

    mag_tensor_t* t = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, 128, 55);
    mag_tensor_fill(t, val);

    mag_e8m23_t* data = mag_tensor_unpack_cloned_data(t);
    for (std::int64_t i=0; i < t->numel; ++i) {
        ASSERT_FLOAT_EQ(data[i], val);
    }
    mag_tensor_free_cloned_data(data);

    mag_ctx_destroy(ctx);
}

TEST(unpack_data_cpu, unpack_e5m10) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr mag_e8m23_t val {3.1415f};

    mag_tensor_t* t = mag_tensor_create_2d(ctx, MAG_DTYPE_E5M10, 128, 55);
    mag_tensor_fill(t, val);

    mag_e8m23_t* data = mag_tensor_unpack_cloned_data(t);
    for (std::int64_t i=0; i < t->numel; ++i) {
        ASSERT_NEAR(data[i], val, 1e-3);
    }
    mag_tensor_free_cloned_data(data);

    mag_ctx_destroy(ctx);
}

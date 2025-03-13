// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#ifdef MAG_ENABLE_CUDA

#include "prelude.hpp"

TEST(cuda, simple_add) {
    mag_set_log_mode(true);

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA);

    mag_tensor_t* a = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 512, 512, 16);
    mag_tensor_fill(a, 1.0f);

    mag_tensor_t* b = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 512, 512, 16);
    mag_tensor_fill(b, 1.0f);

    printf("Computing...\n");
    clock_t begin = clock();
    mag_tensor_t* result = mag_add(a, b); /* Compute result = a + b */
    clock_t end = clock();
    double secs = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Computed in %f s\n", secs);

    mag_storage_buffer_t& buf{result->storage};
    std::vector<float> data{};
    data.resize(result->numel);
    (*buf.cpy_device_host)(&buf, 0, data.data(), buf.size);
    for (auto x : data) {
        EXPECT_FLOAT_EQ(x, 2.0f);
    }
    mag_tensor_decref(result);
    mag_tensor_decref(b);
    mag_tensor_decref(a);
    mag_ctx_destroy(ctx);
    mag_set_log_mode(false);
}

#endif

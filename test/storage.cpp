// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <filesystem>

TEST(storage, load_store) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* A = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, 10, 4, 2, 5, 2, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);

    if (std::filesystem::exists("test_data/test.magnetron"))
        std::filesystem::remove("test_data/test.magnetron");
    mag_tensor_save(A, "test_data/test.magnetron");
    ASSERT_TRUE(std::filesystem::exists("test_data/test.magnetron"));
    mag_tensor_t* B = mag_tensor_load(ctx, "test_data/test.magnetron");
    ASSERT_EQ(mag_tensor_dtype(B), MAG_DTYPE_F32);
    ASSERT_EQ(mag_tensor_rank(B), 6);
    ASSERT_EQ(mag_tensor_shape(B)[0], 10);
    ASSERT_EQ(mag_tensor_shape(B)[1], 4);
    ASSERT_EQ(mag_tensor_shape(B)[2], 2);
    ASSERT_EQ(mag_tensor_shape(B)[3], 5);
    ASSERT_EQ(mag_tensor_shape(B)[4], 2);
    ASSERT_EQ(mag_tensor_shape(B)[5], 2);
    ASSERT_EQ(mag_tensor_data_size(B), 10 * 4 * 2 * 5 * 2 * 2 * sizeof(float));
    //ASSERT_TRUE(mag_tensor_eq(A, B)); todo

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/test.magnetron"));
}

TEST(storage, load_store_image) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* img = mag_tensor_load_image(ctx, "test_data/car.jpg", MAG_COLOR_CHANNELS_RGB, 0, 0);
    if (std::filesystem::exists("test_data/car.magnetron"))
        std::filesystem::remove("test_data/car.magnetron");
    mag_tensor_save(img, "test_data/car.magnetron");
    ASSERT_TRUE(std::filesystem::exists("test_data/car.magnetron"));
    mag_tensor_t* B = mag_tensor_load(ctx, "test_data/car.magnetron");
    ASSERT_EQ(mag_tensor_dtype(B), MAG_DTYPE_F32);
    ASSERT_EQ(mag_tensor_shape(B)[2], 1536);
    ASSERT_EQ(mag_tensor_shape(B)[1], 2048);
    ASSERT_EQ(mag_tensor_shape(B)[0], 3);
    //ASSERT_TRUE(mag_tensor_eq(img, B)); todo
    //mag_tensor_save_image(B, "test_data/car_from_magnetron.jpg");
    mag_tensor_decref(img);
    mag_tensor_decref(B);
    mag_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/car.magnetron"));
}

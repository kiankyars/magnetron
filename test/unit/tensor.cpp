// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>

TEST(mag_tensor_t, init_1d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 10);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 1);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 1);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10);
    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_TRUE(mag_tensor_is_vector(tensor));
    ASSERT_TRUE(mag_tensor_is_matrix(tensor));
    ASSERT_TRUE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, init_2d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, 10, 4);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 2);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * 4 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10 * 4);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10 * 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10 * 4);
    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_FALSE(mag_tensor_is_vector(tensor));
    ASSERT_TRUE(mag_tensor_is_matrix(tensor));
    ASSERT_TRUE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, init_3d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 10, 4, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 3);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * 4 * 2 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10 * 4 * 2);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 8);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10 * 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10 * 4 * 2);
    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_FALSE(mag_tensor_is_vector(tensor));
    ASSERT_FALSE(mag_tensor_is_matrix(tensor));
    ASSERT_TRUE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, init_4d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10 * 4 * 2 * 5);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 40);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10 * 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10 * 4 * 2);
    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_FALSE(mag_tensor_is_vector(tensor));
    ASSERT_FALSE(mag_tensor_is_matrix(tensor));
    ASSERT_FALSE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, init_5d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_5d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5, 3);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 5);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(mag_tensor_shape(tensor)[4], 3);
    ASSERT_EQ(mag_tensor_shape(tensor)[5], 1);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * 4 * 2 * 5 * 3 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10 * 4 * 2 * 5 * 3);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 40 * 3);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10 * 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10 * 4 * 2);
    ASSERT_EQ(mag_tensor_strides(tensor)[4], 10 * 4 * 2 * 5);
    ASSERT_EQ(mag_tensor_strides(tensor)[5], 10 * 4 * 2 * 5 * 3);

    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_FALSE(mag_tensor_is_vector(tensor));
    ASSERT_FALSE(mag_tensor_is_matrix(tensor));
    ASSERT_FALSE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, init_6d) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5, 3, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(mag_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(mag_tensor_dtype(tensor), MAG_DTYPE_E8M23);
    ASSERT_EQ(mag_tensor_rank(tensor), 6);
    ASSERT_EQ(mag_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(mag_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(mag_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(mag_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(mag_tensor_shape(tensor)[4], 3);
    ASSERT_EQ(mag_tensor_shape(tensor)[5], 2);
    ASSERT_EQ(mag_tensor_data_size(tensor), 10 * 4 * 2 * 5 * 3 * 2 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(tensor), 10 * 4 * 2 * 5 * 3 * 2);
    ASSERT_EQ(mag_tensor_num_cols(tensor), 10);
    ASSERT_EQ(mag_tensor_num_rows(tensor), 40 * 3 * 2);
    ASSERT_EQ(mag_tensor_strides(tensor)[0], 1);
    ASSERT_EQ(mag_tensor_strides(tensor)[1], 10);
    ASSERT_EQ(mag_tensor_strides(tensor)[2], 10 * 4);
    ASSERT_EQ(mag_tensor_strides(tensor)[3], 10 * 4 * 2);
    ASSERT_EQ(mag_tensor_strides(tensor)[4], 10 * 4 * 2 * 5);
    ASSERT_EQ(mag_tensor_strides(tensor)[5], 10 * 4 * 2 * 5 * 3);

    ASSERT_FALSE(mag_tensor_is_scalar(tensor));
    ASSERT_FALSE(mag_tensor_is_vector(tensor));
    ASSERT_FALSE(mag_tensor_is_matrix(tensor));
    ASSERT_FALSE(mag_tensor_is_volume(tensor));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, print) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 2, 2, 2, 2);
    mag_tensor_fill_random_uniform(tensor, 0.0f, 1.0f);
    mag_tensor_print(tensor, false, true);

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, name) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 2, 2, 2, 2);
    mag_tensor_set_name(tensor, "Gradient Backup");
    ASSERT_STREQ(mag_tensor_get_name(tensor), "Gradient Backup");

    mag_tensor_decref(tensor);
    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, deep_clone) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5);
    mag_tensor_fill_random_uniform(origin, -1.0f, 1.0f);
    mag_tensor_t* clone = mag_clone(origin);
    ASSERT_NE(origin, clone);
    ASSERT_EQ(mag_tensor_rank(origin), mag_tensor_rank(clone));
    ASSERT_EQ(mag_tensor_shape(origin)[0], mag_tensor_shape(clone)[0]);
    ASSERT_EQ(mag_tensor_shape(origin)[1], mag_tensor_shape(clone)[1]);
    ASSERT_EQ(mag_tensor_shape(origin)[2], mag_tensor_shape(clone)[2]);
    ASSERT_EQ(mag_tensor_shape(origin)[3], mag_tensor_shape(clone)[3]);
    ASSERT_EQ(mag_tensor_data_size(origin), mag_tensor_data_size(clone));
    ASSERT_EQ(mag_tensor_numel(origin), mag_tensor_numel(clone));
    ASSERT_EQ(mag_tensor_num_cols(origin), mag_tensor_num_cols(clone));
    ASSERT_EQ(mag_tensor_num_rows(origin), mag_tensor_num_rows(clone));
    ASSERT_EQ(mag_tensor_strides(origin)[0], mag_tensor_strides(clone)[0]);
    ASSERT_EQ(mag_tensor_strides(origin)[1], mag_tensor_strides(clone)[1]);
    ASSERT_EQ(mag_tensor_strides(origin)[2], mag_tensor_strides(clone)[2]);
    ASSERT_EQ(mag_tensor_strides(origin)[3], mag_tensor_strides(clone)[3]);
    ASSERT_TRUE(mag_tensor_is_shape_eq(origin, clone));
    ASSERT_TRUE(mag_tensor_are_strides_eq(origin, clone));

    const void* a = mag_tensor_data_ptr(origin);
    const void* b = mag_tensor_data_ptr(clone);
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, mag_tensor_data_size(origin)));

    mag_tensor_decref(origin);
    mag_tensor_decref(clone);
    mag_ctx_destroy(ctx);
}

#if 0 // TODO: Implement mag_tensor_eq
TEST(mag_tensor_t, equals) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5);
    mag_tensor_fill_random_uniform(origin, -1.0f, 1.0f);
    mag_tensor_t* clone = mag_clone(origin);
    mag_tensor_t* clone2 = mag_clone(origin);
    mag_tensor_fill_random_uniform(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(mag_tensor_eq(origin, clone));
    ASSERT_FALSE(mag_tensor_eq(origin, clone2));
    ASSERT_FALSE(mag_tensor_eq(clone, clone2));

    mag_tensor_decref(origin);
    mag_tensor_decref(clone);
    mag_tensor_decref(clone2);

    mag_ctx_destroy(ctx);
}
#endif

TEST(mag_tensor_t, buffer_linearly) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 1, 2, 3, 4);
    mag_tensor_fill(origin, 0.0f);
    auto* buf = static_cast<float*>(mag_tensor_data_ptr(origin));
    buf[0] = 1.0f;
    buf[mag_tensor_numel(origin) - 1] = -1.0f;
    for (int64_t i = 0; i < mag_tensor_numel(origin); ++i) {
        std::cout << buf[i] << " ";
    }
    std::cout << std::endl;

    mag_tensor_decref(origin);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, view) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5);
    mag_tensor_fill(origin, 2.0f);
    int64_t slice_dims[] = {10, 4, 2, 5};
    mag_tensor_t* slice1 = mag_view(origin);
    ASSERT_EQ(mag_tensor_data_ptr(slice1), mag_tensor_data_ptr(origin));
    ASSERT_EQ(mag_tensor_data_size(slice1), mag_tensor_data_size(origin));
    ASSERT_EQ(mag_tensor_numel(slice1), mag_tensor_numel(origin));
    auto* buf = static_cast<float*>(mag_tensor_data_ptr(slice1));
    for (int64_t i = 0; i < mag_tensor_numel(slice1); ++i) {
        ASSERT_FLOAT_EQ(buf[i], 2.0f);
    }
    mag_tensor_t* slice2 = mag_view(origin);
    ASSERT_EQ(mag_tensor_data_ptr(slice2), mag_tensor_data_ptr(origin));
    ASSERT_EQ(mag_tensor_data_size(slice2), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(mag_tensor_numel(slice2), 10 * 4 * 2 * 5);
    auto* buf_slice2 = static_cast<float*>(mag_tensor_data_ptr(slice2));
    for (int64_t i = 0; i < mag_tensor_numel(slice2); ++i) {
        ASSERT_FLOAT_EQ(buf_slice2[i], 2.0f);
    }

    mag_tensor_decref(origin);
    mag_tensor_decref(slice1);
    mag_tensor_decref(slice2);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, transpose) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, 4, 1);
    mag_tensor_fill_random_uniform(origin, -1.0f, 1.0f);
    mag_tensor_t* transposed = mag_transpose(origin);
    ASSERT_FALSE(mag_tensor_is_transposed(origin));
    ASSERT_TRUE(mag_tensor_is_transposed(transposed));
    ASSERT_EQ(mag_tensor_shape(origin)[0], mag_tensor_shape(transposed)[1]);
    ASSERT_EQ(mag_tensor_shape(origin)[1], mag_tensor_shape(transposed)[0]);
    ASSERT_EQ(mag_tensor_data_size(origin), mag_tensor_data_size(transposed));
    ASSERT_EQ(mag_tensor_numel(origin), mag_tensor_numel(transposed));
    ASSERT_EQ(mag_tensor_num_cols(origin), mag_tensor_num_rows(transposed));
    ASSERT_EQ(mag_tensor_num_rows(origin), mag_tensor_num_cols(transposed));
    ASSERT_TRUE(mag_tensor_is_contiguous(origin));
    ASSERT_FALSE(mag_tensor_is_contiguous(transposed));

    mag_tensor_decref(origin);
    mag_tensor_decref(transposed);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, permute) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* origin = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, 4, 1);
    mag_tensor_fill_random_uniform(origin, -1.0f, 1.0f);
    mag_tensor_t* permuted = mag_permute(origin, 5, 4, 3, 2, 1, 0);
    ASSERT_FALSE(mag_tensor_is_transposed(origin));
    ASSERT_FALSE(mag_tensor_is_transposed(permuted));
    ASSERT_FALSE(mag_tensor_is_permuted(origin));
    ASSERT_TRUE(mag_tensor_is_permuted(permuted));
    ASSERT_EQ(mag_tensor_shape(origin)[0], mag_tensor_shape(permuted)[5]);
    ASSERT_EQ(mag_tensor_shape(origin)[1], mag_tensor_shape(permuted)[4]);
    ASSERT_EQ(mag_tensor_shape(origin)[2], mag_tensor_shape(permuted)[3]);
    ASSERT_EQ(mag_tensor_shape(origin)[3], mag_tensor_shape(permuted)[2]);
    ASSERT_EQ(mag_tensor_shape(origin)[4], mag_tensor_shape(permuted)[1]);
    ASSERT_EQ(mag_tensor_shape(origin)[5], mag_tensor_shape(permuted)[0]);
    ASSERT_EQ(mag_tensor_data_size(origin), mag_tensor_data_size(permuted));
    ASSERT_EQ(mag_tensor_numel(origin), mag_tensor_numel(permuted));
    ASSERT_EQ(mag_tensor_num_cols(origin), mag_tensor_num_rows(permuted));
    ASSERT_EQ(mag_tensor_num_rows(origin), mag_tensor_num_cols(permuted));
    ASSERT_TRUE(mag_tensor_is_contiguous(origin));
    ASSERT_FALSE(mag_tensor_is_contiguous(permuted));

    mag_tensor_decref(origin);
    mag_tensor_decref(permuted);

    mag_ctx_destroy(ctx);
}

#if 0 // TODO: Implement mag_tensor_is_close
TEST(mag_tensor_t, isclose) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* origin = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 10, 4, 2, 5);
    mag_tensor_fill_random_uniform(origin, -1.0f, 1.0f);
    mag_tensor_t* clone = mag_clone(origin);
    mag_tensor_t* clone2 = mag_clone(origin);
    mag_tensor_fill_random_uniform(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(mag_tensor_is_close(origin, clone, FLT_EPSILON, nullptr));
    ASSERT_FALSE(mag_tensor_is_close(origin, clone2, FLT_EPSILON, nullptr));
    ASSERT_FALSE(mag_tensor_is_close(clone, clone2, FLT_EPSILON, nullptr));
    mag_tensor_fill(clone, 0.0f);
    mag_tensor_fill(clone2, 0.0f);
    double percent = 0.0;
    ASSERT_TRUE(mag_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 100.0);
    mag_tensor_fill(clone2, 1.0f);
    ASSERT_FALSE(mag_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 0.0);

    mag_tensor_decref(origin);
    mag_tensor_decref(clone);
    mag_tensor_decref(clone2);

    mag_ctx_destroy(ctx);
}
#endif

TEST(mag_tensor_t, copy_buffer_from) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    std::array<float, 2 * 2 * 2 * 2> buf{};
    for (auto& x : buf)
        x = 2.5f;

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 2, 2, 2, 2);
    ASSERT_EQ(mag_tensor_data_size(tensor), sizeof(buf));
    ASSERT_EQ(mag_tensor_numel(tensor), buf.size());
    mag_tensor_copy_buffer_from(tensor, buf.data(), sizeof(buf));

    const void* a = mag_tensor_data_ptr(tensor);
    const void* b = buf.data();
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, mag_tensor_data_size(tensor)));

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, fill) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 8, 10, 11, 2);
    float* buf = static_cast<float*>(mag_tensor_data_ptr(tensor));

    mag_tensor_fill(tensor, 0.0f);
    for (std::int64_t i = 0; i < mag_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 0.0f);
    }

    mag_tensor_fill(tensor, 2.5f);
    for (std::int64_t i = 0; i < mag_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 2.5f);
    }

    mag_tensor_fill(tensor, -1.0f);
    for (std::int64_t i = 0; i < mag_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], -1.0f);
    }

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, random_uniform_pcg) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_set_prng_algorithm(ctx, MAG_PRNG_PCG, reinterpret_cast<std::uint64_t>(this));

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 32, 32, 32, 32);
    mag_tensor_fill_random_uniform(tensor, rmin, rmax);

    auto* buf = static_cast<float*>(mag_tensor_data_ptr(tensor));

    for (int64_t i = 0; i < mag_tensor_numel(tensor); ++i) {
        float x = buf[i];
        ASSERT_GT(x, rmin);
        ASSERT_LT(x, rmax);
    }

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, random_uniform_mersenne) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_set_prng_algorithm(ctx, MAG_PRNG_MERSENNE_TWISTER, reinterpret_cast<std::uint64_t>(this));

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 32, 32, 32, 32);
    mag_tensor_fill_random_uniform(tensor, rmin, rmax);

    auto* buf = static_cast<float*>(mag_tensor_data_ptr(tensor));

    for (int64_t i = 0; i < mag_tensor_numel(tensor); ++i) {
        float x = buf[i];
        ASSERT_GT(x, rmin);
        ASSERT_LT(x, rmax);
    }

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, random_normal_mersenne) {
    constexpr float mean = 0.0;
    constexpr float stddev = 1.0;

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_set_prng_algorithm(ctx, MAG_PRNG_MERSENNE_TWISTER, reinterpret_cast<std::uint64_t>(this));

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 32, 32, 32, 32);
    mag_tensor_fill_random_normal(tensor, mean, stddev);

    auto* buf = static_cast<float*>(mag_tensor_data_ptr(tensor));

    double sum = 0.0;
    double sum_sq = 0.0;
    const int64_t num_elements = mag_tensor_numel(tensor);
    for (int64_t i = 0; i < num_elements; ++i) {
        float x = buf[i];
        sum += x;
        sum_sq += x * x;
    }
    double r_mean = sum / num_elements;
    double r_variance = (sum_sq / num_elements) - (r_mean * r_mean);
    double r_stddev = std::sqrt(r_variance);

    ASSERT_NEAR(r_mean, mean, 0.01);
    ASSERT_NEAR(r_stddev, stddev, 0.01);

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, random_normal_pcg) {
    constexpr float mean = 0.0;
    constexpr float stddev = 1.0;

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_set_prng_algorithm(ctx, MAG_PRNG_PCG, reinterpret_cast<std::uint64_t>(this));

    mag_tensor_t* tensor = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 32, 32, 32, 32);
    mag_tensor_fill_random_normal(tensor, mean, stddev);

    auto* buf = static_cast<float*>(mag_tensor_data_ptr(tensor));

    double sum = 0.0;
    double sum_sq = 0.0;
    const int64_t num_elements = mag_tensor_numel(tensor);
    for (int64_t i = 0; i < num_elements; ++i) {
        float x = buf[i];
        sum += x;
        sum_sq += x * x;
    }
    double r_mean = sum / num_elements;
    double r_variance = (sum_sq / num_elements) - (r_mean * r_mean);
    double r_stddev = std::sqrt(r_variance);

    ASSERT_NEAR(r_mean, mean, 0.01);
    ASSERT_NEAR(r_stddev, stddev, 0.01);

    mag_tensor_decref(tensor);

    mag_ctx_destroy(ctx);
}

TEST(mag_tensor_t, rc_init_strong) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* a = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 10);
    ASSERT_EQ(a->rcb.rc_strong, 1);
    ASSERT_EQ(a->rcb.rc_weak, 0);

    ASSERT_TRUE(mag_tensor_decref(a));

    mag_ctx_destroy(ctx);
}

#if 0 // todo
TEST(mag_tensor_t, rc_ref_view_chain) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* a = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 10);
    ASSERT_EQ(a->rcb.rc_strong, 1);
    ASSERT_EQ(a->rcb.rc_weak, 0);

    mag_tensor_t* b = mag_view(a); // okay so view references original tensor
    ASSERT_EQ(b->rcb.rc_strong, 1);
    ASSERT_EQ(b->rcb.rc_weak, 0);
    ASSERT_EQ(a->rcb.rc_strong, 2);
    ASSERT_EQ(a->rcb.rc_weak, 0);

    ASSERT_TRUE(mag_tensor_decref(b)); // b is freed
    ASSERT_EQ(a->rcb.rc_strong, 1); // b refcount is gone
    ASSERT_EQ(a->rcb.rc_weak, 0);
    ASSERT_TRUE(mag_tensor_decref(a));

    mag_ctx_destroy(ctx);
}
#endif

/*
TEST(mag_tensor_t, rc_ref_leak) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* a = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 10);
    ASSERT_EQ(a->rcb.rc_strong, 1);
    ASSERT_EQ(a->rcb.rc_weak, 0);

    mag_tensor_t* b = mag_tensor_emit_op_va(ctx, MAG_OP_VIEW, a); // okay so view references original tensor
    ASSERT_EQ(b->rcb.rc_strong, 1);
    ASSERT_EQ(b->rcb.rc_weak, 0);
    ASSERT_EQ(a->rcb.rc_strong, 2); // a is now referenced by b
    ASSERT_EQ(a->rcb.rc_weak, 0);
    ASSERT_FALSE(mag_tensor_decref(a)); // a is still referenced by b

    ASSERT_TRUE(mag_tensor_decref(a)); // no more references here

    mag_ctx_destroy(ctx);
}
*/

TEST(mag_tensor_t, rc_ref_inplace_op) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* x = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 5);
    mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
    ASSERT_EQ(x->rcb.rc_strong, 1);
    mag_tensor_t* r = mag_abs_(x);
    ASSERT_EQ(x->rcb.rc_strong, 2);
    ASSERT_EQ(r->rcb.rc_strong, 1);
    mag_tensor_set_name(r, "result");
    mag_tensor_set_name(x, "X");
    mag_tensor_decref(r);
    mag_tensor_decref(x);
    mag_ctx_destroy(ctx);
}

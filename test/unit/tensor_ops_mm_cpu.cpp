// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// For some tests, we use a larger absolute error than machine epsilon,
// because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar high-precision lambdas.

#include "prelude.hpp"

static auto mm_naive(
    const float* A,
    const float* B,
    float* C,
    const std::int64_t M,
    const std::int64_t N,
    const std::int64_t K
) -> void {
    for (std::int64_t i = 0; i < M * N; ++i)
        C[i] = 0.0f;
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t k = 0; k < K; ++k) {
            float a_ik = A[i*K + k];
            for (std::int64_t j = 0; j < N; ++j) {
                C[i*N + j] += a_ik * B[k*N + j];
            }
        }
    }
}

TEST(compute_cpu, mm_naive_verify) {
    static constexpr float A[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr float B[2] = {0.5f, -1.0f};
    float C[3];
    mm_naive(A, B, C, 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);
}

TEST(compute_cpu, mm_square_2x2) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr std::size_t M = 2, N = 2;
    static constexpr float A_data[M][N] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_data[M][N] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(A, A_data, sizeof(A_data));
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(B, B_data, sizeof(B_data));
    mag_tensor_t* R = mag_matmul(A, B);
    ASSERT_EQ(R->rank, 2);
    ASSERT_EQ(R->shape[0], 2);
    ASSERT_EQ(R->shape[1], 2);

    static constexpr float expected[M][N] = {
        0.3717081,   0.7322086,
        -2.3101263, -0.28030172
    };
    auto* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t i=0; i < M; ++i)
        for (std::int64_t j=0; j < N; ++j)
            ASSERT_FLOAT_EQ(buf[i*M + j], expected[i][j]);

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}


TEST(compute_cpu, mm_square_2x2_transpose_x) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr std::size_t M = 2, N = 2;
    static constexpr float A_data[M][N] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_data[M][N] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(A, A_data, sizeof(A_data));
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(B, B_data, sizeof(B_data));
    mag_tensor_t* R = mag_matmul(mag_clone(mag_transpose(A)), B);
    ASSERT_EQ(R->rank, 2);
    ASSERT_EQ(R->shape[0], 2);
    ASSERT_EQ(R->shape[1], 2);

    static constexpr float expected[M*N] = {
        -2.8671103f, -0.58510107f,
        -0.3463393f, -0.6243037f
    };
    auto* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t i=0; i < M*N; ++i)
        ASSERT_FLOAT_EQ(buf[i], expected[i]);

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

#if 0
TEST(compute_cpu, mm_square_2x2_transpose_y) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr std::size_t M = 2, N = 2;
    static constexpr float A_data[M][N] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_data[M][N] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(A, A_data, sizeof(A_data));
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, N);
    mag_tensor_copy_buffer_from(B, B_data, sizeof(B_data));
    mag_tensor_t* R = mag_matmul(mag_transpose(A), B);
    ASSERT_EQ(R->rank, 2);
    ASSERT_EQ(R->shape[0], 2);
    ASSERT_EQ(R->shape[1], 2);

    static constexpr float expected[M*N] = {
        -1.1440332f, -1.0894998f,
        -1.0079648f, -2.3473806f
    };
    auto* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t i=0; i < M*N; ++i)
        ASSERT_FLOAT_EQ(buf[i], expected[i]);

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}
#endif

TEST(compute_cpu, mm_rect_2x3_3x4) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    constexpr std::size_t M = 2, K = 3, N = 4;

    static constexpr float A_data[M][K] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    static constexpr float B_data[K][N] = {
        {1.0f,  2.0f,  3.0f,  4.0f},
        {5.0f,  6.0f,  7.0f,  8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f}
    };

    static constexpr float expected[M][N] = {
        { 38.0f,  44.0f,  50.0f,  56.0f},
        { 83.0f,  98.0f, 113.0f, 128.0f}
    };

    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, K);
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, K, N);
    mag_tensor_copy_buffer_from(A, A_data, sizeof(A_data));
    mag_tensor_copy_buffer_from(B, B_data, sizeof(B_data));

    mag_tensor_t* R = mag_matmul(A, B);

    ASSERT_EQ(R->rank, 2);
    ASSERT_EQ(R->shape[0], (int64_t)M);
    ASSERT_EQ(R->shape[1], (int64_t)N);

    const float* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t i = 0; i < (std::int64_t)M; ++i) {
        for (std::int64_t j = 0; j < (std::int64_t)N; ++j) {
            ASSERT_FLOAT_EQ(buf[i * N + j], expected[i][j]);
        }
    }

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, mm_matrix_vector_2x3) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr std::size_t M = 2, K = 3;

    static constexpr float A_data[M][K] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    static constexpr float v_data[K] = {7.0f, 8.0f, 9.0f};
    static constexpr float expected[M] = {50.0f, 122.0f};

    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, M, K);
    mag_tensor_t* v = mag_tensor_create_1d(ctx, MAG_DTYPE_F32, K);
    mag_tensor_copy_buffer_from(A, A_data, sizeof(A_data));
    mag_tensor_copy_buffer_from(v, v_data, sizeof(v_data));

    mag_tensor_t* R = mag_matmul(A, v);

    ASSERT_EQ(R->rank, 1);
    ASSERT_EQ(R->shape[0], (int64_t)M);

    const float* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t i = 0; i < (std::int64_t)M; ++i) {
        ASSERT_FLOAT_EQ(buf[i], expected[i]);
    }

    mag_tensor_decref(A);
    mag_tensor_decref(v);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, mm_vector_matrix_3x2) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    constexpr std::size_t K = 3, N = 2;

    static constexpr float v_data[1][K] = {
        {1.0f, 2.0f, 3.0f}
    };

    static constexpr float B_data[K][N] = {
        {4.0f, 5.0f},
        {6.0f, 7.0f},
        {8.0f, 9.0f}
    };

    static constexpr float expected[1][N] = {
        {40.0f, 46.0f}
    };

    mag_tensor_t* v = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 1, K);
    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, K, N);
    mag_tensor_copy_buffer_from(v, v_data, sizeof(v_data));
    mag_tensor_copy_buffer_from(B, B_data, sizeof(B_data));

    mag_tensor_t* R = mag_matmul(v, B);

    ASSERT_EQ(R->rank, 2);
    ASSERT_EQ(R->shape[0], 1);
    ASSERT_EQ(R->shape[1], (int64_t)N);

    const float* buf = static_cast<const float*>(mag_tensor_data_ptr(R));
    for (std::int64_t j = 0; j < (std::int64_t)N; ++j) {
        ASSERT_FLOAT_EQ(buf[j], expected[0][j]);
    }

    mag_tensor_decref(v);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

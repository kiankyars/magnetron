// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// For some tests, we use a larger absolute error than machine epsilon,
// because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar high-precision lambdas.

#include "prelude.hpp"
#include <algorithm>
#include <cmath>

static constexpr std::int64_t k_lim_same_shape = 4;
static constexpr std::int64_t k_lim_broadcast = 2;

#define impl_test_unary_op(name, eps, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            \
            mag_tensor_t* r = mag_##op(x); \
            \
            const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r)); \
            ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), (eps)); \
            } \
            mag_tensor_decref(r); \
            mag_tensor_decref(x); \
        } \
        \
        mag_ctx_destroy(ctx); \
    } \
    TEST(compute_cpu, name##_same_shape_inplace) { \
            mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
            \
            for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
            for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
            for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
            for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
            for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
            for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
                std::vector<float> x_origin {}; \
                mag_tensor_buf_f32_to_vec(x, x_origin); \
                \
                mag_tensor_t* r = mag_##op##_(x); \
                mag_tensor_set_name(r, "result");  \
                \
                const auto* b_x = x_origin.data(); \
                mag_tensor_set_name(x, "X"); \
                const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r)); \
                ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r)); \
                ASSERT_EQ(mag_tensor_data_ptr(x), mag_tensor_data_ptr(r)); \
                for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                    ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), (eps)); \
                } \
                mag_tensor_decref(r); \
                mag_tensor_decref(x); \
            } \
            \
            mag_ctx_destroy(ctx); \
        }

impl_test_unary_op(abs, 1e-6, abs, [](float x) -> float {
    return std::abs(x);
})

TEST(compute_cpu, neg_same_shape) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0)
    for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1)
    for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2)
    for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3)
    for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4)
    for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) {
        mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5);
        mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
        mag_tensor_t* r = mag_neg(x);
        const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));
        const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));
        ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));
        for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) {
            ASSERT_EQ(b_r[i], -b_x[i]);
        }
        mag_tensor_decref(x);
        mag_tensor_decref(r);
    }
    mag_ctx_destroy(ctx);
}

impl_test_unary_op(log, 1e-6, log, [](float x) -> float {
    return std::log(x);
})

impl_test_unary_op(sqr, 1e-9, sqr, [](float x) -> float {
    return x*x;
})

impl_test_unary_op(sqrt, 1e-9, sqrt, [](float x) -> float {
    return std::sqrt(x);
})

impl_test_unary_op(sin, 1e-6, sin, [](float x) -> float {
    return std::sin(x);
})

#if 0 // TODO fix
impl_test_unary_op(cos, 1e-6, cos, [](float x) -> float {
    return std::cos(x);
})
#endif

impl_test_unary_op(step, 1e-9, step, [](float x) -> float {
    return x >= 0.0f ? 1.0f : 0.0f;
})

impl_test_unary_op(softmax, 1e-6, softmax, [](float x) -> float {
    return std::exp(x);
})
impl_test_unary_op(softmax_dv, 1e-6, softmax_dv, [](float x) -> float {
    return std::exp(x);
})

impl_test_unary_op(sigmoid, 1e-6, sigmoid, [](float x) -> float {
    return 1.0f / (1.0f + std::exp(-x));
})
impl_test_unary_op(sigmoid_dv, 1e-6, sigmoid_dv, [](float x) -> float {
    return x * (1.0f - x);
})

impl_test_unary_op(hard_sigmoid, 1e-6, hard_sigmoid, [](float x) -> float {
    return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
})
//impl_test_unary_op(hard_sigmoid_dv, HARD_SIGMOID_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(silu, 1e-6, silu, [](float x) -> float {
    return x / (1.0f + std::exp(-x));
})
//impl_test_unary_op(silu_dv, SILU_DV, [](float x) -> float {
//    return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
//})

impl_test_unary_op(tanh, 1e-3, tanh, [](float x) -> float {
    return std::tanh(x);
})
impl_test_unary_op(tanh_dv, 1e-9, tanh_dv, [](float x) -> float {
    return 1.0f / (std::cosh(x)*std::cosh(x));
})

impl_test_unary_op(relu, 1e-9, relu, [](float x) -> float {
    return std::max(x, 0.0f);
})
impl_test_unary_op(relu_dv, 1e-9, relu_dv, [](float x) -> float {
    return x <= 0.0f ? 0.0f : 1.0f;
})

impl_test_unary_op(gelu, 1e-3, gelu, [](float x) -> float {
    return 0.5f*x*(1.0f + std::tanh(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
})
//impl_test_unary_op(gelu_dv, GELU_DV, [](float x) -> float {
//    return x <= 0.0f ? 0.0f : 1.0f;
//})

#undef impl_test_unary_op

#define impl_test_binary_op(name, op, scalar_op) \
    TEST(compute_cpu, name##_same_shape) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_t* y = mag_clone(x); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            mag_tensor_fill_random_uniform(y, -5.0f, 5.0f); \
            \
            mag_tensor_t* r = mag_##op(x, y); \
                                                 \
            \
            const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_y = static_cast<const float*>(mag_tensor_data_ptr(y)); \
            const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(y)); \
            ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(y)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op b_y[i]); \
            } \
                mag_tensor_decref(r); \
                mag_tensor_decref(x); \
                mag_tensor_decref(y); \
            } \
        \
        mag_ctx_destroy(ctx); \
    } \
     \
    TEST(compute_cpu, name##_scalar_broadcast) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        for (std::int64_t factor=2; factor <= 4; ++factor) \
        for (std::int64_t i0=1; i0 <= k_lim_broadcast; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_broadcast; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_broadcast; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_broadcast; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_broadcast; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_broadcast; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0*factor, i1*factor, i2*factor, i3*factor, i4*factor, i5*factor); \
            mag_tensor_t* y = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            mag_tensor_fill(y, 2.2f); \
            \
            mag_tensor_t* r = mag_##op(x, y); \
            \
            const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(x)); \
            ASSERT_NE(mag_tensor_numel(x), mag_tensor_numel(y)); \
            ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op 2.2f); \
            } \
            mag_tensor_decref(r); \
            mag_tensor_decref(x); \
            mag_tensor_decref(y); \
        } \
        \
        mag_ctx_destroy(ctx); \
    } \
    TEST(compute_cpu, name##_same_shape_inplace) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_t* y = mag_clone(x); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            mag_tensor_fill_random_uniform(y, -5.0f, 5.0f); \
            std::vector<float> x_origin {}; \
            mag_tensor_buf_f32_to_vec(x, x_origin); \
            \
            mag_tensor_t* r = mag_##op##_(x, y); \
                                                 \
            \
            const auto* b_x = x_origin.data(); \
            const auto* b_xx = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_y  = static_cast<const float*>(mag_tensor_data_ptr(y)); \
            const auto* b_r  = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(y)); \
            ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(y)); \
            ASSERT_EQ(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_xx[i]); \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op b_y[i]); \
            } \
            mag_tensor_decref(r); \
            mag_tensor_decref(x); \
            mag_tensor_decref(y); \
        } \
        \
        mag_ctx_destroy(ctx); \
    } \
     \
    TEST(compute_cpu, name##_scalar_broadcast_inplace) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        for (std::int64_t factor=2; factor <= 4; ++factor) \
        for (std::int64_t i0=1; i0 <= k_lim_broadcast; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_broadcast; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_broadcast; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_broadcast; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_broadcast; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_broadcast; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0*factor, i1*factor, i2*factor, i3*factor, i4*factor, i5*factor); \
            mag_tensor_t* y = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            mag_tensor_fill(y, 2.2f); \
            \
            std::vector<float> x_origin {}; \
            mag_tensor_buf_f32_to_vec(x, x_origin); \
            \
            mag_tensor_t* r = mag_##op##_(x, y); \
            \
            const auto* b_x  = x_origin.data(); \
            const auto* b_xx = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_r  = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(x)); \
            ASSERT_NE(mag_tensor_numel(x), mag_tensor_numel(y)); \
            ASSERT_EQ(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_xx[i]);  \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op 2.2f); \
            } \
            mag_tensor_decref(r); \
            mag_tensor_decref(x); \
            mag_tensor_decref(y); \
        } \
        \
        mag_ctx_destroy(ctx); \
    } \
    TEST(compute_cpu, name##_scalar) { \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU); \
        for (std::int64_t i0=1; i0 <= k_lim_same_shape; ++i0) \
        for (std::int64_t i1=1; i1 <= k_lim_same_shape; ++i1) \
        for (std::int64_t i2=1; i2 <= k_lim_same_shape; ++i2) \
        for (std::int64_t i3=1; i3 <= k_lim_same_shape; ++i3) \
        for (std::int64_t i4=1; i4 <= k_lim_same_shape; ++i4) \
        for (std::int64_t i5=1; i5 <= k_lim_same_shape; ++i5) { \
            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_F32, i0, i1, i2, i3, i4, i5); \
            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f); \
            \
            mag_tensor_t* r = mag_##op##s(x, static_cast<float>(i0+i1+i2+i3+i4+i5)*0.221f); \
             \
            \
            const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x)); \
            const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r)); \
            ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x)); \
            for (std::int64_t i=0; i < mag_tensor_numel(x); ++i) { \
                ASSERT_FLOAT_EQ(b_r[i], b_x[i] scalar_op (static_cast<float>(i0+i1+i2+i3+i4+i5)*0.221f)); \
            } \
            mag_tensor_decref(r); \
            mag_tensor_decref(x); \
        } \
        \
        mag_ctx_destroy(ctx); \
    }

impl_test_binary_op(add_f32, add, +)
impl_test_binary_op(sub_f32, sub, -)
impl_test_binary_op(mul_f32, mul, *)
impl_test_binary_op(div_f32, div, /)

#undef impl_test_binary_op

static void mag__inner_matmul_naive(
    const float* A,
    const float* B,
    float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K
) {
    for (int64_t i = 0; i < M * N; ++i) C[i] = 0.0f;
    for (int64_t i = 0; i < M; ++i) {       // Rows of A and C
        for (int64_t k = 0; k < K; ++k) {   // Columns of A, Rows of B
            float a_ik = A[i * K + k];      // Access A[i][k]
            for (int64_t j = 0; j < N; ++j) { // Columns of B and C
                C[i * N + j] += a_ik * B[k * N + j]; // C[i][j] += A[i][k] * B[k][j]
            }
        }
    }
}

TEST(compute_cpu, matmul_inner_naive) {
    static constexpr float A[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr float B[2] = {0.5f, -1.0f};
    float C[3];
    mag__inner_matmul_naive(A, B, C, 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);
}

TEST(compute_cpu, matmul_f32_same_shape_2x2) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    static constexpr float A_values[2][2] = {
        {1.6354027, -1.3607267},
        {1.8556793, 1.1689897}
    };
    static constexpr float B_values[2][2] = {
        {-0.6105532, 0.10695228},
        {-1.0069681, -0.40955952}
    };

    // Manually set known values for A and B
    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 2, 2);
    mag_tensor_copy_buffer_from(A, A_values, sizeof(A_values));

    mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 2, 2);
    mag_tensor_copy_buffer_from(B, B_values, sizeof(B_values));

    // Create result tensor R for matrix multiplication
    mag_tensor_t* R = mag_matmul(A, B);
    mag_tensor_print(R, true, true);
    auto* buf = static_cast<const float*>(mag_tensor_data_ptr(R));

    static constexpr float expected[2*2] = {
        0.3717081,   0.7322086,
        -2.3101263, -0.28030172
    };

    for (int i = 0; i < 2*2; ++i) {
        ASSERT_FLOAT_EQ(buf[i], expected[i]);
    }

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, matmul_f32_different_shape_2x2) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    static constexpr  float AV[3*2] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr float BV[2] = {0.5f, -1.0f};

    // Manually set known values for A and B
    mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 3, 2);
    mag_tensor_copy_buffer_from(A, AV, sizeof(AV));

    mag_tensor_t* B = mag_tensor_create_1d(ctx, MAG_DTYPE_F32, 2);
    mag_tensor_copy_buffer_from(B, BV, sizeof(BV));

    // Create result tensor R for matrix multiplication
    mag_tensor_t* R = mag_matmul(A, B);
    //ASSERT_EQ(mag_tensor_rank(R), 1);
    ASSERT_EQ(mag_tensor_shape(R)[0], 3);
    const auto* C = static_cast<const float*>(mag_tensor_data_ptr(R));
    //mag__inner_matmul_naive(A, B, C, 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);

    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, arithmetic_mean) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_F32, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_mean(A);
    ASSERT_NE(R, nullptr);
    double a_mean = 0.0;
    for (std::int64_t i=0; i < mag_tensor_numel(A); ++i)
        a_mean += static_cast<double>(static_cast<const float*>(mag_tensor_data_ptr(A))[i]);
    a_mean /= static_cast<double>(mag_tensor_numel(A));
    ASSERT_NEAR(static_cast<float>(a_mean), *static_cast<const float*>(mag_tensor_data_ptr(R)), 1e-9);
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, min) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_F32, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_min(A);
    ASSERT_NE(R, nullptr);
    float a_min = *std::min_element(static_cast<const float*>(mag_tensor_data_ptr(A)), static_cast<const float*>(mag_tensor_data_ptr(A)) + mag_tensor_numel(A));
    ASSERT_FLOAT_EQ(a_min, *static_cast<const float*>(mag_tensor_data_ptr(R)));
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, max) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_F32, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_max(A);
    ASSERT_NE(R, nullptr);
    float a_min = *std::max_element(static_cast<const float*>(mag_tensor_data_ptr(A)), static_cast<const float*>(mag_tensor_data_ptr(A)) + mag_tensor_numel(A));
    ASSERT_FLOAT_EQ(a_min, *static_cast<const float*>(mag_tensor_data_ptr(R)));
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, hsum) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_F32, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_sum(A);
    ASSERT_NE(R, nullptr);
    double a_sum = 0.0;
    for (std::int64_t i=0; i < mag_tensor_numel(A); ++i)
        a_sum += static_cast<double>(static_cast<const float*>(mag_tensor_data_ptr(A))[i]);
    ASSERT_NEAR(static_cast<float>(a_sum), *static_cast<const float*>(mag_tensor_data_ptr(R)), 1e-9);
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 8192, 8192, 3);
    mag_tensor_t* B = mag_clone(A);
    mag_tensor_fill(B, 3.0);
    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, heavy_compute_single_op_scalar) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_1d(ctx, MAG_DTYPE_F32, 1);
    mag_tensor_t* B =  mag_clone(A);
    mag_tensor_fill(B, 3.0);
    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(compute_cpu, threaded_add) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 512, 512, 32);
    mag_tensor_fill(A, 1.0f);

    mag_tensor_t* B = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 512, 512, 32);
    mag_tensor_fill(B, 2.0f);

    ASSERT_EQ(A->numel, B->numel);

    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);

    const auto* a = static_cast<const float*>(mag_tensor_data_ptr(A));
    const auto* b = static_cast<const float*>(mag_tensor_data_ptr(B));
    const auto* r = static_cast<const float*>(mag_tensor_data_ptr(R));
    const auto numel = mag_tensor_numel(R);
    for (std::int64_t i=0; i < numel; ++i) {
        ASSERT_FLOAT_EQ(r[i], a[i] + b[i]);
    }

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

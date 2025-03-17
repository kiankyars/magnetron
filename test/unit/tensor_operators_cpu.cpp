// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// For some tests, we use a larger absolute error than machine epsilon,
// because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar high-precision lambdas.

#include "prelude.hpp"
#include <algorithm>
#include <cmath>

static constexpr std::int64_t k_lim_same_shape = 4;
static constexpr std::int64_t k_lim_broadcast = 2;

#define impl_test_unary_op(name, eps, op, scalar_op)                                                                   \
    TEST(operators_cpu, name##_same_shape) {                                                                             \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
                                                                                                                       \
        for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)                                                        \
            for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)                                                    \
                for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)                                                \
                    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)                                            \
                        for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)                                        \
                            for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {                                  \
                                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);  \
                                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                         \
                                                                                                                       \
                                mag_tensor_t* r = mag_##op(x);                                                         \
                                                                                                                       \
                                const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));                   \
                                const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));                   \
                                ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));                                   \
                                ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                             \
                                for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                               \
                                    ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), (eps));                                     \
                                }                                                                                      \
                                mag_tensor_decref(r);                                                                  \
                                mag_tensor_decref(x);                                                                  \
                            }                                                                                          \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }                                                                                                                  \
    TEST(operators_cpu, name##_same_shape_inplace) {                                                                     \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
                                                                                                                       \
        for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)                                                        \
            for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)                                                    \
                for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)                                                \
                    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)                                            \
                        for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)                                        \
                            for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {                                  \
                                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);  \
                                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                         \
                                std::vector<float> x_origin{};                                                         \
                                mag_tensor_buf_e8m23_to_vec(x, x_origin);                                              \
                                                                                                                       \
                                mag_tensor_t* r = mag_##op##_(x);                                                      \
                                mag_tensor_set_name(r, "result");                                                      \
                                                                                                                       \
                                const auto* b_x = x_origin.data();                                                     \
                                mag_tensor_set_name(x, "X");                                                           \
                                const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));                   \
                                ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));                                   \
                                ASSERT_EQ(mag_tensor_data_ptr(x), mag_tensor_data_ptr(r));                             \
                                for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                               \
                                    ASSERT_NEAR(b_r[i], scalar_op(b_x[i]), (eps));                                     \
                                }                                                                                      \
                                mag_tensor_decref(r);                                                                  \
                                mag_tensor_decref(x);                                                                  \
                            }                                                                                          \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }

impl_test_unary_op(abs, 1e-6, abs, [](float x) -> float { return std::abs(x); })

TEST(operators_cpu, neg_same_shape) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
                            mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
                            mag_tensor_t* r = mag_neg(x);
                            const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));
                            const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));
                            ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));
                            for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {
                                ASSERT_EQ(b_r[i], -b_x[i]);
                            }
                            mag_tensor_decref(x);
                            mag_tensor_decref(r);
                        }
    mag_ctx_destroy(ctx);
}

impl_test_unary_op(log, 1e-6, log, [](float x) -> float { return std::log(x); })

impl_test_unary_op(sqr, 1e-9, sqr, [](float x) -> float { return x * x; })

impl_test_unary_op(sqrt, 1e-9, sqrt, [](float x) -> float { return std::sqrt(x); })

impl_test_unary_op(sin, 1e-6, sin, [](float x) -> float { return std::sin(x); })

#if 0 // TODO fix
impl_test_unary_op(cos, 1e-6, cos, [](float x) -> float {
return std::cos(x);
})
#endif

impl_test_unary_op(step, 1e-9, step, [](float x) -> float { return x >= 0.0f ? 1.0f : 0.0f; })

impl_test_unary_op(exp, 1e-6, exp, [](float x) -> float { return std::exp(x); })

impl_test_unary_op(softmax, 1e-6, softmax, [](float x) -> float { return std::exp(x); })
impl_test_unary_op(softmax_dv, 1e-6, softmax_dv,
[](float x) -> float { return std::exp(x); })

impl_test_unary_op(sigmoid, 1e-6, sigmoid, [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); })
impl_test_unary_op(sigmoid_dv, 1e-6, sigmoid_dv, [](float x) -> float { return x * (1.0f - x); })
impl_test_unary_op(hard_sigmoid, 1e-6, hard_sigmoid, [](float x) -> float {
   return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
})
// impl_test_unary_op(hard_sigmoid_dv, HARD_SIGMOID_DV, [](float x) -> float {
//     return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
// })

impl_test_unary_op(silu, 1e-6, silu, [](float x) -> float { return x / (1.0f + std::exp(-x)); })
// impl_test_unary_op(silu_dv, SILU_DV, [](float x) -> float {
//     return -(std::exp(x) / ((std::exp(x)+1.0f)*(std::exp(x)+1.0f)));
// })

impl_test_unary_op(tanh, 1e-3, tanh, [](float x) -> float { return std::tanh(x); })
impl_test_unary_op(tanh_dv, 1e-4, tanh_dv, [](float x) -> float { return 1.0f - (std::tanh(x) * std::tanh(x)); })

impl_test_unary_op(relu, 1e-9, relu, [](float x) -> float { return std::max(x, 0.0f); })
impl_test_unary_op(relu_dv, 1e-9, relu_dv, [](float x) -> float { return x <= 0.0f ? 0.0f : 1.0f; })

impl_test_unary_op(gelu, 1e-3, gelu, [](float x) -> float { return 0.5f * x * (1.0f + std::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
                                       })
    // impl_test_unary_op(gelu_dv, GELU_DV, [](float x) -> float {
    //     return x <= 0.0f ? 0.0f : 1.0f;
    // })

TEST(operators_cpu, clone_same_shape) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)
    for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)
    for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)
    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)
    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)
    for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
        mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
        mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
        mag_tensor_t* r = mag_clone(x);
        const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));
        const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));
        ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));
        for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {
            ASSERT_EQ(b_r[i], b_x[i]);
        }
        mag_tensor_decref(x);
        mag_tensor_decref(r);
    }
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, clone_transpose) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)
    for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)
    for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)
    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)
    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)
    for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
        mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
        mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
        mag_tensor_t* r = mag_clone(mag_transpose(x));
        const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));
        const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));
        ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(r));
        for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {
            ASSERT_EQ(b_r[i], b_x[i]);
        }
        mag_tensor_decref(x);
        mag_tensor_decref(r);
    }
    mag_ctx_destroy(ctx);
}

#undef impl_test_unary_op

#define impl_test_binary_op(name, dtype, dtype_T, scalar_op, castor, contor, eps)                                              \
    TEST(operators_cpu, name##_##dtype##_same_shape) {                                                                 \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
        for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)                                                        \
            for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)                                                    \
                for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)                                                \
                    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)                                            \
                        for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)                                        \
                            for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {                                  \
                                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0, i1, i2, i3, i4, i5);\
                                mag_tensor_t* y = mag_clone(x);                                                        \
                                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                         \
                                mag_tensor_fill_random_uniform(y, -5.0f, 5.0f);                                        \
                                                                                                                       \
                                mag_tensor_t* r = mag_##name(x, y);                                                    \
                                                                                                                       \
                                const auto* b_x = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(x));       \
                                const auto* b_y = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(y));       \
                                const auto* b_r = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(r));       \
                                ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                             \
                                ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(y));                                   \
                                ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(y));                                   \
                                for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                               \
                                    ASSERT_NEAR(castor(b_r[i]), scalar_op(b_x[i], b_y[i]), eps);                       \
                                }                                                                                      \
                                mag_tensor_decref(r);                                                                  \
                                mag_tensor_decref(x);                                                                  \
                                mag_tensor_decref(y);                                                                  \
                            }                                                                                          \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    TEST(operators_cpu, name##_##dtype##_scalar_broadcast) {                                                           \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
        for (std::int64_t factor = 2; factor <= 4; ++factor)                                                           \
            for (std::int64_t i0 = 1; i0 <= k_lim_broadcast; ++i0)                                                     \
                for (std::int64_t i1 = 1; i1 <= k_lim_broadcast; ++i1)                                                 \
                    for (std::int64_t i2 = 1; i2 <= k_lim_broadcast; ++i2)                                             \
                        for (std::int64_t i3 = 1; i3 <= k_lim_broadcast; ++i3)                                         \
                            for (std::int64_t i4 = 1; i4 <= k_lim_broadcast; ++i4)                                     \
                                for (std::int64_t i5 = 1; i5 <= k_lim_broadcast; ++i5) {                               \
                                    mag_tensor_t* x =                                                                  \
                                        mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0 * factor, i1 * factor,           \
                                                             i2 * factor, i3 * factor, i4 * factor, i5 * factor);      \
                                    mag_tensor_t* y =                                                                  \
                                        mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0, i1, i2, i3, i4, i5);            \
                                    mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                     \
                                    mag_tensor_fill(y, 2.2f);                                                          \
                                                                                                                       \
                                    mag_tensor_t* r = mag_##name(x, y);                                                  \
                                                                                                                       \
                                    const auto* b_x = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(x));               \
                                    const auto* b_r = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(r));               \
                                    ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(x));                               \
                                    ASSERT_NE(mag_tensor_numel(x), mag_tensor_numel(y));                               \
                                    ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                         \
                                    for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                           \
                                        ASSERT_NEAR(castor(b_r[i]), scalar_op(b_x[i], contor(2.2f)), eps);                                \
                                    }                                                                                  \
                                    mag_tensor_decref(r);                                                              \
                                    mag_tensor_decref(x);                                                              \
                                    mag_tensor_decref(y);                                                              \
                                }                                                                                      \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }                                                                                                                  \
    TEST(operators_cpu, name##_##dtype##_same_shape_inplace) {                                                                  \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
        for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)                                                        \
            for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)                                                    \
                for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)                                                \
                    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)                                            \
                        for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)                                        \
                            for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {                                  \
                                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0, i1, i2, i3, i4, i5);  \
                                mag_tensor_t* y = mag_clone(x);                                                        \
                                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                         \
                                mag_tensor_fill_random_uniform(y, -5.0f, 5.0f);                                        \
                                std::vector<mag_##dtype_T##_t> x_origin{};                                                         \
                                mag_tensor_buf_##dtype_T##_to_vec(x, x_origin);                                              \
                                                                                                                       \
                                mag_tensor_t* r = mag_##name##_(x, y);                                                   \
                                                                                                                       \
                                const auto* b_x = x_origin.data();                                                     \
                                const auto* b_xx = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(x));                  \
                                const auto* b_y = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(y));                   \
                                const auto* b_r = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(r));                   \
                                ASSERT_EQ(mag_tensor_numel(x), mag_tensor_numel(y));                                   \
                                ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(y));                                   \
                                ASSERT_EQ(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                             \
                                for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                               \
                                    ASSERT_NEAR(castor(b_r[i]), castor(b_xx[i]), eps);                                                  \
                                    ASSERT_NEAR(castor(b_r[i]), scalar_op(b_x[i], b_y[i]), eps);                                  \
                                }                                                                                      \
                                mag_tensor_decref(r);                                                                  \
                                mag_tensor_decref(x);                                                                  \
                                mag_tensor_decref(y);                                                                  \
                            }                                                                                          \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    TEST(operators_cpu, name##_##dtype##_scalar_broadcast_inplace) {                                                            \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
        for (std::int64_t factor = 2; factor <= 4; ++factor)                                                           \
            for (std::int64_t i0 = 1; i0 <= k_lim_broadcast; ++i0)                                                     \
                for (std::int64_t i1 = 1; i1 <= k_lim_broadcast; ++i1)                                                 \
                    for (std::int64_t i2 = 1; i2 <= k_lim_broadcast; ++i2)                                             \
                        for (std::int64_t i3 = 1; i3 <= k_lim_broadcast; ++i3)                                         \
                            for (std::int64_t i4 = 1; i4 <= k_lim_broadcast; ++i4)                                     \
                                for (std::int64_t i5 = 1; i5 <= k_lim_broadcast; ++i5) {                               \
                                    mag_tensor_t* x =                                                                  \
                                        mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0 * factor, i1 * factor,           \
                                                             i2 * factor, i3 * factor, i4 * factor, i5 * factor);      \
                                    mag_tensor_t* y =                                                                  \
                                        mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0, i1, i2, i3, i4, i5);            \
                                    mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                     \
                                    mag_tensor_fill(y, 2.2f);                                                          \
                                                                                                                       \
                                    std::vector<mag_##dtype_T##_t> x_origin{};                                                     \
                                    mag_tensor_buf_##dtype_T##_to_vec(x, x_origin);                                          \
                                                                                                                       \
                                    mag_tensor_t* r = mag_##name##_(x, y);                                               \
                                                                                                                       \
                                    const auto* b_x = x_origin.data();                                                 \
                                    const auto* b_xx = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(x));              \
                                    const auto* b_r = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(r));               \
                                    ASSERT_EQ(mag_tensor_numel(r), mag_tensor_numel(x));                               \
                                    ASSERT_NE(mag_tensor_numel(x), mag_tensor_numel(y));                               \
                                    ASSERT_EQ(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                         \
                                    for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                           \
                                        ASSERT_NEAR(castor(b_r[i]), castor(b_xx[i]), eps);                                              \
                                        ASSERT_NEAR(castor(b_r[i]), scalar_op(b_x[i], contor(2.2f)), eps);                                \
                                    }                                                                                  \
                                    mag_tensor_decref(r);                                                              \
                                    mag_tensor_decref(x);                                                              \
                                    mag_tensor_decref(y);                                                              \
                                }                                                                                      \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }                                                                                                                  \
    TEST(operators_cpu, name##_##dtype##_scalar) {                                                                              \
        mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);                                                  \
        for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)                                                        \
            for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)                                                    \
                for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)                                                \
                    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)                                            \
                        for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)                                        \
                            for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {                                  \
                                mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_##dtype, i0, i1, i2, i3, i4, i5);  \
                                mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);                                         \
                                                                                                                       \
                                mag_tensor_t* r =                                                                      \
                                    mag_##name##s(x, static_cast<float>(i0 + i1 + i2 + i3 + i4 + i5) * 0.221f);          \
                                                                                                                       \
                                const auto* b_x = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(x));                   \
                                const auto* b_r = static_cast<const mag_##dtype_T##_t*>(mag_tensor_data_ptr(r));                   \
                                ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));                             \
                                for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {                               \
                                    ASSERT_NEAR(                                                                   \
                                        castor(b_r[i]),                                                                        \
                                        scalar_op(b_x[i], contor(static_cast<float>(i0 + i1 + i2 + i3 + i4 + i5) * 0.221f)), eps);   \
                                }                                                                                      \
                                mag_tensor_decref(r);                                                                  \
                                mag_tensor_decref(x);                                                                  \
                            }                                                                                          \
                                                                                                                       \
        mag_ctx_destroy(ctx);                                                                                          \
    }

#define mag_sadd(x, y) ((x)+(y))
#define mag_ssub(x, y) ((x)-(y))
#define mag_smul(x, y) ((x)*(y))
#define mag_sdiv(x, y) ((x)/(y))
#define mag_castor(x) (x)
#define mag_contor mag_castor
impl_test_binary_op(add, E8M23, e8m23, mag_sadd, mag_castor, mag_contor, 1e-6)
impl_test_binary_op(sub, E8M23, e8m23, mag_ssub, mag_castor, mag_contor, 1e-6)
impl_test_binary_op(mul, E8M23, e8m23, mag_smul, mag_castor, mag_contor, 1e-6)
impl_test_binary_op(div, E8M23, e8m23, mag_sdiv, mag_castor, mag_contor, 1e-6)
#undef mag_castor
#undef mag_contor
#undef mag_sadd
#undef mag_ssub
#undef mag_smul
#undef mag_sdiv
#define mag_castor(x) (mag_e5m10_to_e8m23_ref(x))
#define mag_contor(x) (mag_e8m23_to_e5m10_ref(x))
#define mag_sadd(x, y) mag_castor(x)+mag_castor(y)
#define mag_ssub(x, y) mag_castor(x)-mag_castor(y)
#define mag_smul(x, y) mag_castor(x)*mag_castor(y)
#define mag_sdiv(x, y) mag_castor(x)/mag_castor(y)
impl_test_binary_op(add, E5M10, e5m10, mag_sadd, mag_castor, mag_contor, 1e-2)
impl_test_binary_op(sub, E5M10, e5m10, mag_ssub, mag_castor, mag_contor, 1e-2)
impl_test_binary_op(mul, E5M10, e5m10, mag_smul, mag_castor, mag_contor, 1e-2)
impl_test_binary_op(div, E5M10, e5m10, mag_sdiv, mag_castor, mag_contor, 1e-2)
#undef mag_castor
#undef mag_contor
#undef mag_sadd
#undef mag_ssub
#undef mag_smul
#undef mag_sdiv

#undef impl_test_binary_op

TEST(operators_cpu, pows) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0)
    for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1)
    for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2)
    for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3)
    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4)
    for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
        mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
        mag_tensor_fill_random_uniform(x, 0.0f, 1.0f);
        mag_tensor_t* r = mag_pows(x, static_cast<float>(i0 + i1 + i2 + i3 + i4 + i5) * 0.221f);
        const auto* b_x = static_cast<const float*>(mag_tensor_data_ptr(x));
        const auto* b_r = static_cast<const float*>(mag_tensor_data_ptr(r));
        ASSERT_NE(mag_tensor_data_ptr(r), mag_tensor_data_ptr(x));
        for (std::int64_t i = 0; i < mag_tensor_numel(x); ++i) {
            ASSERT_FLOAT_EQ(
                b_r[i],
                std::pow(b_x[i], (static_cast<float>(i0 + i1 + i2 + i3 + i4 + i5) * 0.221f)));
        }
        mag_tensor_decref(r);
        mag_tensor_decref(x);
    }
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, arithmetic_mean) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_mean(A);
    ASSERT_NE(R, nullptr);
    double a_mean = 0.0;
    for (std::int64_t i = 0; i < mag_tensor_numel(A); ++i)
        a_mean += static_cast<double>(static_cast<const float*>(mag_tensor_data_ptr(A))[i]);
    a_mean /= static_cast<double>(mag_tensor_numel(A));
    ASSERT_NEAR(static_cast<float>(a_mean), *static_cast<const float*>(mag_tensor_data_ptr(R)), 1e-9);
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, min) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_min(A);
    ASSERT_NE(R, nullptr);
    float a_min = *std::min_element(static_cast<const float*>(mag_tensor_data_ptr(A)),
                                    static_cast<const float*>(mag_tensor_data_ptr(A)) + mag_tensor_numel(A));
    ASSERT_FLOAT_EQ(a_min, *static_cast<const float*>(mag_tensor_data_ptr(R)));
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, max) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_max(A);
    ASSERT_NE(R, nullptr);
    float a_min = *std::max_element(static_cast<const float*>(mag_tensor_data_ptr(A)),
                                    static_cast<const float*>(mag_tensor_data_ptr(A)) + mag_tensor_numel(A));
    ASSERT_FLOAT_EQ(a_min, *static_cast<const float*>(mag_tensor_data_ptr(R)));
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, hsum) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_4d(ctx, MAG_DTYPE_E8M23, 4, 1, 3, 2);
    mag_tensor_fill_random_uniform(A, -1.0f, 1.0f);
    mag_tensor_t* R = mag_sum(A);
    ASSERT_NE(R, nullptr);
    double a_sum = 0.0;
    for (std::int64_t i = 0; i < mag_tensor_numel(A); ++i)
        a_sum += static_cast<double>(static_cast<const float*>(mag_tensor_data_ptr(A))[i]);
    ASSERT_NEAR(static_cast<float>(a_sum), *static_cast<const float*>(mag_tensor_data_ptr(R)), 1e-9);
    mag_tensor_decref(A);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, heavy_compute_single_op) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 8192, 8192, 3);
    mag_tensor_t* B = mag_clone(A);
    mag_tensor_fill(B, 3.0);
    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, heavy_compute_single_op_scalar) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_t* B = mag_clone(A);
    mag_tensor_fill(B, 3.0);
    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

TEST(operators_cpu, threaded_add) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_tensor_t* A = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 512, 512, 32);
    mag_tensor_fill(A, 1.0f);

    mag_tensor_t* B = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, 512, 512, 32);
    mag_tensor_fill(B, 2.0f);

    ASSERT_EQ(A->numel, B->numel);

    mag_tensor_t* R = mag_add(A, B);
    ASSERT_NE(R, nullptr);

    const auto* a = static_cast<const float*>(mag_tensor_data_ptr(A));
    const auto* b = static_cast<const float*>(mag_tensor_data_ptr(B));
    const auto* r = static_cast<const float*>(mag_tensor_data_ptr(R));
    const auto numel = mag_tensor_numel(R);
    for (std::int64_t i = 0; i < numel; ++i) {
        ASSERT_FLOAT_EQ(r[i], a[i] + b[i]);
    }

    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(R);
    mag_ctx_destroy(ctx);
}

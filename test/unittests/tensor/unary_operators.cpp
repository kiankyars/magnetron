// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};
static constexpr std::int64_t broadcast_lim {lim-1};

#define impl_unary_operator_test_group(name, data_type, lambda) \
    TEST(cpu_tensor_unary_ops, name##_same_shape_##data_type) { \
        test::test_unary_operator<false, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_broadcast_##data_type) { \
        test::test_unary_operator<true, false>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_unary_operator<false, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_unary_operator<true, true>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    }

impl_unary_operator_test_group(abs, e8m23, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(abs, e5m10, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(sgn, e8m23, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(sgn, e5m10, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(neg, e8m23, [](auto x) { return -x; })
impl_unary_operator_test_group(neg, e5m10, [](auto x) { return -x; })
impl_unary_operator_test_group(log, e8m23, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(log, e5m10, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(sqr, e8m23, [](auto x) { return std::pow(x, 2.0f); })
impl_unary_operator_test_group(sqr, e5m10, [](auto x) { return std::pow(x, 2.0f); })
impl_unary_operator_test_group(sqrt, e8m23, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(sqrt, e5m10, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(sin, e8m23, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(sin, e5m10, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(cos, e8m23, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(cos, e5m10, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(step, e8m23, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(step, e5m10, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(exp, e8m23, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(exp, e5m10, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(floor, e8m23, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(floor, e5m10, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(ceil, e8m23, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(ceil, e5m10, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(round, e8m23, [](auto x) { return std::round(x); })
impl_unary_operator_test_group(round, e5m10, [](auto x) { return std::round(x); })

TEST(cpu_unary_tensor_ops, softmax_same_shape_e8m23) {
    auto ctx = context{compute_device::cpu};
    for_all_shape_perms(lim, /*BROADCAST=*/false ? 2 : 1, [&](std::span<const std::int64_t> shape) {
        tensor t_a {ctx, dtype::e8m23, shape};
        t_a.fill_rand_uniform(-10.0f, 10.0f);
        std::vector<e8m23_t> d_a {t_a.to_vector()};
        tensor t_r {t_a.softmax()};
        if constexpr (/*INPLACE=*/false) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        std::vector<e8m23_t> d_r {t_r.to_vector()};
        ASSERT_EQ(d_a.size(), d_r.size());
        ASSERT_EQ(t_a.dtype(), t_r.dtype());
        std::vector<e8m23_t> d_correct {};
        d_correct.resize(d_r.size());
        softmax(d_a.data(), d_correct.data(), d_a.size());
        for (std::int64_t i = 0; i < d_r.size(); ++i) {
            ASSERT_NEAR(d_correct[i], d_r[i], test::dtype_traits<test::e8m23_t>::test_eps);
        }
    });
}

impl_unary_operator_test_group(sigmoid, e8m23, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(sigmoid, e5m10, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(hard_sigmoid, e8m23, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f)); })
impl_unary_operator_test_group(hard_sigmoid, e5m10, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f)); })  // TODO
impl_unary_operator_test_group(silu, e8m23, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(silu, e5m10, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(tanh, e8m23, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(tanh, e5m10, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(relu, e8m23, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(relu, e5m10, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(gelu, e8m23, [](auto x) { return (x * 0.5f * (1.0f + std::tanh(x))); })
impl_unary_operator_test_group(gelu, e5m10, [](auto x) { return (x * 0.5f * (1.0f + std::tanh(x))); })

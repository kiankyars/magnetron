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
impl_unary_operator_test_group(sgn, e8m23, [](auto x) { return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f); })
impl_unary_operator_test_group(sgn, e5m10, [](auto x) { return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f); })
impl_unary_operator_test_group(neg, e8m23, [](auto x) { return -x; })
impl_unary_operator_test_group(neg, e5m10, [](auto x) { return -x; })
impl_unary_operator_test_group(log, e8m23, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(log, e5m10, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(sqr, e8m23, [](auto x) { return x*x; })
impl_unary_operator_test_group(sqr, e5m10, [](auto x) { return x*x; })
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

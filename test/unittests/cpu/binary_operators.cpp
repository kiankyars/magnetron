// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

static constexpr std::int64_t lim {4};

TEST(cpu_binary_operators, print_test_info) {
    std::cout << "=== Binary Operators ===" << std::endl;
    std::set<std::vector<std::int64_t>> perms {};
    test::for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        perms.emplace(std::vector<std::int64_t>{shape.begin(), shape.end()});
    });
    test::for_all_shape_perms(lim, 2, [&](std::span<const std::int64_t> shape) {
        perms.emplace(std::vector<std::int64_t>{shape.begin(), shape.end()});
    });
    std::cout << "=== Tested Shape Permutations (" << perms.size() << ") ===" << std::endl;
    for (const auto& shape : perms) {
        std::cout << test::shape_to_string(shape) << std::endl;
    }
}

#define impl_binary_operator_test_group(name, op, data_type) \
    TEST(cpu_binary_operators, name##_same_shape_##data_type) { \
        test::test_binary_operator<false, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_binary_operators, name##_broadcast_##data_type) { \
        test::test_binary_operator<true, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_binary_operators, name##_inplace_same_shape_##data_type) { \
        test::test_binary_operator<false, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_binary_operators, name##_inplace_broadcast_##data_type) { \
        test::test_binary_operator<true, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    }

impl_binary_operator_test_group(add, +, e8m23)
impl_binary_operator_test_group(add, +, e5m10)

impl_binary_operator_test_group(sub, -, e8m23)
impl_binary_operator_test_group(sub, -, e5m10)

impl_binary_operator_test_group(mul, *, e8m23)
impl_binary_operator_test_group(mul, *, e5m10)

impl_binary_operator_test_group(div, /, e8m23)
impl_binary_operator_test_group(div, /, e5m10)

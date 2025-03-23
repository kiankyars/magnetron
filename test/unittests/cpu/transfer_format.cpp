// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};

TEST(cpu_transfer, load_store_e8m23) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution<e8m23_t> dist {dtype_traits<e8m23_t>::min, dtype_traits<e8m23_t>::max};
        e8m23_t fill_val {dist(gen)};
        tensor t {ctx, dtype::e8m23, shape};
        t.fill(fill_val);
        std::vector<e8m23_t> data {t.to_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_val);
        }
    });
}

TEST(cpu_transfer, load_store_e5m10) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution<e8m23_t> dist {
            -1.0f,
            1.0f
        };
        e8m23_t fill_val {dist(gen)};
        tensor t {ctx, dtype::e5m10, shape};
        t.fill(fill_val);
        std::vector<e8m23_t> data {t.to_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], fill_val, 1e-3f);
        }
    });
}

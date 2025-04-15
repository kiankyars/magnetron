// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>
#include <__filesystem/filesystem_error.h>
#include <__filesystem/operations.h>

using namespace magnetron;

TEST(file_storage, open_close_empty) {
    context ctx {compute_device::cpu};
    storage_stream stream {ctx};
    ASSERT_NE(&*stream, nullptr);
}

TEST(file_storage, put_get_tensor) {
    context ctx {compute_device::cpu};
    storage_stream stream {ctx};

    tensor a {ctx, dtype::f32, 2, 2};
    tensor b {ctx, dtype::f32, 4, 4};

    ASSERT_EQ(stream.get("tensor_a"), std::nullopt);
    stream.put("tensor_a", a);
    ASSERT_EQ(&*stream.get("tensor_a").value(), &*a);
    ASSERT_EQ(stream.get("tensor_a").value().dtype(), dtype::f32);
    ASSERT_EQ(stream.get("tensor_a").value().shape()[0], 2);
    ASSERT_EQ(stream.get("tensor_a").value().shape()[1], 2);
    ASSERT_EQ(stream.get("tensor_b"), std::nullopt);
    stream.put("tensor_b", b);
    ASSERT_EQ(&*stream.get("tensor_b").value(), &*b);
    ASSERT_EQ(stream.get("tensor_b").value().dtype(), dtype::f32);
    ASSERT_EQ(stream.get("tensor_b").value().shape()[0], 4);
    ASSERT_EQ(stream.get("tensor_b").value().shape()[1], 4);
}

TEST(file_storage, serialize) {
    context ctx {compute_device::cpu};
    storage_stream stream {ctx};
    tensor a {ctx, dtype::f32, 2, 2};
    tensor b {ctx, dtype::f32, 4, 4};
    stream.put("tensor_a", a);
    stream.put("tensor_b", b);

    stream.serialize("test_storage.mag");
    ASSERT_TRUE(std::filesystem::exists("test_storage.mag"));
    ASSERT_TRUE(std::filesystem::remove("test_storage.mag"));
}

TEST(file_storage, deserialize) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<> dist {-10.0f, 10.0f};
    std::vector<test::e8m23_t> fill_a {};
    fill_a.resize(2*2);
    std::generate_n(fill_a.begin(), fill_a.size(), [&]() noexcept -> test::e8m23_t { return dist(gen); });
    std::vector<test::e8m23_t> fill_b {};
    fill_b.resize(4*4);
    std::generate_n(fill_b.begin(), fill_b.size(), [&]() noexcept -> test::e8m23_t { return dist(gen); });

   {
        context ctx {compute_device::cpu};
        storage_stream stream {ctx};
        tensor a {ctx, dtype::f32, 2, 2};
        ASSERT_EQ(fill_a.size(), a.numel());
        a.fill_from(fill_a);
        tensor b {ctx, dtype::f32, 4, 4};
        ASSERT_EQ(fill_b.size(), b.numel());
        b.fill_from(fill_b);
        stream.put("tensor_a", a);
        stream.put("tensor_b", b);

        stream.serialize("test_storage2.mag");
        ASSERT_TRUE(std::filesystem::exists("test_storage2.mag"));
   }
   {
        context ctx {compute_device::cpu};
        storage_stream stream {ctx, "test_storage2.mag"};
        ASSERT_TRUE(stream.get("tensor_a").has_value());
        ASSERT_TRUE(stream.get("tensor_b").has_value());
        tensor a {stream.get("tensor_a").value()};
        tensor b {stream.get("tensor_b").value()};
        ASSERT_EQ(a.dtype(), dtype::f32);
        ASSERT_EQ(a.shape()[0], 2);
        ASSERT_EQ(a.shape()[1], 2);
        ASSERT_EQ(b.dtype(), dtype::f32);
        ASSERT_EQ(b.shape()[0], 4);
        ASSERT_EQ(b.shape()[1], 4);
        ASSERT_EQ(a.data_size(), 2*2*sizeof(float));
        ASSERT_EQ(b.data_size(), 4*4*sizeof(float));
        ASSERT_EQ(a.numel(), 2*2);
        ASSERT_EQ(b.numel(), 4*4);
        std::vector<mag_e8m23_t> dat_a {a.to_vector()};
        std::vector<mag_e8m23_t> dat_b {b.to_vector()};
        for (std::size_t i {}; i < a.numel(); ++i) {
            ASSERT_FLOAT_EQ(dat_a[i], fill_a[i]);
        }
        for (std::size_t i {}; i < b.numel(); ++i) {
            ASSERT_FLOAT_EQ(dat_b[i], fill_b[i]);
        }
        ASSERT_TRUE(std::filesystem::remove("test_storage2.mag"));
   }
}

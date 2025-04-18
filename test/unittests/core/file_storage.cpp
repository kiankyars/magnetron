// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>
#include <filesystem>

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

TEST(file_storage, stress_roundtrip) {
    constexpr std::size_t N = 1'000;
    constexpr std::size_t H = 100, W = 100;
    std::vector<std::vector<test::e8m23_t>> reference;
    reference.reserve(N);

    // --- write & serialize ---
    double serialize_time = 0.0;
    {
        context ctx{compute_device::cpu};
        storage_stream writer{ctx};

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> dist{-1.0f, 1000.0f};

        for (std::size_t i = 0; i < N; ++i) {
            tensor t{ctx, dtype::f32, H, W};
            // fill and capture
            std::vector<test::e8m23_t> data(H * W);
            for (auto &x : data) x = dist(gen);
            t.fill_from(data);

            reference.push_back(std::move(data));
            writer.put(std::to_string(i), t);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        writer.serialize("large_storage.mag");
        auto t1 = std::chrono::high_resolution_clock::now();
        serialize_time = std::chrono::duration<double>(t1 - t0).count();

        ASSERT_TRUE(std::filesystem::exists("large_storage.mag"));
    }
    std::cout << "[perf] serialize(" << N << "×" << H << "×" << W << "): "
              << serialize_time << " s\n";

    // --- deserialize & verify ---
    double deserialize_time = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        context ctx{compute_device::cpu};
        storage_stream reader{ctx, "large_storage.mag"};
        for (std::size_t i = 0; i < N; ++i) {
            auto opt = reader.get(std::to_string(i)); /* Pre lazy load tensor */
        }
        auto t1 = std::chrono::high_resolution_clock::now();


        {
            for (std::size_t i = 0; i < N; ++i) {
                auto opt = reader.get(std::to_string(i));
                ASSERT_TRUE(opt.has_value()) << "Missing tensor " << i;
                tensor t = *opt;
                auto loaded = t.to_vector();
                auto &orig = reference[i];
                ASSERT_EQ(loaded.size(), orig.size())
                    << "Size mismatch at tensor " << i;
                for (size_t j = 0; j < loaded.size(); ++j) {
                    ASSERT_FLOAT_EQ(loaded[j], orig[j])
                        << "Mismatch at tensor " << i << ", element " << j;
                }
            }
        }

        std::cout << "[perf] deserialize: "
                  << std::chrono::duration<double>(t1 - t0).count()
                  << " s\n";

        ASSERT_TRUE(std::filesystem::remove("large_storage.mag"));
    }
}

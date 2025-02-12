// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <numbers>
#include <filesystem>

TEST(core, profiler_small_dims) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_profile_start_recording(ctx);
    for (int i=0; i < 10000; ++i) {
        mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, 3, 3);
        mag_tensor_t* B =  mag_sin(A);
        mag_tensor_t* C =  mag_cos(B);
        [[maybe_unused]]
        mag_tensor_t* D = mag_tanh(C);
        mag_tensor_decref(A);
        mag_tensor_decref(B);
        mag_tensor_decref(C);
        mag_tensor_decref(D);
    }
    mag_ctx_profile_stop_recording(ctx, "perf.csv");
    ASSERT_TRUE(std::filesystem::exists("perf.csv"));
    std::filesystem::remove("perf.csv");
    mag_ctx_destroy(ctx);
}

TEST(core, profiler_big_dims) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_ctx_profile_start_recording(ctx);
    mag_tensor_t* A = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, 32, 32, 4, 4, 4, 4);
    mag_tensor_t* B = mag_sin(A);
    mag_tensor_t* C = mag_cos(B);
    [[maybe_unused]]
    mag_tensor_t* D = mag_tanh(C);
    mag_ctx_profile_stop_recording(ctx, nullptr);
    mag_tensor_decref(A);
    mag_tensor_decref(B);
    mag_tensor_decref(C);
    mag_tensor_decref(D);
    mag_ctx_destroy(ctx);
}

#if 0
TEST(core, crc32) {
    ASSERT_EQ(mag__crc32c("Hello, World!", std::strlen("Hello, World!")), 1297420392);
    uint8_t y = 0x3f;
    ASSERT_EQ(mag__crc32c(& y, sizeof(y)), 1015883460);
    ASSERT_EQ(mag__crc32c(nullptr, 0), 0);
    ASSERT_EQ(mag__crc32c("AB", std::strlen("AB")), 3180610794);
    ASSERT_EQ(mag__crc32c(
            "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.",
            std::strlen(
                    "Ich liebe Berliner Kebap, der ist einfach ultra schmackofatz, gerade um 4 Uhr Morgens nach einer langen Clubnacht.")), 60440201);
    std::vector<std::uint8_t> huge {};
    huge.resize(0xffff);
    for (std::size_t i = 0; i < huge.size(); ++i) {
        huge[i] = i % 0xff;
    }
    ASSERT_EQ(mag__crc32c(huge.data(), huge.size()), 2008503331);
}
#endif
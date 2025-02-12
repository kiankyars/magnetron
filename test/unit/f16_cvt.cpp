// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"



#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_fallback_test
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_features_fallback_test
#include "../magnetron/magnetron_cpu_blas.inl"

TEST(fp16, fp16_to_f32) {
    ASSERT_TRUE(std::abs(mag_f16_to_f32(MAG_F16_E) - std::numbers::e_v<float>) < mag_f16_to_f32(MAG_F16_EPS));
    ASSERT_TRUE(std::abs(mag_f16_to_f32(MAG_F16_PI) - std::numbers::pi_v<float>) < mag_f16_to_f32(MAG_F16_EPS));
    ASSERT_TRUE(std::abs(mag_f16_to_f32(MAG_F16_LN2) - std::numbers::ln2_v<float>) < mag_f16_to_f32(MAG_F16_EPS));
    ASSERT_TRUE(std::abs(mag_f16_to_f32(MAG_F16_ONE) - 1.0f) < mag_f16_to_f32(MAG_F16_EPS));
    ASSERT_TRUE(std::abs(mag_f16_to_f32(MAG_F16_ZERO) - 0.0f) < mag_f16_to_f32(MAG_F16_EPS));
    for (std::uint32_t i=0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_f32_to_f16(mag_f16_to_f32(i)));
    }
}

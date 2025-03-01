// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_fallback_test
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_features_fallback_test
#include "../magnetron/magnetron_cpu_blas.inl"

TEST(e5m10, e5m10_to_e8m23) {
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(MAG_E5M10_E) - std::numbers::e_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(MAG_E5M10_PI) - std::numbers::pi_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(MAG_E5M10_LN2) - std::numbers::ln2_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(MAG_E5M10_ONE) - 1.0f) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(MAG_E5M10_ZERO) - 0.0f) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    for (std::uint32_t i=0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(i)));
    }
}

TEST(e5m10, e8m23_to_e5m10) {
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(mag_e8m23_to_e5m10(std::numbers::e_v<float>)) - std::numbers::e_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(mag_e8m23_to_e5m10(std::numbers::pi_v<float>)) - std::numbers::pi_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(mag_e8m23_to_e5m10(std::numbers::ln2_v<float>)) - std::numbers::ln2_v<float>) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(mag_e8m23_to_e5m10(1.0f)) - 1.0f) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_to_e8m23(mag_e8m23_to_e5m10(0.0f)) - 0.0f) < mag_e5m10_to_e8m23(MAG_E5M10_EPS));
    ASSERT_EQ(mag_e8m23_to_e5m10(1.0f), MAG_E5M10_ONE);
    ASSERT_EQ(mag_e8m23_to_e5m10(0.0f), MAG_E5M10_ZERO);
    ASSERT_EQ(mag_e8m23_to_e5m10(std::numbers::e_v<float>), MAG_E5M10_E);
    for (std::uint32_t i=0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(i)));
    }
}

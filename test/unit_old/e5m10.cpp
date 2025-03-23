// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_fallback_test
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_features_fallback_test
#include "../magnetron/magnetron_cpu_blas.inl"

#include <numbers>

TEST(e5m10, e5m10_to_e8m23) {
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_E), std::numbers::e_v<mag_e8m23_t>, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_PI), std::numbers::pi_v<mag_e8m23_t>, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE), 1.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_NEG_ONE), -1.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO), 0.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_E), mag_e5m10_to_e8m23_ref(MAG_E5M10_E), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_PI), mag_e5m10_to_e8m23_ref(MAG_E5M10_PI), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE), mag_e5m10_to_e8m23_ref(MAG_E5M10_ONE), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_NEG_ONE), mag_e5m10_to_e8m23_ref(MAG_E5M10_NEG_ONE), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO), mag_e5m10_to_e8m23_ref(MAG_E5M10_ZERO), 1e-3);
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_E) - std::numbers::e_v<mag_e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_PI) - std::numbers::pi_v<mag_e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_LN2) - std::numbers::ln2_v<mag_e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE) - 1.0f) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO) - 0.0f) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    for (std::uint16_t i = 0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(mag_e5m10_t{.bits = i})).bits);
    }
}

TEST(e5m10, e8m23_to_e5m10) {
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<mag_e8m23_t>).bits, MAG_E5M10_E.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::pi_v<mag_e8m23_t>).bits, MAG_E5M10_PI.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, MAG_E5M10_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(-1.0f).bits, MAG_E5M10_NEG_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, MAG_E5M10_ZERO.bits);

    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<mag_e8m23_t>).bits, mag_e8m23_to_e5m10_ref(std::numbers::e_v<mag_e8m23_t>).bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::pi_v<mag_e8m23_t>).bits, mag_e8m23_to_e5m10_ref(std::numbers::pi_v<mag_e8m23_t>).bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, mag_e8m23_to_e5m10_ref(1.0f).bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(-1.0f).bits, mag_e8m23_to_e5m10_ref(-1.0f).bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, mag_e8m23_to_e5m10_ref(0.0f).bits);

    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::e_v<mag_e8m23_t>)) -
                         std::numbers::e_v<mag_e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::pi_v<mag_e8m23_t>)) -
                         std::numbers::pi_v<mag_e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::ln2_v<mag_e8m23_t>)) -
                         std::numbers::ln2_v<mag_e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(1.0f)) - 1.0f) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(0.0f)) - 0.0f) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, MAG_E5M10_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, MAG_E5M10_ZERO.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<mag_e8m23_t>).bits, MAG_E5M10_E.bits);
    for (std::uint16_t i = 0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(mag_e5m10_t{.bits = i})).bits);
    }
}
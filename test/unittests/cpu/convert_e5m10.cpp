// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_fallback_test
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_features_fallback_test
#include "../magnetron/magnetron_cpu_blas.inl"

#include <numbers>

TEST(cpu_convert_e5m10, e5m10_to_e8m23) {
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_E), std::numbers::e_v<e8m23_t>, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_PI), std::numbers::pi_v<e8m23_t>, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE), 1.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_NEG_ONE), -1.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO), 0.0f, 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_E), static_cast<e8m23_t>(test::e5m10_t{MAG_E5M10_E.bits}), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_PI), static_cast<e8m23_t>(test::e5m10_t{MAG_E5M10_PI.bits}), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE), static_cast<e8m23_t>(test::e5m10_t{MAG_E5M10_ONE.bits}), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_NEG_ONE), static_cast<e8m23_t>(test::e5m10_t{MAG_E5M10_NEG_ONE.bits}), 1e-3);
    ASSERT_NEAR(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO), static_cast<e8m23_t>(test::e5m10_t{MAG_E5M10_ZERO.bits}), 1e-3);
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_E) - std::numbers::e_v<e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_PI) - std::numbers::pi_v<e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_LN2) - std::numbers::ln2_v<e8m23_t>) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_ONE) - 1.0f) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(MAG_E5M10_ZERO) - 0.0f) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    for (std::uint16_t i = 0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(mag_e5m10_t{.bits = i})).bits);
    }
}

TEST(cpu_convert_e5m10, e8m23_to_e5m10) {
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<e8m23_t>).bits, MAG_E5M10_E.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::pi_v<e8m23_t>).bits, MAG_E5M10_PI.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, MAG_E5M10_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(-1.0f).bits, MAG_E5M10_NEG_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, MAG_E5M10_ZERO.bits);

    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<e8m23_t>).bits, *test::e5m10_t{std::numbers::e_v<e8m23_t>});
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::pi_v<e8m23_t>).bits, *test::e5m10_t{std::numbers::pi_v<e8m23_t>});
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, *test::e5m10_t{1.0f});
    ASSERT_EQ(mag_e8m23_cvt_e5m10(-1.0f).bits, *test::e5m10_t{-1.0f});
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, *test::e5m10_t{0.0f});

    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::e_v<e8m23_t>)) -
                         std::numbers::e_v<e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::pi_v<e8m23_t>)) -
                         std::numbers::pi_v<e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(std::numbers::ln2_v<e8m23_t>)) -
                         std::numbers::ln2_v<e8m23_t>) < mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(1.0f)) - 1.0f) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_TRUE(std::abs(mag_e5m10_cvt_e8m23(mag_e8m23_cvt_e5m10(0.0f)) - 0.0f) <
                mag_e5m10_cvt_e8m23(MAG_E5M10_EPS));
    ASSERT_EQ(mag_e8m23_cvt_e5m10(1.0f).bits, MAG_E5M10_ONE.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(0.0f).bits, MAG_E5M10_ZERO.bits);
    ASSERT_EQ(mag_e8m23_cvt_e5m10(std::numbers::e_v<e8m23_t>).bits, MAG_E5M10_E.bits);
    for (std::uint16_t i = 0; i < 0xfff; ++i) {
        ASSERT_EQ(i, mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(mag_e5m10_t{.bits = i})).bits);
    }
}
// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

// Some bruteforce exhaustive "inductive" proofs that the invariant division impl is correct for all x and y in {0, ..., (2^32)-1} \ {0} for y

TEST(ivdiv_proof, proof_div32) {
    for (std::uint32_t x {}; x < std::numeric_limits<std::uint32_t>::max(); ++x) {
        for (std::uint32_t y {1}; y < std::numeric_limits<std::uint32_t>::max(); ++y) {
            if ((x / y) != mag_ivdiv32(x, y, mag_ivdiv_mkdi(y))) [[unlikely]] {
                std::cout << x << " / " << y << std::endl;
                ASSERT_EQ((x / y), mag_ivdiv32(x, y, mag_ivdiv_mkdi(y)));
            }
        }
    }
}

TEST(ivdiv_proof, proof_rem32) {
    for (std::uint32_t x {}; x < std::numeric_limits<std::uint32_t>::max(); ++x) {
        for (std::uint32_t y {1}; y < std::numeric_limits<std::uint32_t>::max(); ++y) {
            if ((x % y) != mag_ivrem32(x, y, mag_ivdiv_mkdi(y))) [[unlikely]] {
                std::cout << x << " / " << y << std::endl;
                ASSERT_EQ((x % y), mag_ivrem32(x, y, mag_ivdiv_mkdi(y)));
            }
        }
    }
}

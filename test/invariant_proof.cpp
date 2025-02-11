// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

/*
** Some bruteforce exhaustive "inductive" proofs that the invariant division impl is correct for
** all x and y in {0, ..., (2^32)-1} \ {0} for y
*/

#include <magnetron_internal.h>

#include <cstdint>
#include <limits>
#include <iostream>
#include <thread>
#include <vector>

static constexpr std::uint32_t lim = std::numeric_limits<std::uint32_t>::max();

static auto proof_div(std::uint32_t start, std::uint32_t end) -> void {
    for (std::uint32_t x = start; x < end; ++x) {
        for (std::uint32_t y = 1; y < lim/4; ++y) {
            if ((x / y) != mag_ivdiv32(x, y, mag_ivdiv_mkdi(y))) [[unlikely]] {
                std::cout << x << " / " << y << std::endl;
                std::abort();
            }
        }
    }
}

static auto proof_rem(std::uint32_t start, std::uint32_t end) -> void {
    for (std::uint32_t x = start; x < end; ++x) {
        for (std::uint32_t y = 1; y < lim; ++y) {
            if ((x % y) != mag_ivrem32(x, y, mag_ivdiv_mkdi(y))) [[unlikely]] {
                std::cout << x << " % " << y << std::endl;
                std::abort();
            }
        }
    }
}

auto main() -> int {
    std::uint32_t nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads {};
    std::uint32_t chunk = lim / nt;
    for (std::uint32_t i = 0; i < nt; ++i) {
        std::uint32_t start = i*chunk;
        std::uint32_t end = (i == nt-1) ? lim : start + chunk;
        threads.emplace_back([=] {
            proof_div(start, end);
            proof_rem(start, end);
        });
    }
    for (auto&& t : threads) t.join();
    std::cout << "Proof successful!" << std::endl;
    return EXIT_SUCCESS;
}

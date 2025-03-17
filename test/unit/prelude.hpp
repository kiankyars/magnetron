#pragma once

#include <gtest/gtest.h>
#include <magnetron_internal.h>

#include <bit>
#include <cstdint>
#include <cstring>

inline auto mag_tensor_buf_e8m23_to_vec(const mag_tensor_t* tensor, std::vector<float>& out) -> void {
    out.clear();
    out.reserve(mag_tensor_numel(tensor));
    std::memcpy(out.data(), mag_tensor_data_ptr(tensor), mag_tensor_data_size(tensor));
}

[[nodiscard]] constexpr auto mag_e5m10_to_e8m23_ref(const mag_e5m10_t x) noexcept -> mag_e8m23_t {
    std::uint32_t w = static_cast<std::uint32_t>(x.bits)<<16;
    std::uint32_t sign = w & 0x80000000u;
    std::uint32_t two_w = w+w;
    std::uint32_t offs = 0xe0u<<23;
    std::uint32_t t1 = (two_w>>4) + offs;
    std::uint32_t t2 = (two_w>>17) | (126u<<23);
    mag_e8m23_t norm_x = std::bit_cast<mag_e8m23_t>(t1)*0x1.0p-112f;
    mag_e8m23_t denorm_x = std::bit_cast<mag_e8m23_t>(t2)-0.5f;
    std::uint32_t denorm_cutoff = 1u<<27;
    std::uint32_t r = sign | (two_w < denorm_cutoff
        ? std::bit_cast<std::uint32_t>(denorm_x)
        : std::bit_cast<std::uint32_t>(norm_x));
    return std::bit_cast<mag_e8m23_t>(r);
}

[[nodiscard]] constexpr auto mag_e8m23_to_e5m10_ref(const mag_e8m23_t x) noexcept -> mag_e5m10_t {
    mag_e8m23_t base = fabs(x)*0x1.0p+112f*0x1.0p-110f;
    std::uint32_t shl1_w = std::bit_cast<std::uint32_t>(x)+std::bit_cast<std::uint32_t>(x);
    std::uint32_t sign = std::bit_cast<std::uint32_t>(x) & 0x80000000u;
    mag_e8m23_t flex = base + std::bit_cast<mag_e8m23_t>(0x07800000u+(mag_xmax(0x71000000u, shl1_w&0xff000000u)>>1));
    std::uint32_t exp_bits = std::bit_cast<std::uint32_t>(flex)>>13 & 0x00007c00u;
    std::uint32_t mant_bits = std::bit_cast<std::uint32_t>(flex) & 0x00000fffu;
    std::uint32_t nonsign = exp_bits + mant_bits;
    return mag_e5m10_t{.bits=static_cast<std::uint16_t>((sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign))};
}

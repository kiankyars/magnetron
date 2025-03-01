#pragma once

#include <magnetron_internal.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>

inline auto mag_tensor_buf_e8m23_to_vec(const mag_tensor_t* tensor, std::vector<float>& out) -> void {
    out.clear();
    out.reserve(mag_tensor_numel(tensor));
    std::memcpy(out.data(), mag_tensor_data_ptr(tensor), mag_tensor_data_size(tensor));
}

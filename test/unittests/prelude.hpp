// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <bit>
#include <cstdint>
#include <random>

#include <magnetron/magnetron.hpp>
#include <magnetron_internal.h>

#include <gtest/gtest.h>
#include <half.hpp>

namespace magnetron::test {
    using e8m23_t = float;
    using e5m10_t = half_float::half;

    template <typename T>
    struct dtype_traits final {
        static constexpr T min {std::numeric_limits<T>::min()};
        static constexpr T max {std::numeric_limits<T>::min()};
        static constexpr e8m23_t eps {std::numeric_limits<T>::epsilon()};
        static inline const e8m23_t test_eps {std::numeric_limits<T>::epsilon()};
    };

    [[nodiscard]] inline auto shape_to_string(std::span<const std::int64_t> shape) -> std::string {
        std::stringstream ss {};
        ss << "(";
        for (std::size_t i {}; i < shape.size(); ++i) {
            ss << shape[i];
            if (i != shape.size() - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        return ss.str();
    }

    inline thread_local std::random_device rd {};
    inline thread_local std::mt19937 gen {rd()};

    template <typename F> requires std::is_invocable_v<F, std::span<const std::int64_t>>
    auto for_all_shape_perms(std::int64_t lim, std::int64_t fac, F&& f) -> void {
        assert(lim > 0);
        ++lim;
        std::vector<std::int64_t> shape {};
        shape.reserve(k_max_dims);
        for (std::int64_t i0 = 1; i0 < lim; ++i0) {
            for (std::int64_t i1 = 0; i1 < lim; ++i1) {
                for (std::int64_t i2 = 0; i2 < lim; ++i2) {
                    for (std::int64_t i3 = 0; i3 < lim; ++i3) {
                        for (std::int64_t i4 = 0; i4 < lim; ++i4) {
                            for (std::int64_t i5 = 0; i5 < lim; ++i5) {
                                shape.clear();
                                if (i0 > 0) shape.emplace_back(i0*fac);
                                if (i1 > 0) shape.emplace_back(i1*fac);
                                if (i2 > 0) shape.emplace_back(i2*fac);
                                if (i3 > 0) shape.emplace_back(i3*fac);
                                if (i4 > 0) shape.emplace_back(i4*fac);
                                if (i5 > 0) shape.emplace_back(i5*fac);
                                f(std::span{shape});
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool BROADCAST, bool INPLACE, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, e8m23_t, e8m23_t>
    auto test_binary_operator(std::int64_t lim, e8m23_t eps, dtype ty, A&& a, B&& b, e8m23_t min = -10.0, e8m23_t max = 10.0) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape) {
            tensor t_a {ctx, ty, shape};
            t_a.fill_rand_uniform(min, max);
            tensor t_b {t_a.clone()};
            std::vector<e8m23_t> d_a {t_a.to_vector()};
            std::vector<e8m23_t> d_b {t_b.to_vector()};
            tensor t_r {std::invoke(a, t_a, t_b)};
            if constexpr (INPLACE) {
                ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
            } else {
                ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            }
            std::vector<e8m23_t> d_r {t_r.to_vector()};
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), d_r.size());
            ASSERT_EQ(t_a.dtype(), t_b.dtype());
            ASSERT_EQ(t_a.dtype(), t_r.dtype());
            for (std::int64_t i = 0; i < d_r.size(); ++i) {
                ASSERT_NEAR(std::invoke(b, d_a[i], d_b[i]), d_r[i], eps);
            }
        });
    }

    template <typename T>
    [[nodiscard]] auto compute_mean(std::span<const T> data) -> mag_e11m52_t {
        mag_e11m52_t sum = 0.0;
        for (const T x : data) sum += x;
        return sum / static_cast<mag_e11m52_t>(data.size());
    }

    template <typename T>
    [[nodiscard]] auto compute_mean(const mag_tensor_t* tensor) -> mag_e11m52_t {
        return compute_mean(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }

    template <typename T>
    [[nodiscard]] auto compute_std(std::span<const T> data) -> mag_e11m52_t {
        mag_e11m52_t sum = 0.0;
        mag_e11m52_t mean = compute_mean(data);
        for (const T x : data) {
            sum += std::pow((x - mean), 2.0);
        }
        return std::sqrt(sum / static_cast<mag_e11m52_t>(data.size()));
    }

    template <typename T>
    [[nodiscard]] auto compute_std(const mag_tensor_t* tensor) -> mag_e11m52_t {
        return compute_std(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }
}

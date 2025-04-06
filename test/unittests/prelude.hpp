// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <bit>
#include <cstdint>
#include <random>

#include <magnetron/magnetron.hpp>
#include <magnetron_internal.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "prelude.hpp"
using namespace testing;

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

    template <>
    struct dtype_traits<e5m10_t> final {
        static constexpr e5m10_t min {std::numeric_limits<e5m10_t>::min()};
        static constexpr e5m10_t max {std::numeric_limits<e5m10_t>::min()};
        static inline const e8m23_t eps {std::numeric_limits<e5m10_t>::epsilon()};
        static inline const e8m23_t test_eps {std::numeric_limits<e5m10_t>::epsilon()+0.04f}; // We increase the epsilon for f16 a little, as multiplication fails if not
    };

    [[nodiscard]] inline auto shape_as_vec(tensor t) -> std::vector<std::int64_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->shape), std::end(internal->shape)};
    }

    [[nodiscard]] inline auto strides_as_vec(tensor t) -> std::vector<std::int64_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->strides), std::end(internal->strides)};
    }

    [[nodiscard]] inline auto op_inputs_as_vec(tensor t) -> std::vector<mag_tensor_t*> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->op_inputs), std::end(internal->op_inputs)};
    }

    [[nodiscard]] inline auto op_params_as_vec(tensor t) -> std::vector<mag_opp_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->op_params), std::end(internal->op_params)};
    }

    [[nodiscard]] inline auto init_op_params_as_vec(tensor t) -> std::vector<mag_opp_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->init_op_params), std::end(internal->init_op_params)};
    }

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
                                shape.reserve(k_max_dims);
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
        ctx.stop_grad_recorder();
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
    [[nodiscard]] auto compute_mean(std::span<const T> data) -> mag_e8m23_t {
        mag_e8m23_t sum {};
        for (const T x : data) sum += x;
        return sum / static_cast<mag_e8m23_t>(data.size());
    }

    template <typename T>
    [[nodiscard]] auto compute_mean(const mag_tensor_t* tensor) -> mag_e8m23_t {
        return compute_mean(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_get_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }

    template <typename T>
    [[nodiscard]] auto compute_std(std::span<const T> data) -> mag_e11m52_t {
        mag_e8m23_t sum {};
        mag_e8m23_t mean {compute_mean(data)};
        for (const T x : data) {
            sum += std::pow(x-mean, 2.0f);
        }
        return std::sqrt(sum / static_cast<mag_e8m23_t>(data.size()));
    }

    template <typename T>
    [[nodiscard]] auto compute_std(const mag_tensor_t* tensor) -> mag_e11m52_t {
        return compute_std(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_get_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }

    // Tiny module-like library for testing training and models
    namespace nn {
        // Base class for all modules
        class module {
        public:
            module(const module&) = delete;
            module(module&&) = delete;
            auto operator=(const module&) -> module& = delete;
            auto operator=(module&&) -> module& = delete;
            virtual ~module() = default;

            [[nodiscard]] auto params() noexcept -> std::span<tensor> { return m_params; }

        protected:
            module() = default;

            auto register_param(tensor param) -> void {
                param.requires_grad(true);
                m_params.emplace_back(param);
            }

            auto register_params(std::span<tensor> params) -> void {
                for (auto param : params)
                    register_param(param);
            }

        private:
            std::vector<tensor> m_params {};
        };

        // Base class for all optimizers
        class optimizer {
        public:
            optimizer(const optimizer&) = delete;
            optimizer(optimizer&&) = delete;
            auto operator=(const optimizer&) -> optimizer& = delete;
            auto operator=(optimizer&&) -> optimizer& = delete;
            virtual ~optimizer() = default;

            virtual auto step() -> void = 0;

            [[nodiscard]] auto params() noexcept -> std::span<tensor> { return m_params; }
            auto set_params(std::span<tensor> params) -> void {
                m_params = params;
            }

            auto zero_grad() -> void {
                for (auto param : params()) {
                   param.zero_grad();
                }
            }

            [[nodiscard]] static auto mse(tensor y_hat, tensor y) -> tensor {
                tensor delta {y_hat - y};
                return (delta*delta).mean();
            }

        protected:
            explicit optimizer(std::span<tensor> params) : m_params{params} {}

        private:
            std::span<tensor> m_params{};
        };

        // Stochastic Gradient Descent optimizer
        class sgd final : public optimizer {
        public:
            explicit sgd(std::span<tensor> params, e8m23_t lr) : optimizer{params}, lr{lr} {}

            auto step() -> void override {
                for (auto& param : params()) {
                    auto grad {param.grad()};
                    if (!grad.has_value()) [[unlikely]] {
                        throw std::runtime_error("Parameter has no gradient");
                    }
                    tensor delta {param - *param.grad()*lr};
                    delta.requires_grad(true);
                    param.copy_buffer_from(delta.data_ptr(), delta.data_size());
                }
            }

            e8m23_t lr {};
        };

        // Linear/Dense layer
        class linear_layer final : public module {
        public:
            linear_layer(context& ctx, std::int64_t in_features, std::int64_t out_features, bool has_bias = true) {
                tensor weight {ctx, dtype::e8m23, out_features, in_features};
                weight.set_name("weight");
                weight.fill_rand_normal(0.0f, 1.0f);
                weight = weight/static_cast<e8m23_t>(std::sqrt(in_features + out_features));
                register_param(weight);
                this->weight = weight;
                if (has_bias) {
                    tensor bias {ctx, dtype::e8m23, out_features};
                    bias.fill(0);
                    bias.set_name("bias");
                    register_param(bias);
                    this->bias = bias;
                }
            }

            [[nodiscard]] auto operator()(tensor x) const -> tensor {
                tensor y {x & weight->T().clone()};
                if (bias.has_value())
                    y = y + *bias;
                return y;
            }

            std::optional<tensor> weight {};
            std::optional<tensor> bias {};
        };
    }
}

#include "prelude.hpp"

class module {
public:
    module(const module&) = delete;
    module(module&&) = delete;
    auto operator =(const module&) -> module& = delete;
    auto operator =(module&&) -> module& = delete;
    virtual ~module() = default;

    [[nodiscard]] auto params() noexcept -> std::span<mag_tensor_t*> { return m_params; }

protected:
    module() = default;

    auto register_param(mag_tensor_t* param) -> void {
        mag_tensor_set_requires_grad(param, true);
        m_params.emplace_back(param);
    }

    auto register_params(std::span<mag_tensor_t*> params) -> void {
       for (auto* param : params)
           register_param(param);
    }

private:
    std::vector<mag_tensor_t*> m_params {};
};

class optimizer {
public:
    optimizer(const optimizer&) = delete;
    optimizer(optimizer&&) = delete;
    auto operator =(const optimizer&) -> optimizer& = delete;
    auto operator =(optimizer&&) -> optimizer& = delete;
    virtual ~optimizer() = default;

    virtual auto step() -> void = 0;

    [[nodiscard]] auto params() noexcept -> std::span<mag_tensor_t*> { return m_params; }

    auto zero_grad() -> void {
        for (auto* param : params()) {
            mag_tensor_zero_grad(param);
        }
    }

    [[nodiscard]] static auto mse(mag_tensor_t* y_hat, mag_tensor_t* y) -> mag_tensor_t* {
        mag_tensor_t* delta = mag_sub(y_hat, y);
        mag_tensor_t* loss = mag_mul(delta, delta);
        return mag_mean(loss);
    }

protected:
    explicit optimizer(std::span<mag_tensor_t*> params) : m_params{params} {}

private:
    std::span<mag_tensor_t*> m_params {};
};

class sgd final : public optimizer {
public:
    explicit sgd(std::span<mag_tensor_t*> params, float lr) : optimizer{params}, lr{lr} {}

    auto step() -> void override {
        for (auto*& param : params()) {
            param = mag_sub(param, mag_muls(mag_tensor_grad(param), lr));
        }
    }

    float lr {};
};

class linear_layer final : public module {
public:
    linear_layer(
        mag_ctx_t* ctx,
        std::int64_t in_features,
        std::int64_t out_features,
        bool has_bias = true
    ) {
        weight = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, out_features, in_features);
        mag_tensor_fill_random_normal(weight, 0.0f, 1.0f);
        weight = mag_divs_(weight, static_cast<float>(std::sqrt(in_features+out_features)));
        register_param(weight);
        if (has_bias) {
            bias = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, out_features);
            register_param(bias);
            mag_tensor_fill(this->bias, 0.0f);
        }
    }

    [[nodiscard]] auto operator ()(mag_tensor_t* x) const -> mag_tensor_t* {
        mag_tensor_t* y = mag_matmul(x, mag_clone(mag_transpose(weight)));
        if (bias) y = mag_add(y, bias);
        return y;
    }

    mag_tensor_t* weight = nullptr;
    mag_tensor_t* bias = nullptr;
};

TEST(network, full_xor_model) {
    class xor_network final : public module {
    public:
        explicit xor_network(mag_ctx_t* ctx) : l1{ctx, 2, 2}, l2 {ctx, 2, 1} {
            register_params(l1.params());
            register_params(l2.params());
        }

        [[nodiscard]] auto operator ()(mag_tensor_t* x) const -> mag_tensor_t* {
            auto* y = mag_tanh(l1(x));
            y = mag_tanh(l2(y));
            return y;
        }

    private:
        linear_layer l1;
        linear_layer l2;
    };

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    xor_network model {ctx};
    sgd optimizer {model.params(), 0.1f};

    static constexpr std::array<std::array<float, 2>, 4> x_data = {
        { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} }
    };
    static constexpr std::array<std::array<float, 1>, 4> y_data = {
        { {0.0f}, {1.0f}, {1.0f}, {0.0f} }
    };

    mag_tensor_t* x = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, x_data.size(), x_data[0].size());
    mag_tensor_copy_buffer_from(x, x_data.data(), sizeof(x_data));

    mag_tensor_t* y = mag_tensor_create_2d(ctx, MAG_DTYPE_E8M23, y_data.size(), y_data[0].size());
    mag_tensor_copy_buffer_from(y, y_data.data(), sizeof(y_data));

    constexpr std::int64_t epochs = 3;
    for (std::int64_t epoch = 0; epoch < epochs; ++epoch) {
        mag_tensor_t* y_hat = model(x);
        mag_tensor_t* loss = optimizer::mse(y_hat, y);
        mag_tensor_backward(loss);
        optimizer.step();
        optimizer.zero_grad();
    }

    mag_ctx_destroy(ctx);
}


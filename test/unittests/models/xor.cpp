// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

class xor_model final : public nn::module {
public:
    explicit xor_model(context& ctx)
        : l1{ctx, 2, 2}, l2{ctx, 2, 1} {
        register_params(l1.params());
        register_params(l2.params());
    }

    [[nodiscard]] auto operator()(tensor x) const -> tensor {
        tensor y {l1(x).tanh()};
        y = l2(y).tanh();
        return y;
    }

private:
    nn::linear_layer l1;
    nn::linear_layer l2;
};

TEST(models, xor) {
    context ctx {compute_device::cpu};
    xor_model model{ctx};
    nn::sgd optimizer{model.params(), 0.1f};

    static constexpr std::array<std::array<mag_e8m23_t, 2>, 4> x_data {
        {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}}
    };
    static constexpr std::array<std::array<mag_e8m23_t, 1>, 4> y_data {
        {{0.0f}, {1.0f}, {1.0f}, {0.0f}}
    };

    tensor x {ctx, dtype::e8m23, x_data.size(), x_data[0].size()};
    x.copy_buffer_from(&x_data, sizeof(x_data));

    tensor y {ctx, dtype::e8m23, y_data.size(), y_data[0].size()};
    y.copy_buffer_from(&y_data, sizeof(y_data));

    constexpr std::int64_t epochs {2000};
    for (std::int64_t epoch = 0; epoch < epochs; ++epoch) {
        tensor y_hat {model(x)};
        tensor loss {nn::optimizer::mse(y_hat, y)};
        loss.backward();
        //loss.export_graphviz_forward("fwd_" + std::to_string(epoch) + ".dot");
        //loss.export_graphviz_backward("bwd_" + std::to_string(epoch) + ".dot");
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss(0) << std::endl;
        }
        optimizer.step();
        optimizer.zero_grad();
    }

    tensor y_hat {model(x)};

    std::vector<e8m23_t> output {y_hat.to_vector()};
    for (auto r : output) {
        std::cout << r << " ";
    }
    std::cout << std::endl;
    for (auto r : y_data) {
        std::cout << r[0] << " ";
    }
}

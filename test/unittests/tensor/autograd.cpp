// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

TEST(cpu_autograd, scalar_simple) {
    context ctx {compute_device::cpu};
    tensor x {ctx, dtype::e8m23, 1};
    x.fill(3.0f);
    tensor y {ctx, dtype::e8m23, 1};
    y.fill(2.0f);
    tensor k {ctx, dtype::e8m23, 1};
    k.fill(10.0f);

    tensor z {(x + y)*(x - y)/k};
    z.backward();

    ASSERT_TRUE(x.grad().has_value());
    ASSERT_TRUE(y.grad().has_value());
    ASSERT_TRUE(k.grad().has_value());
    ASSERT_TRUE(z.grad().has_value());

    // check forward pass
    ASSERT_FLOAT_EQ(x(0), 3.0f);
    ASSERT_FLOAT_EQ(y(0), 2.0f);
    ASSERT_FLOAT_EQ(z(0), 0.5f);

    // check backward pass
    ASSERT_FLOAT_EQ(x.grad().value()(0), 0.6f);     // ∂z/∂x = 0.6
    ASSERT_FLOAT_EQ(y.grad().value()(0), -0.4f);    // ∂z/∂y = -0.4
    ASSERT_FLOAT_EQ(z.grad().value()(0), 1.0f);     // ∂z/∂z = 1.0
}

TEST(cpu_autograd, scalar_complex) {
    context ctx {compute_device::cpu};
    tensor two {ctx, dtype::e8m23, 1};
    two.fill(2.0f);
    tensor x {ctx, dtype::e8m23, 1};
    x.fill(-4.0f);
    tensor z {two*x+two+x};
    tensor q {z.relu()+z*x};
    tensor h {(z*z).relu()};
    tensor y {h+q+q*x};
    y.backward();

    ASSERT_TRUE(two.grad().has_value());
    ASSERT_TRUE(x.grad().has_value());
    ASSERT_TRUE(z.grad().has_value());
    ASSERT_TRUE(q.grad().has_value());
    ASSERT_TRUE(h.grad().has_value());
    ASSERT_TRUE(y.grad().has_value());

    // check forward pass
    ASSERT_FLOAT_EQ(x(0), -4.0f);
    ASSERT_FLOAT_EQ(y(0), -20.0f);

    // check backward pass
    ASSERT_FLOAT_EQ(x.grad().value()(0), 46.0f);    // ∂z/∂x = 46.0
    ASSERT_FLOAT_EQ(y.grad().value()(0), 1.0f);     // ∂z/∂y = 1.0
}

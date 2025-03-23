/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "prelude.hpp"

TEST(autograd, bin_ops1) {
    /*
     * x = 3.0
     * y = 2.0
     * z = (x + y) * (x - y)
     * z.backward()
     */

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    auto* x = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_set_requires_grad(x, true);
    mag_tensor_fill(x, 3.0f);

    auto* y = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_fill(y, 2.0f);
    mag_tensor_set_requires_grad(y, true);

    auto* k = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_fill(k, 10.0f);
    mag_tensor_set_requires_grad(k, true);

    auto* z = mag_div(mag_mul(mag_add(x, y), mag_sub(x, y)), k);
    mag_tensor_set_requires_grad(z, true);
    mag_tensor_backward(z);

    ASSERT_NE(mag_tensor_grad(x), nullptr);
    ASSERT_NE(mag_tensor_grad(y), nullptr);
    ASSERT_NE(mag_tensor_grad(k), nullptr);
    ASSERT_NE(mag_tensor_grad(z), nullptr);

    // check forward pass
    mag_e8m23_t vx = mag_tensor_get_scalar_virtual_index(x, 0);
    mag_e8m23_t vy = mag_tensor_get_scalar_virtual_index(y, 0);
    mag_e8m23_t vz = mag_tensor_get_scalar_virtual_index(z, 0);
    ASSERT_FLOAT_EQ(vx, 3.0f);
    ASSERT_FLOAT_EQ(vy, 2.0f);
    ASSERT_EQ(vz, 0.5f);

    // check backward pass
    mag_e8m23_t gx = mag_tensor_get_scalar_virtual_index(mag_tensor_grad(x), 0);
    mag_e8m23_t gy = mag_tensor_get_scalar_virtual_index(mag_tensor_grad(y), 0);
    mag_e8m23_t gz = mag_tensor_get_scalar_virtual_index(mag_tensor_grad(z), 0);

    ASSERT_FLOAT_EQ(gx, 0.6f);  // ∂z/∂x = 0.6
    ASSERT_FLOAT_EQ(gy, -0.4f); // ∂z/∂y = -0.4
    ASSERT_FLOAT_EQ(gz, 1.0f);  // ∂z/∂z = 1.0

    mag_tensor_decref(x);
    mag_tensor_decref(y);
    mag_tensor_decref(z);
    mag_ctx_destroy(ctx);
}

TEST(autograd, bin_ops2) {
    /*
     * x = -4.0
     * z = 2 * x + 2 + x
     * q = z.relu() + z * x
     * h = (z * z).relu()
     * y = h + q + q * x
     * y.backward()
     */

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    auto* two = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_set_requires_grad(two, true);
    mag_tensor_fill(two, 2.0f);

    auto* x = mag_tensor_create_1d(ctx, MAG_DTYPE_E8M23, 1);
    mag_tensor_set_requires_grad(x, true);
    mag_tensor_fill(x, -4.0f);

    auto* z = mag_add(mag_add(mag_mul(two, x), two), x);
    auto* q = mag_add(mag_relu(z), mag_mul(z, x));
    auto* h = mag_relu(mag_mul(z, z));
    auto* y = mag_add(q, mag_add(mag_mul(q, x), h));
    mag_tensor_set_requires_grad(y, true);
    mag_tensor_backward(y);

    // mag_tensor_export_graphviz(y, "autograd2.dot");

    // check forward pass
    mag_e8m23_t vx = mag_tensor_get_scalar_virtual_index(x, 0);
    mag_e8m23_t vy = mag_tensor_get_scalar_virtual_index(y, 0);
    ASSERT_FLOAT_EQ(vx, -4.0f);
    ASSERT_FLOAT_EQ(vy, -20.0f);

    ASSERT_NE(mag_tensor_grad(x), nullptr);
    ASSERT_NE(mag_tensor_grad(y), nullptr);
    ASSERT_NE(mag_tensor_grad(z), nullptr);
    ASSERT_NE(mag_tensor_grad(q), nullptr);
    ASSERT_NE(mag_tensor_grad(h), nullptr);

    // check backward pass
    mag_e8m23_t gx = mag_tensor_get_scalar_virtual_index(mag_tensor_grad(x), 0);
    mag_e8m23_t gy = mag_tensor_get_scalar_virtual_index(mag_tensor_grad(y), 0);

    ASSERT_FLOAT_EQ(gx, 46.0f); // ∂z/∂x = 46.0
    ASSERT_FLOAT_EQ(gy, 1.0f);  // ∂z/∂y = 1.0

    mag_tensor_decref(x);
    mag_tensor_decref(y);
    mag_tensor_decref(z);
    mag_ctx_destroy(ctx);
}

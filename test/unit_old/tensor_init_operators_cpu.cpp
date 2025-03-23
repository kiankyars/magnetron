// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// For some tests, we use a larger absolute error than machine epsilon,
// because the BLAS uses SIMD for certain functions which have higher accuracy than the scalar high-precision lambdas.

#include "prelude.hpp"
#include <algorithm>
#include <cmath>
#include <random>

static constexpr std::int64_t k_lim_same_shape = 6;

TEST(init_operators_cpu, broadcast_e8m23) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> dist {-10.0f, 10.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
                            mag_e8m23_t filled = dist(gen);
                            mag_tensor_fill(x, filled);
                            const auto* buf = static_cast<const mag_e8m23_t*>(mag_tensor_data_ptr(x));
                            auto numel = mag_tensor_numel(x);
                            for (std::int64_t i = 0; i < numel; ++i) {
                                ASSERT_FLOAT_EQ(buf[i], filled);
                            }
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}

TEST(init_operators_cpu, broadcast_e5m10) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> dist {-10.0f, 10.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E5M10, i0, i1, i2, i3, i4, i5);
                            mag_e8m23_t filled = dist(gen);
                            mag_tensor_fill(x, filled);
                            const auto* buf = static_cast<const mag_e5m10_t*>(mag_tensor_data_ptr(x));
                            auto numel = mag_tensor_numel(x);
                            for (std::int64_t i = 0; i < numel; ++i) {
                                ASSERT_EQ(buf[i].bits, mag_e8m23_to_e5m10_ref(filled).bits);
                            }
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}

TEST(init_operators_cpu, random_uniform_e8m23) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> dist {-10.0f, 10.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
                            mag_e8m23_t min = dist(gen);
                            mag_e8m23_t max = std::uniform_real_distribution<mag_e8m23_t>(min, dist.max() + 10.0f)(gen);
                            mag_tensor_fill_random_uniform(x, min, max);
                            const auto* buf = static_cast<const mag_e8m23_t*>(mag_tensor_data_ptr(x));
                            auto numel = mag_tensor_numel(x);
                            for (std::int64_t i = 0; i < numel; ++i) {
                                ASSERT_GE(buf[i], min);
                                ASSERT_LE(buf[i], max);
                            }
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}

TEST(init_operators_cpu, random_uniform_e5m10) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> dist {-10.0f, 10.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E5M10, i0, i1, i2, i3, i4, i5);
                            mag_e8m23_t min = dist(gen);
                            mag_e8m23_t max = std::uniform_real_distribution<mag_e8m23_t>(min, dist.max() + 10.0f)(gen);
                            mag_tensor_fill_random_uniform(x, min, max);
                            const auto* buf = static_cast<const mag_e5m10_t*>(mag_tensor_data_ptr(x));
                            auto numel = mag_tensor_numel(x);
                            for (std::int64_t i = 0; i < numel; ++i) {
                                ASSERT_GE(mag_e5m10_to_e8m23_ref(buf[i]), mag_e5m10_to_e8m23_ref(mag_e8m23_to_e5m10_ref(min)));
                                ASSERT_LE(mag_e5m10_to_e8m23_ref(buf[i]), mag_e5m10_to_e8m23_ref(mag_e8m23_to_e5m10_ref(max)));
                            }
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}

#if 0 // TODO

TEST(init_operators_cpu, random_normal_e8m23) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> mean_dist {-5.0f, 5.0f};
    std::uniform_real_distribution<mag_e8m23_t> stddev_dist {0.5f, 5.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 1; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 1; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 1; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 1; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 1; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 1; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E8M23, i0, i1, i2, i3, i4, i5);
                            mag_e8m23_t mean {mean_dist(gen)};
                            mag_e8m23_t std {stddev_dist(gen)};
                            mag_tensor_fill_random_normal(x, mean, std);
                            ASSERT_NEAR(mean, compute_mean<mag_e8m23_t>(x), 1e-4);
                            ASSERT_NEAR(std, compute_std<mag_e8m23_t>(x), 1e-4);
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}

TEST(init_operators_cpu, random_normal_e5m10) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<mag_e8m23_t> mean_dist {-5.0f, 5.0f};
    std::uniform_real_distribution<mag_e8m23_t> stddev_dist {0.5f, 5.0f};
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);
    for (std::int64_t i0 = 2; i0 <= k_lim_same_shape; ++i0) {
        for (std::int64_t i1 = 2; i1 <= k_lim_same_shape; ++i1) {
            for (std::int64_t i2 = 2; i2 <= k_lim_same_shape; ++i2) {
                for (std::int64_t i3 = 2; i3 <= k_lim_same_shape; ++i3) {
                    for (std::int64_t i4 = 2; i4 <= k_lim_same_shape; ++i4) {
                        for (std::int64_t i5 = 2; i5 <= k_lim_same_shape; ++i5) {
                            mag_tensor_t* x = mag_tensor_create_6d(ctx, MAG_DTYPE_E5M10, i0&~1, i1&~1, i2&~1, i3&~1, i4&~1, i5&~1);
                            mag_e8m23_t mean {mean_dist(gen)};
                            mag_e8m23_t std {stddev_dist(gen)};
                            mag_tensor_fill_random_normal(x, mean, std);
                            const auto* buf = static_cast<const mag_e5m10_t*>(mag_tensor_data_ptr(x));
                            auto numel = mag_tensor_numel(x);
                            mag_e11m52_t new_mean = 0.0;
                            for (std::int64_t i=0; i < numel; ++i) new_mean += mag_e5m10_to_e8m23_ref(buf[i]);
                            new_mean = new_mean / static_cast<mag_e11m52_t>(numel);
                            mag_e11m52_t new_stddev = 0.0;
                            for (std::int64_t i=0; i < numel; ++i) new_stddev += (mag_e5m10_to_e8m23_ref(buf[i]) - mean)*(mag_e5m10_to_e8m23_ref(buf[i]) - mean);
                            new_stddev = std::sqrt(new_stddev / (static_cast<mag_e11m52_t>(numel)-1));
                            ASSERT_NEAR(mean, new_mean, 1e-1);
                            ASSERT_NEAR(std, new_stddev, 1e-1);
                            mag_tensor_decref(x);
                        }
                    }
                }
            }
        }
    }
    mag_ctx_destroy(ctx);
}
#endif

// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(image, load) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* img = mag_tensor_load_image(ctx, "test_data/test_img.png", MAG_COLOR_CHANNELS_RGB, 0, 0);
    mag_tensor_print(img, true, true);
    ASSERT_EQ(mag_tensor_image_channels(img), 3);
    ASSERT_EQ(mag_tensor_image_width(img), 2);
    ASSERT_EQ(mag_tensor_image_height(img), 4);
    ASSERT_EQ(mag_tensor_shape(img)[2], mag_tensor_image_width(img));
    ASSERT_EQ(mag_tensor_shape(img)[1], mag_tensor_image_height(img));
    ASSERT_EQ(mag_tensor_shape(img)[0], mag_tensor_image_channels(img)); // RGB

    auto* buf = mag_tensor_data_ptr(img);
    for (int64_t i=0; i < mag_tensor_numel(img); ++i) {
        ASSERT_GE(static_cast<float*>(buf)[i], 0.0f);
        ASSERT_LE(static_cast<float*>(buf)[i], 1.0f);
    }

    // TODO: check data

    mag_tensor_decref(img);
    mag_ctx_destroy(ctx);
}

TEST(image, load_resize) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* img = mag_tensor_load_image(ctx, "test_data/car.jpg", MAG_COLOR_CHANNELS_RGB, 256, 211);
    ASSERT_EQ(mag_tensor_shape(img)[2], 256);
    ASSERT_EQ(mag_tensor_shape(img)[1], 211);
    ASSERT_EQ(mag_tensor_shape(img)[0], 3); // RGB
    ASSERT_EQ(mag_tensor_shape(img)[2], mag_tensor_image_width(img));
    ASSERT_EQ(mag_tensor_shape(img)[1], mag_tensor_image_height(img));
    ASSERT_EQ(mag_tensor_shape(img)[0], mag_tensor_image_channels(img)); // RGB

    // mag_tensor_save_image(img, "test_data/car_resized.jpg");

    auto* buf = static_cast<float*>(mag_tensor_data_ptr(img));
    for (int64_t i=0; i < mag_tensor_numel(img); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    mag_tensor_decref(img);

    mag_ctx_destroy(ctx);
}

TEST(image, draw_box) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* img = mag_tensor_load_image(ctx, "test_data/car.jpg", MAG_COLOR_CHANNELS_RGB, 256, 256);
    mag_tensor_img_draw_box(img, 40, 40, 80, 80, 1, mag_pack_color_f32(1.0f, 0.0f, 0.0f));
    mag_tensor_img_draw_box(img, 120, 150, 160, 200, 4, mag_pack_color_f32(1.0f, 1.0f, 1.0f));
    mag_tensor_print(img, true, false);
    //mag_tensor_save_image(img, "test_data/car2.jpg");

    mag_tensor_decref(img);
    mag_ctx_destroy(ctx);
}

TEST(image, draw_text) {
    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* img = mag_tensor_load_image(ctx, "test_data/car.jpg", MAG_COLOR_CHANNELS_RGB, 256, 256);
    mag_tensor_img_draw_text(img, 100, 100, 10, 0xffffff, "Hallö!");
    mag_tensor_img_draw_text(img, 100, 200, 15, 0xffffff, "I want Pizza Salami! I want Pizza Salami! I want Pizza Salami! I want Pizza Salami!");
    mag_tensor_print(img, true, false);
    //mag_tensor_save_image(img, "test_data/car3.jpg");

    mag_tensor_decref(img);
    mag_ctx_destroy(ctx);
}

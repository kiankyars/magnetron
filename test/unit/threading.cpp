// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <thread>
#include <atomic>
#include <chrono>

#include "prelude.hpp"

using namespace std::chrono_literals;

TEST(threading, StoreLoadTest) {
    mag_atomic_t val = 0;
    mag_atomic_store(&val, 42, MAG_MO_RELAXED);
    int loaded = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(loaded, 42);
}

TEST(threading, FetchAddTest) {
    mag_atomic_t val = 10;
    int old = mag_atomic_fetch_add(&val, 5, MAG_MO_RELAXED);
    EXPECT_EQ(old, 10);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 15);
}

TEST(threading, FetchSubTest) {
    mag_atomic_t val = 20;
    int old = mag_atomic_fetch_sub(&val, 3, MAG_MO_RELAXED);
    EXPECT_EQ(old, 20);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 17);
}

TEST(threading, FetchAndTest) {
    mag_atomic_t val = 0xF0;
    int old = mag_atomic_fetch_and(&val, 0x0F, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xF0);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x00);
}

TEST(threading, FetchOrTest) {
    mag_atomic_t val = 0x0F;
    int old = mag_atomic_fetch_or(&val, 0xF0, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0x0F);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0xFF);
}

TEST(threading, FetchXorTest) {
    mag_atomic_t val = 0xAA;
    int old = mag_atomic_fetch_xor(&val, 0xFF, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xAA);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x55);
}

TEST(threading, ExchangeTest) {
    mag_atomic_t val = 100;
    int old = mag_atomic_exchange(&val, 200, MAG_MO_RELAXED);
    EXPECT_EQ(old, 100);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 200);
}

TEST(threading, CompareExchangeWeakSuccessTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 10;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 20);
    EXPECT_EQ(expected, 10);
}

TEST(threading, CompareExchangeWeakFailTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 5;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 10);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 10);
}

TEST(threading, CompareExchangeStrongSuccessTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 30;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 40);
    EXPECT_EQ(expected, 30);
}

TEST(threading, CompareExchangeStrongFailTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 25;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 30);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 30);
}
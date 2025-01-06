// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <thread>
#include <atomic>
#include <chrono>

#include "prelude.hpp"

using namespace std::chrono_literals;

TEST(threading, create_thread) {
    std::atomic<bool> executed = false;
    static volatile std::atomic_uintptr_t main_tid {mag_thread_id()};
    static volatile std::atomic_uintptr_t second_tid {};

    std::cout << "Main thread ID: " << std::hex << main_tid.load(std::memory_order_seq_cst) << std::endl;

    auto thread_func = [](void* arg) -> mag_thread_ret_t {
        mag_thread_sched_set_prio(MAG_THREAD_SCHED_PRIO_MEDIUM);
        std::this_thread::sleep_for(1s); // Simulate some work
        second_tid.store(mag_thread_id(), std::memory_order_seq_cst);
        static_cast<std::atomic<bool>*>(arg)->store(true, std::memory_order_seq_cst);
        std::cout << "Second thread ID: " << std::hex << second_tid.load(std::memory_order_seq_cst) << std::endl;
        return NULL;
    };

    mag_thread_t* thread = mag_thread_create(thread_func, &executed);
    ASSERT_NE(thread, nullptr);
    mag_thread_join(thread);
    EXPECT_EQ(true, executed.load(std::memory_order_seq_cst));
    ASSERT_NE(main_tid.load(std::memory_order_seq_cst), second_tid.load(std::memory_order_seq_cst));
}

TEST(threading, scheduler_yield) {
    std::atomic<bool> thread_started = false;
    std::atomic<bool> thread_done = false;

    auto thread_func = [](void* arg) -> mag_thread_ret_t {
        auto* state = static_cast<std::pair<std::atomic<bool>*, std::atomic<bool>*>*>(arg);
        std::atomic<bool>& started = *state->first;
        std::atomic<bool>& done = *state->second;
        started.store(true, std::memory_order_seq_cst);
        for (int i = 0; i < 10; ++i) {
            mag_thread_sched_yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        done.store(true, std::memory_order_seq_cst);
        return nullptr;
    };

    std::pair<std::atomic<bool>*, std::atomic<bool>*> thread_state {&thread_started, &thread_done};
    mag_thread_t* thread = mag_thread_create(thread_func, &thread_state);
    ASSERT_NE(thread, nullptr);

    while (!thread_started) {
        mag_thread_sched_yield();
    }
    EXPECT_FALSE(thread_done.load(std::memory_order_seq_cst));
    mag_thread_join(thread);
    EXPECT_TRUE(thread_done.load(std::memory_order_seq_cst));
}

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
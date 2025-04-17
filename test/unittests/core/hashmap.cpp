// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

class hash_map_test : public Test {
protected:
    mag_hashmap_t* map {};
    auto SetUp() -> void override {
        map = mag_hashmap_create(
            sizeof(int),
            16,
            0,
            +[](const void* key, std::uint32_t seed) -> std::uint64_t {
                return mag_hash(key, sizeof(int), seed);
            },
            +[](const void* a, const void* b, void*) -> bool {
                return *static_cast<const int*>(a) == *static_cast<const int*>(b);
            },
            nullptr,
            nullptr,
            MAG_DEF_MAP_GROW_FACTOR,
            MAG_DEF_MAP_SHRINK_FACTOR,
            MAG_DEF_MAP_LOAD_FACTOR
        );
        ASSERT_NE(map, nullptr);
    }

    auto TearDown() -> void override {
        mag_hashmap_destroy(map);
    }
};

TEST_F(hash_map_test, empty_map) {
    EXPECT_EQ(mag_hashmap_count(map), 0u);
    int key = 123;
    EXPECT_EQ(mag_hashmap_lookup(map, &key), nullptr);
    EXPECT_EQ(mag_hashmap_delete(map, &key), nullptr);
}

TEST_F(hash_map_test, insert_and_get) {
    int key = 7;
    // first insert → returns NULL
    const auto* prev = static_cast<const int*>(mag_hashmap_insert(map, &key));
    ASSERT_EQ(prev, nullptr);

    // now get should find it
    const auto* found = static_cast<const int*>(mag_hashmap_lookup(map, &key));
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, key);

    // inserting the same key again → returns non‑NULL (the replaced item)
    const auto* replaced = static_cast<const int*>(mag_hashmap_insert(map, &key));
    ASSERT_NE(replaced, nullptr);
    EXPECT_EQ(*replaced, key);

    // count stayed at 1
    EXPECT_EQ(mag_hashmap_count(map), 1u);
}

TEST_F(hash_map_test, delete_and_get) {
    int a = 42, b = 99;

    // delete nonexistent → NULL
    EXPECT_EQ(mag_hashmap_delete(map, &b), nullptr);

    // insert and then delete
    mag_hashmap_insert(map, &a);
    EXPECT_EQ(mag_hashmap_count(map), 1u);

    const auto* del = static_cast<const int*>(mag_hashmap_delete(map, &a));
    ASSERT_NE(del, nullptr);
    EXPECT_EQ(*del, a);

    // now it's gone
    EXPECT_EQ(mag_hashmap_lookup(map, &a), nullptr);
    EXPECT_EQ(mag_hashmap_count(map), 0u);
}

TEST_F(hash_map_test, clear_map) {
    int vals[] = {1,2,3,4,5};
    for (int v : vals) {
        mag_hashmap_insert(map, &v);
    }
    EXPECT_EQ(mag_hashmap_count(map), 5u);

    // clear leaves capacity intact
    mag_hashmap_clear(map, false);
    EXPECT_EQ(mag_hashmap_count(map), 0u);

    // can re‑insert after clear
    int x = 7;
    EXPECT_EQ(mag_hashmap_insert(map, &x), nullptr);
    // note: first insert returns NULL
    EXPECT_EQ(mag_hashmap_count(map), 1u);
}

TEST_F(hash_map_test, scan_callback) {
    int keys[] = {10,20,30};
    for (int k : keys) mag_hashmap_insert(map, &k);

    std::vector<int> seen;
    auto cb = [](const void* item, void* ud) {
        auto vec = static_cast<std::vector<int>*>(ud);
        vec->push_back(*static_cast<const int*>(item));
        return true;
    };

    bool cont = mag_hashmap_scan(map, cb, &seen);
    EXPECT_TRUE(cont);
    std::ranges::sort(seen);
    EXPECT_EQ(seen, std::vector<int>({10,20,30}));
}

TEST_F(hash_map_test, IterationYieldsAllItems) {
    int keys[] = {11,22,33,44};
    for (int k : keys) mag_hashmap_insert(map, &k);
    EXPECT_EQ(mag_hashmap_count(map), 4u);
    size_t iter = 0;
    void* item = nullptr;
    std::vector<int> got;
    while (mag_hashmap_iter(map, &iter, &item)) {
        got.push_back(*static_cast<int*>(item));
    }
    std::ranges::sort(got);
    EXPECT_EQ(got, std::vector<int>({11,22,33,44}));
}

TEST_F(hash_map_test, auto_grow_under_pressure) {
    // default capacity = 16, load_factor = .60 → grow at 9
    std::vector<int> data;
    for (int i = 0; i < 20; ++i) {
        data.push_back(i);
        mag_hashmap_insert(map, &data.back());
    }
    // after growing, still findable
    for (int v : data) {
        auto* p = static_cast<const int*>(mag_hashmap_lookup(map, &v));
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, v);
    }
    EXPECT_EQ(mag_hashmap_count(map), data.size());
}
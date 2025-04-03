// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

TEST(op_param, pack_e8m23) {
    e8m23_t val {3.1415f};
    mag_opp_t p {mag_opp_pack_e8m23(val)};
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_E8M23);
    ASSERT_EQ(mag_opp_unpack_value(p), std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_opp_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<e8m23_t>::min();
    p = mag_opp_pack_e8m23(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_E8M23);
    ASSERT_EQ(mag_opp_unpack_value(p), std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_opp_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<e8m23_t>::max();
    p = mag_opp_pack_e8m23(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_E8M23);
    ASSERT_EQ(mag_opp_unpack_value(p), std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_opp_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_e8m23_or(p, 0.0f), val);
}

TEST(op_param, pack_i62) {
    std::int64_t val {-123456};
    mag_opp_t p {mag_opp_pack_i62(val)};
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_I62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or(p, 0), val);

    val = -(((1ll<<62)>>1)); /* min val for int62 */
    p = mag_opp_pack_i62(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_I62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or(p, 0), val);

    val = ((1ll<<62)>>1)-1; /* max val for int62 */
    p = mag_opp_pack_i62(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_I62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_i62_or(p, 0), val);
}

TEST(op_param, pack_u62) {
    std::uint64_t val {123456};
    mag_opp_t p {mag_opp_pack_u62(val)};
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_U62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or(p, 0), val);

    val = 0; /* min val for uint62 */
    p = mag_opp_pack_u62(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_U62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or(p, 0), val);

    val = (1ull<<62)-1; /* max val for uint62 */
    p = mag_opp_pack_u62(val);
    ASSERT_EQ(mag_opp_unpack_type(p), MAG_OPP_U62);
    ASSERT_EQ(mag_opp_unpack_value(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or_panic(p), val);
    ASSERT_EQ(mag_opp_unpack_u62_or(p, 0), val);
}

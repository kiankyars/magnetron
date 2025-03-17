/*
** (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
**
**
** !!! Make sure all functions in this file are static. This is required to correctly clone the impl for each specialized compilation unit.
** This file implements the core math for magnetron, optimized for different CPU instruction sets.
** This file is also included into different compilation units, which are all compiled with different architecture flags, thus the impl is 'cloned'.
** At runtime the best impl for the host-CPU is chose automatically, by detecting the CPU and querying the hardware features.
**
** !!! Minimum Requirements!!!
**  AMD 64 CPUs: SSE & SSE2 (any 64-bit AMD64 CPU).
**  ARM 64 CPUs: ARM v8-a (Raspberry Pi 4, 5, Apple M1-4, Neoverse/Graviton etc..)
**
** +==============+=============+==============+======================================================+
** | AMD 64 Versions and Features
** +==============+=============+==============+======================================================+
** | x86-64-v1	| CMOV, CX8, FPU, FXSR, MMX, OSFXSR, SCE, SSE, SSE2
** | x86-64-v2	| CMPXCHG16B, LAHF-SAHF, POPCNT, SSE3, SSE4_1, SSE4_2, SSSE3
** | x86-64-v3	| AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE
** | x86-64-v4	| AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL
** +==============+=============+==============+======================================================+
** Some CPUs fall inbetween those, for example my old rusty test server has four old AMD Opteron CPUs with 16 cores each. They support AVX but not AVX2.
** For CPUs like this, we still support more granular feature levels: SSE42, AVX, AVX2 and AVX512F.
**
** +==============+=============+==============+======================================================+
** | ARM 64 Versions and Features
** +==============+=============+==============+======================================================+
** | armv8-a      |  Armv8-A    |              |  +fp, +simd
** | armv8.1-a    |  Armv8.1-A  |  armv8-a,    |  +crc, +lse, +rdma
** | armv8.2-a    |  Armv8.2-A  |  armv8.1-a   |
** | armv8.3-a    |  Armv8.3-A  |  armv8.2-a,  |  +pauth, +fcma, +jscvt
** | armv8.4-a    |  Armv8.4-A  |  armv8.3-a,  |  +flagm, +fp16fml, +dotprod, +rcpc2
** | armv8.5-a    |  Armv8.5-A  |  armv8.4-a,  |  +sb, +ssbs, +predres, +frintts, +flagm2
** | armv8.6-a    |  Armv8.6-A  |  armv8.5-a,  |  +bf16, +i8mm
** | armv8.7-a    |  Armv8.7-A  |  armv8.6-a,  |  +wfxt, +xs
** | armv8.8-a    |  Armv8.8-a  |  armv8.7-a,  |  +mops
** | armv8.9-a    |  Armv8.9-a  |  armv8.8-a   |
** | armv9-a      |  Armv9-A    |  armv8.5-a,  |  +sve, +sve2
** | armv9.1-a    |  Armv9.1-A  |  armv9-a,    |  +bf16, +i8mm
** | armv9.2-a    |  Armv9.2-A  |  armv9.1-a   |
** | armv9.3-a    |  Armv9.3-A  |  armv9.2-a,  |  +mops
** | armv9.4-a    |  Armv9.4-A  |  armv9.3-a   |
** | armv8-r      |  Armv8-R    |  armv8-r     |
** +==============+=============+==============+======================================================+
*/

#include "magnetron_internal.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>

#define MAG_TAU (2.0f*3.14159265358979323846264338327950288f) /* τ = 2π */

#if defined(__APPLE__) && defined(MAG_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif
#ifdef MAG_OPENBLAS
#include <cblas.h>
#endif

#if defined(_MSC_VER) && defined(__AVX2__) /*MSVC does not define FMA and F16C with AVX 2*/
#define __FMA__ 1
#define __F16C__ 1
#endif

#define MAG_E5M10_E                 (mag_e5m10_t){.bits=0x4170}
#define MAG_E5M10_EPS               (mag_e5m10_t){.bits=0x1400}
#define MAG_E5M10_INF               (mag_e5m10_t){.bits=0x7c00}
#define MAG_E5M10_LN10              (mag_e5m10_t){.bits=0x409b}
#define MAG_E5M10_LN2               (mag_e5m10_t){.bits=0x398c}
#define MAG_E5M10_LOG10_2           (mag_e5m10_t){.bits=0x34d1}
#define MAG_E5M10_LOG10_E           (mag_e5m10_t){.bits=0x36f3}
#define MAG_E5M10_LOG2_10           (mag_e5m10_t){.bits=0x42a5}
#define MAG_E5M10_LOG2_E            (mag_e5m10_t){.bits=0x3dc5}
#define MAG_E5M10_MAX               (mag_e5m10_t){.bits=0x7bff}
#define MAG_E5M10_MAX_SUBNORMAL     (mag_e5m10_t){.bits=0x03ff}
#define MAG_E5M10_MIN               (mag_e5m10_t){.bits=0xfbff}
#define MAG_E5M10_MIN_POS           (mag_e5m10_t){.bits=0x0400}
#define MAG_E5M10_MIN_POS_SUBNORMAL (mag_e5m10_t){.bits=0x0001}
#define MAG_E5M10_NAN               (mag_e5m10_t){.bits=0x7e00}
#define MAG_E5M10_NEG_INF           (mag_e5m10_t){.bits=0xfc00}
#define MAG_E5M10_NEG_ONE           (mag_e5m10_t){.bits=0xbc00}
#define MAG_E5M10_NEG_ZERO          (mag_e5m10_t){.bits=0x8000}
#define MAG_E5M10_ONE               (mag_e5m10_t){.bits=0x3c00}
#define MAG_E5M10_PI                (mag_e5m10_t){.bits=0x4248}
#define MAG_E5M10_SQRT2             (mag_e5m10_t){.bits=0x3da8}
#define MAG_E5M10_ZERO              (mag_e5m10_t){.bits=0x0000}

static MAG_AINLINE mag_e5m10_t mag_e8m23_cvt_e5m10(mag_e8m23_t x) {
    uint16_t r;
    #ifdef __F16C__
        #ifdef _MSC_VER
            r = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0);
        #else
            r = _cvtss_sh(x, 0);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        __fp16 h = (__fp16)x;
        r = *(uint16_t*)&h;
    #else
        union {
            uint32_t u;
            mag_e8m23_t f;
        } reinterpret;
        mag_e8m23_t base = fabs(x)*0x1.0p+112f*0x1.0p-110f;
        reinterpret.f = x;
        uint32_t shl1_w = reinterpret.u+reinterpret.u;
        uint32_t sign = reinterpret.u & 0x80000000u;
        reinterpret.u = 0x07800000u+(mag_xmax(0x71000000u, shl1_w&0xff000000u)>>1);
        reinterpret.f = base + reinterpret.f;
        uint32_t exp_bits = (reinterpret.u>>13) & 0x00007c00u;
        uint32_t mant_bits = reinterpret.u & 0x00000fffu;
        uint32_t nonsign = exp_bits + mant_bits;
        r = (sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign);
    #endif
    return (mag_e5m10_t){.bits=r};
}

static MAG_AINLINE mag_e8m23_t mag_e5m10_cvt_e8m23(mag_e5m10_t x) {
    #ifdef __F16C__
        #ifdef _MSC_VER
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x.bits)));
        #else
            return _cvtsh_ss(x.bits);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        return *(__fp16*)&x.bits;
    #else
        union {
            uint32_t u;
            mag_e8m23_t f;
        } reinterpret;
        uint32_t w = (uint32_t)x.bits<<16;
        uint32_t sign = w & 0x80000000u;
        uint32_t two_w = w+w;
        uint32_t offs = 0xe0u<<23;
        uint32_t t1 = (two_w>>4) + offs;
        uint32_t t2 = (two_w>>17) | (126u<<23);
        reinterpret.u = t1;
        mag_e8m23_t norm_x = reinterpret.f*0x1.0p-112f;
        reinterpret.u = t2;
        mag_e8m23_t denorm_x = reinterpret.f-0.5f;
        uint32_t denorm_cutoff = 1u<<27;
        uint32_t r = sign | (two_w < denorm_cutoff
            ? (reinterpret.f = denorm_x, reinterpret.u)
            : (reinterpret.f = norm_x, reinterpret.u));
        reinterpret.u = r;
        return reinterpret.f;
    #endif
}

#define mag_e8m23p(t) ((const mag_e8m23_t*)(t)->storage.base)
#define mag_e8m23p_mut(t) ((mag_e8m23_t*)(t)->storage.base)
#define mag_e5m10p(t) ((const mag_e5m10_t*)(t)->storage.base)
#define mag_e5m10p_mut(t) ((mag_e5m10_t*)(t)->storage.base)

/* Generate N uniform canonical floats in [0, 1) using active algorithm and rescale to [min, max]. */
static void MAG_AINLINE mag_prng_gen_uniform_vec_e8m23(mag_prng_state_t* prng, mag_e8m23_t* o, int64_t n, mag_e8m23_t min, mag_e8m23_t max) {
    mag_e8m23_t rescale_uniform = max - min;
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t ii=0; ii < n; ++ii) {
                if (--*rem <= 0) {
                    *rem = 624;
                    *next = 0;
                    uint32_t y, i;
                    for (i = 0; i < 624-397; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    for (; i < 624-1; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    y = (state[624-1] & 0x80000000u) | (*state & 0x7fffffffu);
                    state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                uint32_t y = state[(*next)++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680;
                y ^= (y << 15) & 0xefc60000;
                y ^= y >> 18;
                o[ii] = min + rescale_uniform * (1.f/(mag_e8m23_t)(1<<23)*((mag_e8m23_t)(y>>9) + 0.5f)); /* Generate canonical and rescale. */
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t* inc = &prng->pcg.inc;
            for (int64_t ii=0; ii < n; ++ii) {
                uint64_t prev = *state;
                *state = prev*6364136223846793005ull + *inc;
                uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
                uint32_t rot = prev >> 59u;
                uint32_t y = (mixed>>rot) | (mixed << ((-rot)&31));
                o[ii] = min + rescale_uniform * (1.f/(mag_e8m23_t)(1<<23)*((mag_e8m23_t)(y>>9) + 0.5f)); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N uniform canonical floats in [0, 1) using active algorithm and rescale to [min, max]. */
static void MAG_AINLINE mag_prng_gen_uniform_vec_e5m10(mag_prng_state_t* prng, mag_e5m10_t* o, int64_t n, mag_e8m23_t min, mag_e8m23_t max) {
    mag_e8m23_t rescale_uniform = max - min;
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t ii=0; ii < n; ++ii) {
                if (--*rem <= 0) {
                    *rem = 624;
                    *next = 0;
                    uint32_t y, i;
                    for (i = 0; i < 624-397; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    for (; i < 624-1; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    y = (state[624-1] & 0x80000000u) | (*state & 0x7fffffffu);
                    state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                uint32_t y = state[(*next)++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680;
                y ^= (y << 15) & 0xefc60000;
                y ^= y >> 18;
                o[ii] = mag_e8m23_cvt_e5m10(min + rescale_uniform*(1.f/(mag_e8m23_t)(1<<23)*((mag_e8m23_t)(y>>9) + 0.5f))); /* Generate canonical and rescale. */
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t* inc = &prng->pcg.inc;
            for (int64_t ii=0; ii < n; ++ii) {
                uint64_t prev = *state;
                *state = prev*6364136223846793005ull + *inc;
                uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
                uint32_t rot = prev >> 59u;
                uint32_t y = (mixed>>rot) | (mixed << ((-rot)&31));
                o[ii] = mag_e8m23_cvt_e5m10(min + rescale_uniform*(1.f/(mag_e8m23_t)(1<<23)*((mag_e8m23_t)(y>>9) + 0.5f))); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N normal (Gauss) distributed floats. */
static void MAG_HOTPROC mag_prng_gen_normal_vec_e8m23(mag_prng_state_t* prng, mag_e8m23_t* o, int64_t n, mag_e8m23_t mean, mag_e8m23_t std) {
    mag_assert2((n & 1) == 0);
    mag_prng_gen_uniform_vec_e8m23(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n; i += 2) { /* Map uniform to normal dist with Box-Muller transform. */
        mag_e8m23_t* u1 = o+i;
        mag_e8m23_t* u2 = o+i+1;
        mag_e8m23_t mag = std*sqrtf(-2.0f*logf(*u1));
        mag_e8m23_t y0 = mag*cosf(MAG_TAU**u2) + mean;
        mag_e8m23_t y1 = mag*sinf(MAG_TAU**u2) + mean;
        *u1 = y0;
        *u2 = y1;
    }
}

/* Generate N normal (Gauss) distributed floats. */
static void MAG_HOTPROC mag_prng_gen_normal_vec_e5m10(mag_prng_state_t* prng, mag_e5m10_t* o, int64_t n, mag_e8m23_t mean, mag_e8m23_t std) {
    mag_assert2((n & 1) == 0);
    mag_prng_gen_uniform_vec_e5m10(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n; i += 2) { /* Map uniform to normal dist with Box-Muller transform. */
        mag_e8m23_t u1 = mag_e5m10_cvt_e8m23(o[i]);
        mag_e8m23_t u2 = mag_e5m10_cvt_e8m23(o[i+1]);
        mag_e8m23_t mag = std*sqrtf(-2.0f*logf(u1));
        mag_e8m23_t y0 = mag*cosf(MAG_TAU*u2) + mean;
        mag_e8m23_t y1 = mag*sinf(MAG_TAU*u2) + mean;
        o[i] = mag_e8m23_cvt_e5m10(y0);
        o[i+1] = mag_e8m23_cvt_e5m10(y1);
    }
}

#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

static float32x4_t mag_simd_expf(float32x4_t x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0  */
    float32x4_t r = vdupq_n_f32(0x1.8p23f);
    float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    float32x4_t n = vsubq_f32(z, r);
    float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    float32x4_t u = vmulq_f32(b, b);
    float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
        vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
    uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
           vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static float32x4_t mag_simd_tanh(float32x4_t x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t neg_one = vdupq_n_f32(-1.0f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    float32x4_t a = vmulq_f32(neg_two, x);
    float32x4_t b = mag_simd_expf(a);
    float32x4_t c = vaddq_f32(one, b);
    float32x4_t inv = vrecpeq_f32(c);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
    return vaddq_f32(neg_one, vmulq_f32(two, inv));
}

static void mag_simd_sincos(float32x4_t x, float32x4_t *osin, float32x4_t *ocos) {
    uint32x4_t sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);
    float32x4_t y = vmulq_f32(x, vdupq_n_f32(1.27323954473516f));
    uint32x4_t emm2 = vcvtq_u32_f32(y);
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);
    uint32x4_t poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));
    x = vmlaq_f32(x, y, vdupq_n_f32(-0.78515625f));
    x = vmlaq_f32(x, y, vdupq_n_f32(-2.4187564849853515625e-4f));
    x = vmlaq_f32(x, y, vdupq_n_f32(-3.77489497744594108e-8f));
    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    uint32x4_t sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y1, y2;
    y1 = vmlaq_f32(vdupq_n_f32(-1.388731625493765e-003f), z, vdupq_n_f32(2.443315711809948e-005f));
    y2 = vmlaq_f32(vdupq_n_f32(8.3321608736e-3f), z, vdupq_n_f32(-1.9515295891e-4f));
    y1 = vmlaq_f32(vdupq_n_f32(4.166664568298827e-002f), y1, z);
    y2 = vmlaq_f32(vdupq_n_f32(-1.6666654611e-1f), y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y1 = vmlsq_f32(y1, z, vdupq_n_f32(0.5f));
    y2 = vmlaq_f32(x, y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));
    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *osin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ocos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)

static __m512 mag_simd_expf(const __m512 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
    __m512 r = _mm512_set1_ps(0x1.8p23f);
    __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    __m512 n = _mm512_sub_ps(z, r);
    __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    __m512 u = _mm512_mul_ps(b, b);
    __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))), u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F))
    );
    __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) return res;
    __m512 zero = _mm512_setzero_ps();
    __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

static __m512 mag_simd_tanh(__m512 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_one = _mm512_set1_ps(-1.0f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    __m512 a = _mm512_mul_ps(neg_two, x);
    __m512 b = mag_simd_expf(a);
    __m512 c = _mm512_add_ps(one, b);
    __m512 inv = _mm512_rcp14_ps(c);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm512_fmadd_ps(two, inv, neg_one);
}

#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)

static __m256 mag_simd_expf(const __m256 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
    __m256 r = _mm256_set1_ps(0x1.8p23f);
    __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    __m256 n = _mm256_sub_ps(z, r);
    __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),_mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    __m256 u = _mm256_mul_ps(b, b);
    __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,_mm256_set1_ps(0x1.573e2ep-5f)), u,_mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,_mm256_set1_ps(0x1.fffdb6p-2f))),u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
    __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),_mm256_set1_epi32(0x82000000u));
    __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
        _mm256_castsi256_ps(d),
        _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(c),
        _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k))))
    );
}

static __m256 mag_simd_tanh(__m256 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    __m256 a = _mm256_mul_ps(neg_two, x);
    __m256 b = mag_simd_expf(a);
    __m256 c = _mm256_add_ps(one, b);
    __m256 inv = _mm256_rcp_ps(c);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm256_fmadd_ps(two, inv, neg_one);
}

#elif MAG_APPROXMATH && defined(__SSE2__)

static __m128 mag_simd_expf(const __m128 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
    __m128 r = _mm_set1_ps(0x1.8p23f);
    __m128 z = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0x1.715476p+0f)), r);
    __m128 n = _mm_sub_ps(z, r);
    __m128 b = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(0x1.62e4p-1f))), _mm_mul_ps(n, _mm_set1_ps(0x1.7f7d1cp-20f)));
    __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    __m128i c = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    __m128 u = _mm_mul_ps(b, b);
    __m128 j = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.0e4020p-7f), b), _mm_set1_ps(0x1.573e2ep-5f)),u),
    _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.555e66p-3f), b), _mm_set1_ps(0x1.fffdb6p-2f))), u),
    _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c)) return _mm_add_ps(_mm_mul_ps(j, k), k);
    __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),_mm_set1_epi32(0x82000000u));
    __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    __m128i d = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
        _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s2, j), s2), s1)),
        _mm_andnot_ps(_mm_castsi128_ps(c), _mm_add_ps(_mm_mul_ps(k, j), k))))
    );
}

static __m128 mag_simd_tanh(__m128 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m128 one = _mm_set1_ps(1.0f);
    __m128 neg_one = _mm_set1_ps(-1.0f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 neg_two = _mm_set1_ps(-2.0f);
    __m128 a = _mm_mul_ps(neg_two, x);
    __m128 b = mag_simd_expf(a);
    __m128 c = _mm_add_ps(one, b);
    __m128 inv = _mm_rcp_ps(c);
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm_add_ps(neg_one, _mm_mul_ps(two, inv));
}

static void mag_simd_sincos(__m128 x, __m128 *osin, __m128 *ocos) {
    __m128 sign_mask_sin_ps = _mm_cmplt_ps(x, _mm_set1_ps(0.0f));
    __m128i sign_mask_sin = _mm_castps_si128(sign_mask_sin_ps);
    x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
    __m128 y = _mm_mul_ps(x, _mm_set1_ps(1.27323954473516f));
    __m128i emm2 = _mm_cvtps_epi32(y);
    emm2 = _mm_add_epi32(emm2, _mm_set1_epi32(1));
    emm2 = _mm_and_si128(emm2, _mm_set1_epi32(~1));
    y = _mm_cvtepi32_ps(emm2);
    __m128i poly_mask = _mm_cmpeq_epi32(emm2, _mm_set1_epi32(2));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-0.78515625f)));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-2.4187564849853515625e-4f)));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-3.77489497744594108e-8f)));
    __m128i tmp = _mm_cmpeq_epi32(emm2, _mm_set1_epi32(4));
    sign_mask_sin = _mm_xor_si128(sign_mask_sin, tmp);
    __m128i sign_mask_cos = _mm_cmpeq_epi32(_mm_sub_epi32(emm2, _mm_set1_epi32(2)), _mm_set1_epi32(4));
    __m128 z = _mm_mul_ps(x, x);
    __m128 y1 = _mm_add_ps(_mm_set1_ps(-1.388731625493765e-003f), _mm_mul_ps(z, _mm_set1_ps(2.443315711809948e-005f)));
    __m128 y2 = _mm_add_ps(_mm_set1_ps(8.3321608736e-3f), _mm_mul_ps(z, _mm_set1_ps(-1.9515295891e-4f)));
    y1 = _mm_add_ps(_mm_set1_ps(4.166664568298827e-002f), _mm_mul_ps(y1, z));
    y2 = _mm_add_ps(_mm_set1_ps(-1.6666654611e-1f), _mm_mul_ps(y2, z));
    y1 = _mm_mul_ps(y1, z);
    y2 = _mm_mul_ps(y2, z);
    y1 = _mm_mul_ps(y1, z);
    y1 = _mm_sub_ps(y1, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
    y2 = _mm_add_ps(x, _mm_mul_ps(y2, x));
    y1 = _mm_add_ps(y1, _mm_set1_ps(1.0f));
    __m128 poly_mask_ps = _mm_castsi128_ps(poly_mask);
    __m128 ys = _mm_or_ps(_mm_and_ps(poly_mask_ps, y1), _mm_andnot_ps(poly_mask_ps, y2));
    __m128 yc = _mm_or_ps(_mm_and_ps(poly_mask_ps, y2), _mm_andnot_ps(poly_mask_ps, y1));
    __m128 sign_mask_sin_ps2 = _mm_castsi128_ps(sign_mask_sin);
    __m128 neg_ys = _mm_sub_ps(_mm_setzero_ps(), ys);
    __m128 osin_ps = _mm_or_ps(_mm_and_ps(sign_mask_sin_ps2, neg_ys), _mm_andnot_ps(sign_mask_sin_ps2, ys));
    __m128 sign_mask_cos_ps = _mm_castsi128_ps(sign_mask_cos);
    __m128 neg_yc = _mm_sub_ps(_mm_setzero_ps(), yc);
    __m128 ocos_ps = _mm_or_ps(_mm_and_ps(sign_mask_cos_ps, yc), _mm_andnot_ps(sign_mask_cos_ps, neg_yc));
    *osin = osin_ps;
    *ocos = ocos_ps;
}

#endif

static void MAG_HOTPROC mag_vadd_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t* y
) {
#ifdef MAG_ACCELERATE
    vDSP_vadd(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] + y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vadd_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    const mag_e5m10_t* y
) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vaddq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vadd_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vaddq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_add_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_add_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_add_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vsub_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t* y
) {
#ifdef MAG_ACCELERATE
    vDSP_vsub(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] - y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vsub_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    const mag_e5m10_t* y
) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vsubq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vsub_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vsubq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_sub_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_sub_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_sub_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vmul_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t* y
) {
#ifdef MAG_ACCELERATE
    vDSP_vmul(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] * y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vmul_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    const mag_e5m10_t* y
) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vmulq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vmul_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vmulq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_mul_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_mul_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_mul_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) * mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vdiv_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t* y
) {
#ifdef MAG_ACCELERATE
    vDSP_vdiv(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] / y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vdiv_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    const mag_e5m10_t* y
) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vdivq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vdiv_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vdivq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_div_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_div_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_div_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vadds_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t y
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] + y;
    }
}

static void MAG_HOTPROC mag_vadds_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    mag_e5m10_t y
) {
    mag_e8m23_t ys =  mag_e5m10_cvt_e8m23(y);
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + ys);
    }
}

static void MAG_HOTPROC mag_vsubs_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t y
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] - y;
    }
}

static void MAG_HOTPROC mag_vsubs_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    mag_e5m10_t y
) {
    mag_e8m23_t ys =  mag_e5m10_cvt_e8m23(y);
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - ys);
    }
}

static void MAG_HOTPROC mag_vmuls_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t y
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] * y;
    }
}

static void MAG_HOTPROC mag_vmuls_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    mag_e5m10_t y
) {
    mag_e8m23_t ys =  mag_e5m10_cvt_e8m23(y);
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) * ys);
    }
}

static void MAG_HOTPROC mag_vdivs_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t y
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] / y;
    }
}

static void MAG_HOTPROC mag_vdivs_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    mag_e5m10_t y
) {
    mag_e8m23_t ys =  mag_e5m10_cvt_e8m23(y);
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / ys);
    }
}

static void MAG_HOTPROC mag_vpows_e8m23(
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x,
    const mag_e8m23_t y
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = powf(x[i], y);
    }
}

static void MAG_HOTPROC mag_vpows_e5m10(
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x,
    mag_e5m10_t y
) {
    mag_e8m23_t ys =  mag_e5m10_cvt_e8m23(y);
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(powf(mag_e5m10_cvt_e8m23(x[i]), ys));
    }
}

static mag_e8m23_t MAG_UNUSED MAG_HOTPROC mag_vdot_e8m23(
    int64_t numel,
    const mag_e8m23_t* x,
    const mag_e8m23_t* y
) {
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    int64_t k = numel & -16;
    float32x4_t acc[4] = {vdupq_n_f32(0)};
    float32x4_t vx[4];
    float32x4_t vy[4];
    for (int64_t i=0; i < k; i += 16) { /* Process STEP elements at a time */
        vx[0] = vld1q_f32(x+i+(0<<2));
        vy[0] = vld1q_f32(y+i+(0<<2));
        acc[0] = vfmaq_f32(acc[0], vx[0], vy[0]);
        vx[1] = vld1q_f32(x+i+(1<<2));
        vy[1] = vld1q_f32(y+i+(1<<2));
        acc[1] = vfmaq_f32(acc[1], vx[1], vy[1]);
        vx[2] = vld1q_f32(x+i+(2<<2));
        vy[2] = vld1q_f32(y+i+(2<<2));
        acc[2] = vfmaq_f32(acc[2], vx[2], vy[2]);
        vx[3] = vld1q_f32(x+i+(3<<2));
        vy[3] = vld1q_f32(y+i+(3<<2));
        acc[3] = vfmaq_f32(acc[3], vx[3], vy[3]);
    }
    acc[1] = vaddq_f32(acc[1], acc[3]); /* Fold acc[1] += acc[3] */
    *acc = vaddq_f32(*acc, acc[2]);     /* Fold acc[0] += acc[2] */
    *acc = vaddq_f32(*acc, acc[1]);     /* Fold acc[0] += acc[1] */
    mag_e8m23_t sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#elif defined(__AVX512F__) && defined(__FMA__)
    int64_t k = numel & -64;
    __m512 acc[4] = {_mm512_setzero_ps()};
    __m512 vx[4];
    __m512 vy[4];
    for (int64_t i=0; i < k; i += 64) {
        vx[0] = _mm512_loadu_ps(x+i+(0<<4));
        vy[0] = _mm512_loadu_ps(y+i+(0<<4));
        acc[0] = _mm512_fmadd_ps(vx[0], vy[0], acc[0]);
        vx[1] = _mm512_loadu_ps(x+i+(1<<4));
        vy[1] = _mm512_loadu_ps(y+i+(1<<4));
        acc[1] = _mm512_fmadd_ps(vx[1], vy[1], acc[1]);
        vx[2] = _mm512_loadu_ps(x+i+(2<<4));
        vy[2] = _mm512_loadu_ps(y+i+(2<<4));
        acc[2] = _mm512_fmadd_ps(vx[2], vy[2], acc[2]);
        vx[3] = _mm512_loadu_ps(x+i+(3<<4));
        vy[3] = _mm512_loadu_ps(y+i+(3<<4));
        acc[3] = _mm512_fmadd_ps(vx[3], vy[3], acc[3]);
    }
    acc[1] = _mm512_add_ps(acc[1], acc[3]);
    *acc = _mm512_add_ps(*acc, acc[2]);
    *acc = _mm512_add_ps(*acc, acc[1]);
    mag_e8m23_t sum = _mm512_reduce_add_ps(*acc);
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#elif defined(__AVX__) && defined(__FMA__)
    int64_t k = numel & -32;
    __m256 acc[4] = {_mm256_setzero_ps()};
    __m256 vx[4];
    __m256 vy[4];
    for (int64_t i=0; i < k; i += 32) {
        vx[0] = _mm256_loadu_ps(x+i+(0<<3));
        vy[0] = _mm256_loadu_ps(y+i+(0<<3));
        acc[0] = _mm256_fmadd_ps(vx[0], vy[0], acc[0]);
        vx[1] = _mm256_loadu_ps(x+i+(1<<3));
        vy[1] = _mm256_loadu_ps(y+i+(1<<3));
        acc[1] = _mm256_fmadd_ps(vx[1], vy[1], acc[1]);
        vx[2] = _mm256_loadu_ps(x+i+(2<<3));
        vy[2] = _mm256_loadu_ps(y+i+(2<<3));
        acc[2] = _mm256_fmadd_ps(vx[2], vy[2], acc[2]);
        vx[3] = _mm256_loadu_ps(x+i+(3<<3));
        vy[3] = _mm256_loadu_ps(y+i+(3<<3));
        acc[3] = _mm256_fmadd_ps(vx[3], vy[3], acc[3]);
    }
    acc[1] = _mm256_add_ps(acc[1], acc[3]);
    *acc = _mm256_add_ps(*acc, acc[2]);
    *acc = _mm256_add_ps(*acc, acc[1]);
    __m128 v0 = _mm_add_ps(_mm256_castps256_ps128(*acc), _mm256_extractf128_ps(*acc, 1));
    v0 = _mm_hadd_ps(v0, v0);
    v0 = _mm_hadd_ps(v0, v0);
    mag_e8m23_t sum = _mm_cvtss_f32(v0);
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#elif defined(__SSE2__)
    int64_t k = numel & -16;
    __m128 acc[4] = {_mm_setzero_ps()};
    __m128 vx[4];
    __m128 vy[4];
    for (int64_t i=0; i < k; i += 16) {
        vx[0] = _mm_loadu_ps(x+i+(0<<2));
        vy[0] = _mm_loadu_ps(y+i+(0<<2));
        acc[0] = _mm_add_ps(acc[0], _mm_mul_ps(vx[0], vy[0]));
        vx[1] = _mm_loadu_ps(x+i+(1<<2));
        vy[1] = _mm_loadu_ps(y+i+(1<<2));
        acc[1] = _mm_add_ps(acc[1], _mm_mul_ps(vx[1], vy[1]));
        vx[2] = _mm_loadu_ps(x+i+(2<<2));
        vy[2] = _mm_loadu_ps(y+i+(2<<2));
        acc[2] = _mm_add_ps(acc[2], _mm_mul_ps(vx[2], vy[2]));
        vx[3] = _mm_loadu_ps(x+i+(3<<2));
        vy[3] = _mm_loadu_ps(y+i+(3<<2));
        acc[3] = _mm_add_ps(acc[3], _mm_mul_ps(vx[3], vy[3]));
    }
    #ifdef __SSE3__
        acc[1] = _mm_add_ps(acc[1], acc[3]);
        *acc = _mm_add_ps(*acc, acc[2]);
        *acc = _mm_add_ps(*acc, acc[1]);
        *acc = _mm_hadd_ps(*acc, *acc);
        *acc = _mm_hadd_ps(*acc, *acc);
        mag_e8m23_t sum = _mm_cvtss_f32(*acc);
    #else
        __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(*acc, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        mag_e8m23_t sum = _mm_cvtss_f32(sums);
    #endif
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#else
    mag_e11m52_t r = 0.0;
    for (int64_t i=0; i < numel; ++i) r += (mag_e11m52_t)x[i] * (mag_e11m52_t)y[i];
    return (mag_e8m23_t)r;
#endif
}

static mag_e11m52_t MAG_HOTPROC mag_vsum_f64_e8m23( /* Σx. */
    int64_t numel,
    const mag_e8m23_t* x
) {
#ifdef MAG_ACCELERATE
    mag_e8m23_t sum;
    vDSP_sve(x, 1, &sum, numel);
    return (mag_e11m52_t)sum;
#else
    mag_e11m52_t sum = 0.0;
    for (int64_t i=0; i < numel; ++i) {
        sum += (mag_e11m52_t)x[i];
    }
    return sum;
#endif
}

static mag_e11m52_t MAG_HOTPROC mag_vsum_f64_e5m10( /* Σx. */
    int64_t numel,
    const mag_e5m10_t* x
) {
    mag_e11m52_t sum = 0.0;
    for (int64_t i=0; i < numel; ++i) {
        sum += mag_e5m10_cvt_e8m23(x[i]);
    }
    return sum;
}

static mag_e8m23_t MAG_HOTPROC mag_vmin_e8m23( /* min x */
    int64_t numel,
    const mag_e8m23_t* x
) {
    mag_e8m23_t min = INFINITY;
    for (int64_t i=0; i < numel; ++i) {
        min = fminf(min, x[i]);
    }
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmin_e5m10( /* min x */
    int64_t numel,
    const mag_e5m10_t* x
) {
    mag_e8m23_t min = INFINITY;
    for (int64_t i=0; i < numel; ++i) {
        min = fminf(min, mag_e5m10_cvt_e8m23(x[i]));
    }
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmax_e8m23( /* max x */
    int64_t numel,
    const mag_e8m23_t* x
) {
    mag_e8m23_t min = -INFINITY;
    for (int64_t i=0; i < numel; ++i) {
        min = fmaxf(min, x[i]);
    }
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmax_e5m10( /* min x */
    int64_t numel,
    const mag_e5m10_t* x
) {
    mag_e8m23_t min = -INFINITY;
    for (int64_t i=0; i < numel; ++i) {
        min = fmaxf(min, mag_e5m10_cvt_e8m23(x[i]));
    }
    return min;
}

static void MAG_HOTPROC mag_vabs_e8m23( /* o = |x| */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = fabsf(x[i]);
    }
}

static void MAG_HOTPROC mag_vabs_e5m10( /* o = |x| */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(fabsf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vneg_e8m23( /* o = -x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = -x[i];
    }
}

static void MAG_HOTPROC mag_vneg_e5m10( /* o = -x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(-mag_e5m10_cvt_e8m23(x[i]));
    }
}

static void MAG_HOTPROC mag_vlog_e8m23( /* o = log x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1);
    for (; i+3 < numel; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        xi = vmaxq_f32(xi, vdupq_n_f32(0));
        uint32x4_t invalid_mask = vcleq_f32(xi, vdupq_n_f32(0));
        int32x4_t ux = vreinterpretq_s32_f32(xi);
        int32x4_t emm0 = vshrq_n_s32(ux, 23);
        ux = vandq_s32(ux, vdupq_n_s32(~0x7f800000u));
        ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
        xi = vreinterpretq_f32_s32(ux);
        emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
        float32x4_t e = vcvtq_f32_s32(emm0);
        e = vaddq_f32(e, one);
        uint32x4_t mask = vcltq_f32(xi, vdupq_n_f32(0.707106781186547524f));
        float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xi), mask));
        xi = vsubq_f32(xi, one);
        e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
        xi = vaddq_f32(xi, tmp);
        float32x4_t z = vmulq_f32(xi, xi);
        float32x4_t y = vdupq_n_f32(7.0376836292e-2f);
        y = vmlaq_f32(vdupq_n_f32(-1.1514610310e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(1.1676998740e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-1.2420140846e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(1.4249322787e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-1.6668057665e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(2.0000714765e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-2.4999993993e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(3.3333331174e-1f), y, xi);
        y = vmulq_f32(y, xi);
        y = vmulq_f32(y, z);
        y = vmlaq_f32(y, e, vdupq_n_f32(-2.12194440e-4f));
        y = vmlsq_f32(y, z, vdupq_n_f32(0.5f));
        xi = vaddq_f32(xi, y);
        xi = vmlaq_f32(xi, e, vdupq_n_f32(0.693359375f));
        xi = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(xi), invalid_mask));
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
/* TODO */
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
/* TODO */
#elif MAG_APPROXMATH && defined(__SSE2__)
    const __m128 one = _mm_set1_ps(1.0f);
    for (; i+3 < numel; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        xi = _mm_max_ps(xi, _mm_set1_ps(0.0f));
        __m128 invalid_mask = _mm_cmple_ps(xi, _mm_set1_ps(0.0f));
        __m128i ux = _mm_castps_si128(xi);
        __m128i emm0 = _mm_srli_epi32(ux, 23);
        ux = _mm_and_si128(ux, _mm_set1_epi32(~0x7f800000u));
        ux = _mm_or_si128(ux, _mm_castps_si128(_mm_set1_ps(0.5f)));
        xi = _mm_castsi128_ps(ux);
        emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
        __m128 e = _mm_cvtepi32_ps(emm0);
        e = _mm_add_ps(e, one);
        __m128 mask = _mm_cmplt_ps(xi, _mm_set1_ps(0.707106781186547524f));
        __m128 tmp = _mm_and_ps(xi, mask);
        xi = _mm_sub_ps(xi, one);
        e = _mm_sub_ps(e, _mm_and_ps(one, mask));
        xi = _mm_add_ps(xi, tmp);
        __m128 z = _mm_mul_ps(xi, xi);
        __m128 y = _mm_set1_ps(7.0376836292e-2f);
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.1514610310e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(1.1676998740e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.2420140846e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(1.4249322787e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.6668057665e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(2.0000714765e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-2.4999993993e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(3.3333331174e-1f));
        y = _mm_mul_ps(y, xi);
        y = _mm_mul_ps(y, z);
        y = _mm_add_ps(_mm_mul_ps(e, _mm_set1_ps(-2.12194440e-4f)), y);
        y = _mm_sub_ps(y, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
        xi = _mm_add_ps(xi, y);
        xi = _mm_add_ps(_mm_mul_ps(e, _mm_set1_ps(0.693359375f)), xi);
        xi = _mm_or_ps(xi, invalid_mask);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < numel; ++i) { /* Process leftovers scalar-wise */
        o[i] = logf(x[i]);
    }
}

static void MAG_HOTPROC mag_vlog_e5m10( /* o = log x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(logf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vsqr_e8m23( /* o = x² */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*x[i];
}

static void MAG_HOTPROC mag_vsqr_e5m10( /* o = x² */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t e8m23 = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(e8m23*e8m23);
    }
}

static void MAG_HOTPROC mag_vsqrt_e8m23( /* o = √x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = sqrtf(x[i]);
    }
}

static void MAG_HOTPROC mag_vsqrt_e5m10( /* o = √x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(sqrtf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vsin_e8m23( /* o = sin x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
       int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        float32x4_t ocos;
        mag_simd_sincos(xi, &xi, &ocos);
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    /* TODO */
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    /* TODO */
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < numel; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        __m128 ocos;
        mag_simd_sincos(xi, &xi, &ocos);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = sinf(x[i]);
    }
}

static void MAG_HOTPROC mag_vsin_e5m10( /* o = sin x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(sinf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vcos_e8m23( /* o = cos x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        float32x4_t osin;
        mag_simd_sincos(xi, &osin, &xi);
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    /* TODO */
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    /* TODO */
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < numel; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        __m128 osin;
        mag_simd_sincos(xi, &osin, &xi);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = cosf(x[i]);
    }
}

static void MAG_HOTPROC mag_vcos_e5m10( /* o = cos x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(cosf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vstep_e8m23( /* Heaviside step function. */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] >= 0.0f ? 1.0f : 0.0f;
    }
}

static void MAG_HOTPROC mag_vstep_e5m10( /* Heaviside step function. */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) >= 0.0f ? 1.0f : 0.0f);
    }
}

static void MAG_HOTPROC mag_vexp_e8m23( /* o = eˣ */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        vst1q_f32(o+i, mag_simd_expf(vld1q_f32(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i+15 < numel; i += 16) {
        _mm512_storeu_ps(o+i, mag_simd_expf(_mm512_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    for (; i+7 < numel; i += 8) {
        _mm256_storeu_ps(o+i, mag_simd_expf(_mm256_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < numel; i += 4) {
        _mm_storeu_ps(o+i, mag_simd_expf(_mm_loadu_ps(x+i)));
    }
#endif
    for (; i < numel; ++i) o[i] = expf(x[i]); /* Scalar drain loop */
}

static void MAG_HOTPROC mag_vexp_e5m10( /* o = eˣ */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(expf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vsoftmax_e8m23( /* softmax : ℝ -> (0, ∞), x |-> eˣ */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        vst1q_f32(o+i, mag_simd_expf(vld1q_f32(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i+15 < numel; i += 16) {
        _mm512_storeu_ps(o+i, mag_simd_expf(_mm512_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    for (; i+7 < numel; i += 8) {
        _mm256_storeu_ps(o+i, mag_simd_expf(_mm256_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < numel; i += 4) {
        _mm_storeu_ps(o+i, mag_simd_expf(_mm_loadu_ps(x+i)));
    }
#endif
    for (; i < numel; ++i) o[i] = expf(x[i]); /* Scalar drain loop */
}

static void MAG_HOTPROC mag_vsoftmax_e5m10( /* softmax : ℝ -> (0, ∞), x |-> eˣ */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(expf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vsoftmax_dv_e8m23( /* softmax' = softmax : ℝ -> (0, ∞), x |-> eˣ */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    mag_vsoftmax_e8m23(numel, o, x);
}

static void MAG_HOTPROC mag_vsoftmax_dv_e5m10( /* softmax : ℝ -> (0, ∞), x |-> eˣ */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    mag_vsoftmax_e5m10(numel, o, x);
}

static void MAG_HOTPROC mag_vsigmoid_e8m23( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i+3 < numel; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t neg_x = vsubq_f32(zero, xx);
        float32x4_t exp_neg_x = mag_simd_expf(neg_x);
        float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
        vst1q_f32(o+i, vdivq_f32(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    for (; i+15 < numel; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 neg_x = _mm512_sub_ps(zero, xx);
        __m512 exp_neg_x = mag_simd_expf(neg_x);
        __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
        _mm512_storeu_ps(o+i, _mm512_div_ps(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_setzero_ps();
    for (; i+7 < numel; i += 8) {
        __m256 xx = _mm256_loadu_ps(x+i);
        __m256 neg_x = _mm256_sub_ps(zero, xx);
        __m256 exp_neg_x = mag_simd_expf(neg_x);
        __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
        _mm256_storeu_ps(o+i, _mm256_div_ps(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_setzero_ps();
    for (; i+3 < numel; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 neg_x = _mm_sub_ps(zero, xx);
        __m128 exp_neg_x = mag_simd_expf(neg_x);
        __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
        _mm_storeu_ps(o+i, _mm_div_ps(one, one_plus_exp_neg_x));
    }
#endif
    for (; i < numel; ++i) o[i] = 1.0f / (1.0f + expf(-x[i])); /* Scalar drain loop */
}

static void MAG_HOTPROC mag_vsigmoid_e5m10( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(1.0f / (1.0f + expf(-mag_e5m10_cvt_e8m23(x[i]))));
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_e8m23( /* σ' : ℝ -> (0, 1), x |-> x * (1-x) */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] * (1.0f - x[i]);
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_e5m10( /* σ' : ℝ -> (0, 1), x |-> x * (1-x) */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t e8m23 = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(e8m23 * (1.0f - e8m23));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_e8m23( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_e5m10( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(fminf(1.0f, fmaxf(0.0f, (mag_e5m10_cvt_e8m23(x[i]) + 3.0f) / 6.0f)));
    }
}

static void MAG_HOTPROC mag_vsilu_e8m23( /* silu : ℝ -> ℝ, x |-> x/(1 + e^(-x)) */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i+3 < numel; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t neg_x = vsubq_f32(zero, xx);
        float32x4_t exp_neg_x = mag_simd_expf(neg_x);
        float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
        vst1q_f32(o+i, vdivq_f32(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 one = _mm512_set1_ps(1);
    __m512 zero = _mm512_setzero_ps();
    for (; i+15 < numel; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 neg_x = _mm512_sub_ps(zero, xx);
        __m512 exp_neg_x = mag_simd_expf(neg_x);
        __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
        _mm512_storeu_ps(o+i, _mm512_div_ps(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 one = _mm256_set1_ps(1);
    __m256 zero = _mm256_setzero_ps();
    for (; i+7 < numel; i += 8) {
        const __m256 xx = _mm256_loadu_ps(x+i);
        __m256 neg_x = _mm256_sub_ps(zero, xx);
        __m256 exp_neg_x = mag_simd_expf(neg_x);
        __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
        _mm256_storeu_ps(o+i, _mm256_div_ps(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_setzero_ps();
    for (; i+3 < numel; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 neg_x = _mm_sub_ps(zero, xx);
        __m128 exp_neg_x = mag_simd_expf(neg_x);
        __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
        _mm_storeu_ps(o+i, _mm_div_ps(xx, one_plus_exp_neg_x));
    }
#endif
    for (; i < numel; ++i) {
        o[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void MAG_HOTPROC mag_vsilu_e5m10( /* silu : ℝ -> ℝ, x |-> x/(1 + e^(-x)) */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t e8m23 = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(e8m23/(1.0f + expf(-e8m23)));
    }
}

static void MAG_HOTPROC mag_vsilu_dv_e8m23( /* silu' : ℝ -> TODO */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    mag_panic("NYI!");
}

static void MAG_HOTPROC mag_vsilu_dv_e5m10( /* silu' : ℝ -> TODO */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    mag_panic("NYI!");
}

static void MAG_HOTPROC mag_vtanh_e8m23( /* tanh : ℝ -> (-1, 1), x |-> tanh x */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        vst1q_f32(o+i, mag_simd_tanh(vld1q_f32(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i+15 < numel; i += 16) {
        _mm512_storeu_ps(o+i, mag_simd_tanh(_mm512_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    for (; i+7 < numel; i += 8) {
        _mm256_storeu_ps(o+i, mag_simd_tanh(_mm256_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < numel; i += 4) {
        _mm_storeu_ps(o+i, mag_simd_tanh(_mm_loadu_ps(x+i)));
    }
#endif
    for (; i < numel; ++i) {
        o[i] = tanhf(x[i]);
    }
}

static void MAG_HOTPROC mag_vtanh_e5m10( /* o = tanh x */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(tanhf(mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vtanh_dv_e8m23( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t t = tanhf(x[i]);
        o[i] = 1.0f - t*t;
    }
}

static void MAG_HOTPROC mag_vtanh_dv_e5m10( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t e8m23 = tanhf(mag_e5m10_cvt_e8m23(x[i]));
        o[i] = mag_e8m23_cvt_e5m10(1.0f - e8m23*e8m23);
    }
}

static void MAG_HOTPROC mag_vrelu_e8m23( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_xmax(0.0f, x[i]);
    }
}

static void MAG_HOTPROC mag_vrelu_e5m10( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_xmax(0.0f, mag_e5m10_cvt_e8m23(x[i])));
    }
}

static void MAG_HOTPROC mag_vrelu_dv_e8m23( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] <= 0.0f ? 0.0f : 1.0f; /* relu' is mathematically undefined for x = 0, but we return 0 in this case. */
    }
}

static void MAG_HOTPROC mag_vrelu_dv_e5m10( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) <= 0.0f ? 0.0f : 1.0f);
    }
}

static void MAG_HOTPROC mag_vgelu_e8m23( /* gelu : ℝ -> ℝ, x |-> TODO */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t coeff1 = vdupq_n_f32(0.79788456080286535587989211986876f);
    float32x4_t coeff2 = vdupq_n_f32(MAG_GELU_COEFF);
    for (; i+3 < numel; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t a = vaddq_f32(one, vmulq_f32(coeff2, vmulq_f32(xx, xx)));
        float32x4_t b = vaddq_f32(one, mag_simd_tanh(vmulq_f32(coeff1, vmulq_f32(xx, a))));
        float32x4_t c = vmulq_f32(half, vmulq_f32(xx, b));
        vst1q_f32(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 coeff1 = _mm512_set1_ps(0.79788456080286535587989211986876f);
    __m512 coeff2 = _mm512_set1_ps(MAG_GELU_COEFF);
    for (; i+15 < numel; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 a = _mm512_fmadd_ps(coeff2, _mm512_mul_ps(xx, xx), one);
        __m512 b = _mm512_add_ps(one, mag_simd_tanh(_mm512_mul_ps(coeff1, _mm512_mul_ps(xx, a))));
        __m512 c = _mm512_mul_ps(half, _mm512_mul_ps(xx, b));
        _mm512_storeu_ps(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 coeff1 = _mm256_set1_ps(0.79788456080286535587989211986876f);
    __m256 coeff2 = _mm256_set1_ps(MAG_GELU_COEFF);
    for (; i+7 < numel; i += 8) {
        __m256 xx = _mm256_loadu_ps(x+i);
        __m256 a = _mm256_fmadd_ps(coeff2, _mm256_mul_ps(xx, xx), one);
        __m256 b = _mm256_add_ps(one, mag_simd_tanh(_mm256_mul_ps(coeff1, _mm256_mul_ps(xx, a))));
        __m256 c = _mm256_mul_ps(half, _mm256_mul_ps(xx, b));
        _mm256_storeu_ps(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 one = _mm_set1_ps(1.0f);
    __m128 coeff1 = _mm_set1_ps(0.79788456080286535587989211986876f);
    __m128 coeff2 = _mm_set1_ps(MAG_GELU_COEFF);
    for (; i+3 < numel; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 a = _mm_add_ps(one, _mm_mul_ps(coeff2, _mm_mul_ps(xx, xx)));
        __m128 b = _mm_add_ps(one, mag_simd_tanh(_mm_mul_ps(coeff1, _mm_mul_ps(xx, a))));
        __m128 c = _mm_mul_ps(half, _mm_mul_ps(xx, b));
        _mm_storeu_ps(o+i, c);
    }
#endif
    for (; i < numel; ++i) {
        o[i] = 0.5f*x[i]*(1.0f + tanhf(0.79788456080286535587989211986876f*x[i]*(1.0f + MAG_GELU_COEFF*x[i]*x[i])));
    }
}

static void MAG_HOTPROC mag_vgelu_e5m10( /* gelu : ℝ -> ℝ, x |-> TODO */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t e8m23 = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(0.5f*e8m23*(1.0f + tanhf(0.79788456080286535587989211986876f*e8m23*(1.0f + MAG_GELU_COEFF*e8m23*e8m23))));
    }
}

static void MAG_HOTPROC mag_vgelu_dv_e8m23( /* gelu' : ℝ -> ℝ, x |-> TODO */
    int64_t numel,
    mag_e8m23_t* o,
    const mag_e8m23_t* x
) {
    mag_panic("NYI");
}

static void MAG_HOTPROC mag_vgelu_dv_e5m10( /* gelu' : ℝ -> ℝ, x |-> TODO */
    int64_t numel,
    mag_e5m10_t* o,
    const mag_e5m10_t* x
) {
    mag_panic("NYI");
}

static void mag_blas_nop(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) { (void)payload; (void)ctx; }

static void mag_blas_clone(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    mag_assert2(mag_tensor_is_shape_eq(x, r));
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    const mag_e8m23_t* b_x = mag_e8m23p(x);
    memcpy(b_r, b_x, mag_tensor_data_size(r));
}

static void mag_blas_init_broadcast_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e8m23_t xi = r->init_op_params[0].x.e8m23;
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    if (xi == 0.0f) {
        memset(b_r, 0, mag_tensor_data_size(r));
        return;
    }
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_broadcast_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e5m10_t xi = mag_e8m23_cvt_e5m10(r->init_op_params[0].x.e8m23);
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_rand_uniform_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e8m23_t min = r->init_op_params[0].x.e8m23;
    mag_e8m23_t max = r->init_op_params[1].x.e8m23;
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e8m23(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_uniform_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e8m23_t min = r->init_op_params[0].x.e8m23;
    mag_e8m23_t max = r->init_op_params[1].x.e8m23;
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e5m10(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_normal_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e8m23_t mean = r->init_op_params[0].x.e8m23;
    mag_e8m23_t stddev = r->init_op_params[1].x.e8m23;
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e8m23(payload->local_prng, b_r, numel, mean, stddev);
}

static void mag_blas_init_rand_normal_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    mag_e8m23_t mean = r->init_op_params[0].x.e8m23;
    mag_e8m23_t stddev = r->init_op_params[1].x.e8m23;
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e5m10(payload->local_prng, b_r, numel, mean, stddev);
}

static void MAG_HOTPROC mag_blas_mean_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    const mag_e8m23_t* b_x = mag_e8m23p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e11m52_t sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e8m23_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_e8m23(
                            x_d0,
                            p_x
                        );
                    }
                }
            }
        }
    }
    sum /= (mag_e11m52_t)x->numel;
    *b_r = (mag_e8m23_t)sum;
}

static void MAG_HOTPROC mag_blas_mean_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    const mag_e5m10_t* b_x = mag_e5m10p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e11m52_t sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e5m10_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_e5m10(
                            x_d0,
                            p_x
                        );
                    }
                }
            }
        }
    }
    sum /= (mag_e11m52_t)x->numel;
    *b_r = mag_e8m23_cvt_e5m10((mag_e8m23_t)sum);
}

static void MAG_HOTPROC mag_blas_min_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    const mag_e8m23_t* b_x = mag_e8m23p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e8m23_t min = INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e8m23_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        min = fminf(mag_vmin_e8m23(x_d0, p_x), min);
                    }
                }
            }
        }
    }
    *b_r = min;
}

static void MAG_HOTPROC mag_blas_min_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    const mag_e5m10_t* b_x = mag_e5m10p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e8m23_t min = INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e5m10_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        min = fminf(mag_vmin_e5m10(x_d0, p_x), min);
                    }
                }
            }
        }
    }
    *b_r = mag_e8m23_cvt_e5m10(min);
}

static void MAG_HOTPROC mag_blas_max_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    const mag_e8m23_t* b_x = mag_e8m23p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e8m23_t max = -INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e8m23_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        max = fmaxf(mag_vmax_e8m23(x_d0, p_x), max);
                    }
                }
            }
        }
    }
    *b_r = max;
}

static void MAG_HOTPROC mag_blas_max_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    const mag_e5m10_t* b_x = mag_e5m10p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e8m23_t max = -INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e5m10_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        max = fmaxf(mag_vmax_e5m10(x_d0, p_x), max);
                    }
                }
            }
        }
    }
    *b_r = mag_e8m23_cvt_e5m10(max);
}

static void MAG_HOTPROC mag_blas_sum_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    const mag_e8m23_t* b_x = mag_e8m23p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e11m52_t sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e8m23_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_e8m23(x_d0, p_x);
                    }
                }
            }
        }
    }
    *b_r = (mag_e8m23_t)sum;
}

static void MAG_HOTPROC mag_blas_sum_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    (void)ctx;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* const x = r->op_inputs[0];
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    const mag_e5m10_t* b_x = mag_e5m10p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_e11m52_t sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const mag_e5m10_t* p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_e5m10(x_d0, p_x);
                    }
                }
            }
        }
    }
    *b_r = mag_e8m23_cvt_e5m10((mag_e8m23_t)sum);
}

#define mag_cpu_blas_impl_unary(T, name) \
    static void MAG_HOTPROC mag_blas_##name##_##T(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) { \
        (void)ctx; \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        mag_load_local_storage_group(r, r_s, strides); \
        mag_load_local_storage_group(x, x_s, strides); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t numel = r->numel; \
        int64_t chunk = (numel + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t vmel = mag_xmin(ra + chunk, numel) - ra; \
        if (mag_unlikely(vmel <= 0)) return; \
        mag_##T##_t* pr = br + ra; \
        const mag_##T##_t* px = bx + ra; \
        mag_bnd_chk(pr, br, mag_tensor_data_size(r)); \
        mag_bnd_chk(px, bx, mag_tensor_data_size(x)); \
        mag_v##name##_##T(vmel, pr, px); \
    }

#define mag_cpu_blas_impl_unary_scalar(T, name) \
    static void MAG_HOTPROC mag_blas_##name##s_##T(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) { \
        (void)ctx; \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        mag_##T##_t xi = r->op_params->x.T; \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        mag_load_local_storage_group(r, r_s, strides); \
        mag_load_local_storage_group(x, x_s, strides); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t numel = r->numel; \
        int64_t chunk = (numel + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t vmel = mag_xmin(ra + chunk, numel) - ra; \
        if (mag_unlikely(vmel <= 0)) return; \
        mag_##T##_t* pr = br + ra; \
        const mag_##T##_t* px = bx + ra; \
        mag_bnd_chk(pr, br, mag_tensor_data_size(r)); \
        mag_bnd_chk(px, bx, mag_tensor_data_size(x)); \
        mag_v##name##s_##T(vmel, pr, px, xi); \
    }

#define mag_cpu_blas_impl_binary(T, name, op) \
    static void MAG_HOTPROC mag_blas_##name##_##T(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) { \
        (void)ctx; \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        const mag_tensor_t* y = r->op_inputs[1]; \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        const mag_##T##_t* by = mag_##T##p(y); \
        mag_load_local_storage_group(r, rd, shape); \
        mag_load_local_storage_group(r, rs, strides); \
        mag_load_local_storage_group(x, xd, shape); \
        mag_load_local_storage_group(x, xs, strides); \
        mag_load_local_storage_group(y, yd, shape); \
        mag_load_local_storage_group(y, ys, strides); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        if (xd0==yd0 && xd1==yd1 && xd2==yd2 && xd3==yd3 && xd4==yd4 && xd5==yd5) { \
            int64_t numel = r->numel; \
            int64_t chunk = (numel + tc - 1)/tc; \
            int64_t ra = ti*chunk; \
            int64_t vmel = mag_xmin(ra + chunk, numel) - ra; \
            if (mag_unlikely(vmel <= 0)) return; \
            mag_##T##_t* pr = br + ra; \
            const mag_##T##_t* px = bx + ra; \
            const mag_##T##_t* py = by + ra; \
            mag_bnd_chk(pr, br, mag_tensor_data_size(r)); \
            mag_bnd_chk(px, bx, mag_tensor_data_size(x)); \
            mag_bnd_chk(py, by, mag_tensor_data_size(y)); \
            mag_v##name##_##T(vmel, pr, px, py); \
            return; \
        } \
        int64_t numel = xd5*xd4*xd3*xd2*xd1; \
        int64_t chunk = (numel + tc - 1)/tc; \
        int64_t ra = chunk*ti; \
        int64_t rb = mag_xmin(ra+chunk, numel); \
        if (ys0 == 1) { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                int64_t ro = ri; \
                int64_t xi1 = ro % xd1; ro /= xd1; \
                int64_t xi2 = ro % xd2; ro /= xd2; \
                int64_t xi3 = ro % xd3; ro /= xd3; \
                int64_t xi4 = ro % xd4; ro /= xd4; \
                int64_t xi5 = ro; \
                int64_t yi5 = xi5 % yd5; \
                int64_t yi4 = xi4 % yd4; \
                int64_t yi3 = xi3 % yd3; \
                int64_t yi2 = xi2 % yd2; \
                int64_t yi1 = xi1 % yd1; \
                mag_##T##_t* pr = br + xi5*rs5 + xi4*rs4 + xi3*rs3 + xi2*rs2 + xi1*rs1; \
                const mag_##T##_t* px = bx + xi5*xs5 + xi4*xs4 + xi3*xs3 + xi2*xs2 + xi1*xs1; \
                const mag_##T##_t* py = by + yi5*ys5 + yi4*ys4 + yi3*ys3 + yi2*ys2 + yi1*ys1; \
                mag_bnd_chk(py, by, mag_tensor_data_size(y)); \
                int64_t yor = xd0 / yd0; \
                for (int64_t i=0; i < yor; ++i) { \
                    mag_##T##_t* ppr = pr + i*yd0; \
                    const mag_##T##_t* ppx = px + i*yd0; \
                    mag_bnd_chk(ppr, br, mag_tensor_data_size(r)); \
                    mag_bnd_chk(ppx, bx, mag_tensor_data_size(x)); \
                    mag_v##name##_##T(yd0, ppr, ppx, py); \
                } \
            } \
        } else { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                int64_t ro = ri; \
                int64_t xi1 = ro % xd1; ro /= xd1; \
                int64_t xi2 = ro % xd2; ro /= xd2; \
                int64_t xi3 = ro % xd3; ro /= xd3; \
                int64_t xi4 = ro % xd4; ro /= xd4; \
                int64_t xi5 = ro; \
                int64_t yi5 = xi5 % yd5; \
                int64_t yi4 = xi4 % yd4; \
                int64_t yi3 = xi3 % yd3; \
                int64_t yi2 = xi2 % yd2; \
                int64_t yi1 = xi1 % yd1; \
                mag_##T##_t* pr = br + xi5*rs5 + xi4*rs4 + xi3*rs3 + xi2*rs2 + xi1*rs1; \
                const mag_##T##_t* px = bx + xi5*xs5 + xi4*xs4 + xi3*xs3 + xi2*xs2 + xi1*xs1; \
                for (int64_t i=0; i < rd0; ++i) { \
                    const mag_##T##_t* py = by + yi5*ys5 + yi4*ys4 + yi3*ys3 + yi2*ys2 + yi1*ys1 + i%yd0*ys0; \
                    mag_bnd_chk(pr+i, br, mag_tensor_data_size(r)); \
                    mag_bnd_chk(px+i, bx, mag_tensor_data_size(x)); \
                    mag_bnd_chk(py, by, mag_tensor_data_size(y)); \
                    pr[i] = op((px[i]), (*py)); \
                } \
            } \
        } \
    }

#define mag_sadd(x, y) ((x)+(y))
#define mag_ssub(x, y) ((x)-(y))
#define mag_smul(x, y) ((x)*(y))
#define mag_sdiv(x, y) ((x)/(y))

mag_cpu_blas_impl_unary(e8m23, abs)
mag_cpu_blas_impl_unary(e8m23, neg)
mag_cpu_blas_impl_unary(e8m23, log)
mag_cpu_blas_impl_unary(e8m23, sqr)
mag_cpu_blas_impl_unary(e8m23, sqrt)
mag_cpu_blas_impl_unary(e8m23, sin)
mag_cpu_blas_impl_unary(e8m23, cos)
mag_cpu_blas_impl_unary(e8m23, step)
mag_cpu_blas_impl_unary(e8m23, exp)
mag_cpu_blas_impl_unary(e8m23, softmax)
mag_cpu_blas_impl_unary(e8m23, softmax_dv)
mag_cpu_blas_impl_unary(e8m23, sigmoid)
mag_cpu_blas_impl_unary(e8m23, sigmoid_dv)
mag_cpu_blas_impl_unary(e8m23, hard_sigmoid)
mag_cpu_blas_impl_unary(e8m23, silu)
mag_cpu_blas_impl_unary(e8m23, silu_dv)
mag_cpu_blas_impl_unary(e8m23, tanh)
mag_cpu_blas_impl_unary(e8m23, tanh_dv)
mag_cpu_blas_impl_unary(e8m23, relu)
mag_cpu_blas_impl_unary(e8m23, relu_dv)
mag_cpu_blas_impl_unary(e8m23, gelu)
mag_cpu_blas_impl_unary(e8m23, gelu_dv)
mag_cpu_blas_impl_unary_scalar(e8m23, add)
mag_cpu_blas_impl_unary_scalar(e8m23, sub)
mag_cpu_blas_impl_unary_scalar(e8m23, mul)
mag_cpu_blas_impl_unary_scalar(e8m23, div)
mag_cpu_blas_impl_unary_scalar(e8m23, pow)
mag_cpu_blas_impl_binary(e8m23, add, mag_sadd)
mag_cpu_blas_impl_binary(e8m23, sub, mag_ssub)
mag_cpu_blas_impl_binary(e8m23, mul, mag_smul)
mag_cpu_blas_impl_binary(e8m23, div, mag_sdiv)

#undef mag_sadd
#undef mag_ssub
#undef mag_smul
#undef mag_sdiv

#define mag_sadd(x, y) mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x)+mag_e5m10_cvt_e8m23(y))
#define mag_ssub(x, y) mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x)-mag_e5m10_cvt_e8m23(y))
#define mag_smul(x, y) mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x)*mag_e5m10_cvt_e8m23(y))
#define mag_sdiv(x, y) mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x)/mag_e5m10_cvt_e8m23(y))

mag_cpu_blas_impl_unary(e5m10, abs)
mag_cpu_blas_impl_unary(e5m10, neg)
mag_cpu_blas_impl_unary(e5m10, log)
mag_cpu_blas_impl_unary(e5m10, sqr)
mag_cpu_blas_impl_unary(e5m10, sqrt)
mag_cpu_blas_impl_unary(e5m10, sin)
mag_cpu_blas_impl_unary(e5m10, cos)
mag_cpu_blas_impl_unary(e5m10, step)
mag_cpu_blas_impl_unary(e5m10, exp)
mag_cpu_blas_impl_unary(e5m10, softmax)
mag_cpu_blas_impl_unary(e5m10, softmax_dv)
mag_cpu_blas_impl_unary(e5m10, sigmoid)
mag_cpu_blas_impl_unary(e5m10, sigmoid_dv)
mag_cpu_blas_impl_unary(e5m10, hard_sigmoid)
mag_cpu_blas_impl_unary(e5m10, silu)
mag_cpu_blas_impl_unary(e5m10, silu_dv)
mag_cpu_blas_impl_unary(e5m10, tanh)
mag_cpu_blas_impl_unary(e5m10, tanh_dv)
mag_cpu_blas_impl_unary(e5m10, relu)
mag_cpu_blas_impl_unary(e5m10, relu_dv)
mag_cpu_blas_impl_unary(e5m10, gelu)
mag_cpu_blas_impl_unary(e5m10, gelu_dv)
mag_cpu_blas_impl_unary_scalar(e5m10, add)
mag_cpu_blas_impl_unary_scalar(e5m10, sub)
mag_cpu_blas_impl_unary_scalar(e5m10, mul)
mag_cpu_blas_impl_unary_scalar(e5m10, div)
mag_cpu_blas_impl_unary_scalar(e5m10, pow)
mag_cpu_blas_impl_binary(e5m10, add, mag_sadd)
mag_cpu_blas_impl_binary(e5m10, sub, mag_ssub)
mag_cpu_blas_impl_binary(e5m10, mul, mag_smul)
mag_cpu_blas_impl_binary(e5m10, div, mag_sdiv)

#undef mag_sadd
#undef mag_ssub
#undef mag_smul
#undef mag_sdiv

#undef mag_cpu_blas_impl_unary_scalar
#undef mag_cpu_blas_impl_unary
#undef mag_cpu_blas_impl_binary

#if defined(MAG_OPENBLAS) || defined(MAG_ACCELERATE)

static void MAG_HOTPROC mag_blas_matmul_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    const mag_tensor_t* y = r->op_inputs[1];
    mag_e8m23_t* br = mag_e8m23p_mut(r);
    const mag_e8m23_t* bx = mag_e8m23p(x);
    const mag_e8m23_t* by = mag_e8m23p(y);
    mag_load_local_storage_group(r, rd, shape);
    mag_load_local_storage_group(r, rs, strides);
    mag_load_local_storage_group(x, xd, shape);
    mag_load_local_storage_group(x, xs, strides);
    mag_load_local_storage_group(y, yd, shape);
    mag_load_local_storage_group(y, ys, strides);
    int64_t ti = payload->thread_idx;
    if (ti != 0) return;
    mag_assert2(mag_tensor_is_contiguous(x) && mag_tensor_is_contiguous(y) && mag_tensor_is_contiguous(r));
    bool trans_a = mag_tensor_is_transposed(x);
    if (x->op == MAG_OP_CLONE && x->op_inputs[0]) trans_a |= mag_tensor_is_transposed(x->op_inputs[0]);
    bool trans_b = mag_tensor_is_transposed(y);
    if (y->op == MAG_OP_CLONE && y->op_inputs[0]) trans_b |= mag_tensor_is_transposed(y->op_inputs[0]);
    int64_t b2 = yd2/xd2;
    int64_t b3 = yd3/xd3;
    int64_t b4 = yd4/xd4;
    int64_t b5 = yd5/xd5;
    for (int64_t i5=0; i5 < xd5; ++i5) {
        for (int64_t i4=0; i4 < xd4; ++i4) {
            for (int64_t i3=0; i3 < xd3; ++i3) {
                for (int64_t i2=0; i2 < xd2; ++i2) {
                    int64_t xi5 = i5/b5;
                    int64_t xi4 = i4/b4;
                    int64_t xi3 = i3/b3;
                    int64_t xi2 = i2/b2;
                    const mag_e8m23_t* px = bx + xi5*xs5 + xi4*xs4 + xi3*xs3 + xi2*xs2;
                    const mag_e8m23_t* py = by + i5*ys5 + i4*ys4 + i3*ys3 + i2*ys2;
                    mag_e8m23_t* pr = br + i5*rs5 + i4*rs4 + i3*rs3 + i2*rs2;
                    mag_bnd_chk(pr, br, mag_tensor_data_size(r));
                    mag_bnd_chk(px, bx, mag_tensor_data_size(x));
                    mag_bnd_chk(py, by, mag_tensor_data_size(y));
                    cblas_sgemm(
                        CblasRowMajor,
                        trans_a ? CblasTrans : CblasNoTrans,
                        trans_b ? CblasTrans : CblasNoTrans,
                        rd0,
                        yd1,
                        xd1,
                        1.0f,
                        px, xd1,
                        py, yd1,
                        0.0f,
                        pr, yd1
                    );
                }
            }
        }
    }
}

#else

static void MAG_HOTPROC mag_sgemm(
    const bool trans_a, const bool trans_b,
    const int64_t M, const int64_t N, const int64_t K,
    const mag_e8m23_t* A, const int64_t lda,
    const mag_e8m23_t* B, const int64_t ldb,
    mag_e8m23_t* C, const int64_t ldc
) {
    for (int64_t i=0; i < M; i++) {
        for (int64_t j=0; j < N; j++) {
            mag_e8m23_t sum = 0.0f;
            if (!trans_a && !trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += A[i*lda + p] * B[p*ldb + j];
                }
            } else if (!trans_a && trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += A[i*lda + p] * B[j*ldb + p];
                }
            } else if (trans_a && !trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += A[p*lda + i] * B[p*ldb + j];
                }
            } else {
                for (int64_t p=0; p < K; p++) {
                    sum += A[p*lda + i] * B[j*ldb + p];
                }
            }
            C[i*ldc + j] = sum * C[i*ldc + j];
        }
    }
}

static void MAG_HOTPROC mag_blas_matmul_e8m23(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    const mag_tensor_t* y = r->op_inputs[1];
    mag_e8m23_t* br = mag_e8m23p_mut(r);
    const mag_e8m23_t* bx = mag_e8m23p(x);
    const mag_e8m23_t* by = mag_e8m23p(y);
    mag_load_local_storage_group(r, rd, shape);
    mag_load_local_storage_group(r, rs, strides);
    mag_load_local_storage_group(x, xd, shape);
    mag_load_local_storage_group(x, xs, strides);
    mag_load_local_storage_group(y, yd, shape);
    mag_load_local_storage_group(y, ys, strides);
    int64_t ti = payload->thread_idx;
    if (ti != 0) return;
    bool trans_a = mag_tensor_is_transposed(x);
    if (x->op == MAG_OP_CLONE && x->op_inputs[0]) trans_a |= mag_tensor_is_transposed(x->op_inputs[0]);
    bool trans_b = mag_tensor_is_transposed(y);
    if (y->op == MAG_OP_CLONE && y->op_inputs[0]) trans_b |= mag_tensor_is_transposed(y->op_inputs[0]);
    memset(br, 0, mag_tensor_data_size(r));
    int64_t b2 = yd2/xd2;
    int64_t b3 = yd3/xd3;
    int64_t b4 = yd4/xd4;
    int64_t b5 = yd5/xd5;
    for (int64_t i5=0; i5 < xd5; ++i5) {
        for (int64_t i4=0; i4 < xd4; ++i4) {
            for (int64_t i3=0; i3 < xd3; ++i3) {
                for (int64_t i2=0; i2 < xd2; ++i2) {
                    int64_t xi5 = i5/b5;
                    int64_t xi4 = i4/b4;
                    int64_t xi3 = i3/b3;
                    int64_t xi2 = i2/b2;
                    const mag_e8m23_t* px = bx + xi5*xs5 + xi4*xs4 + xi3*xs3 + xi2*xs2;
                    const mag_e8m23_t* py = by + i5*ys5 + i4*ys4 + i3*ys3 + i2*ys2;
                    mag_e8m23_t* pr = br + i5*rs5 + i4*rs4 + i3*rs3 + i2*rs2;
                    mag_bnd_chk(pr, br, mag_tensor_data_size(r));
                    mag_bnd_chk(px, bx, mag_tensor_data_size(x));
                    mag_bnd_chk(py, by, mag_tensor_data_size(y));
                    mag_sgemm(
                        trans_a,
                        trans_b,
                        rd0,
                        yd1,
                        xd1,
                        px, xd1,
                        py, yd1,
                        pr, yd1
                    );
                }
            }
        }
    }
}

#endif

static void MAG_HOTPROC mag_hgemm(
    const bool trans_a, const bool trans_b,
    const int64_t M, const int64_t N, const int64_t K,
    const mag_e5m10_t* A, const int64_t lda,
    const mag_e5m10_t* B, const int64_t ldb,
    mag_e5m10_t* C, const int64_t ldc
) {
    for (int64_t i=0; i < M; i++) {
        for (int64_t j=0; j < N; j++) {
            mag_e8m23_t sum = 0.0f;
            if (!trans_a && !trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += mag_e5m10_cvt_e8m23(A[i*lda + p]) * mag_e5m10_cvt_e8m23(B[p*ldb + j]);
                }
            } else if (!trans_a && trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += mag_e5m10_cvt_e8m23(A[i*lda + p]) * mag_e5m10_cvt_e8m23(B[j*ldb + p]);
                }
            } else if (trans_a && !trans_b) {
                for (int64_t p=0; p < K; p++) {
                    sum += mag_e5m10_cvt_e8m23(A[p*lda + i]) * mag_e5m10_cvt_e8m23(B[p*ldb + j]);
                }
            } else {
                for (int64_t p=0; p < K; p++) {
                    sum += mag_e5m10_cvt_e8m23(A[p*lda + i]) * mag_e5m10_cvt_e8m23(B[j*ldb + p]);
                }
            }
            C[i*ldc + j] = mag_e8m23_cvt_e5m10(sum * mag_e5m10_cvt_e8m23(C[i*ldc + j]));
        }
    }
}
static void MAG_HOTPROC mag_blas_matmul_e5m10(const mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    const mag_tensor_t* y = r->op_inputs[1];
    mag_e5m10_t* br = mag_e5m10p_mut(r);
    const mag_e5m10_t* bx = mag_e5m10p(x);
    const mag_e5m10_t* by = mag_e5m10p(y);
    mag_load_local_storage_group(r, rd, shape);
    mag_load_local_storage_group(r, rs, strides);
    mag_load_local_storage_group(x, xd, shape);
    mag_load_local_storage_group(x, xs, strides);
    mag_load_local_storage_group(y, yd, shape);
    mag_load_local_storage_group(y, ys, strides);
    int64_t ti = payload->thread_idx;
    if (ti != 0) return;
    bool trans_a = mag_tensor_is_transposed(x);
    if (x->op == MAG_OP_CLONE && x->op_inputs[0]) trans_a |= mag_tensor_is_transposed(x->op_inputs[0]);
    bool trans_b = mag_tensor_is_transposed(y);
    if (y->op == MAG_OP_CLONE && y->op_inputs[0]) trans_b |= mag_tensor_is_transposed(y->op_inputs[0]);
    memset(br, 0, mag_tensor_data_size(r));
    int64_t b2 = yd2/xd2;
    int64_t b3 = yd3/xd3;
    int64_t b4 = yd4/xd4;
    int64_t b5 = yd5/xd5;
    for (int64_t i5=0; i5 < xd5; ++i5) {
        for (int64_t i4=0; i4 < xd4; ++i4) {
            for (int64_t i3=0; i3 < xd3; ++i3) {
                for (int64_t i2=0; i2 < xd2; ++i2) {
                    int64_t xi5 = i5/b5;
                    int64_t xi4 = i4/b4;
                    int64_t xi3 = i3/b3;
                    int64_t xi2 = i2/b2;
                    const mag_e5m10_t* px = bx + xi5*xs5 + xi4*xs4 + xi3*xs3 + xi2*xs2;
                    const mag_e5m10_t* py = by + i5*ys5 + i4*ys4 + i3*ys3 + i2*ys2;
                    mag_e5m10_t* pr = br + i5*rs5 + i4*rs4 + i3*rs3 + i2*rs2;
                    mag_bnd_chk(pr, br, mag_tensor_data_size(r));
                    mag_bnd_chk(px, bx, mag_tensor_data_size(x));
                    mag_bnd_chk(py, by, mag_tensor_data_size(y));
                    mag_hgemm(
                        trans_a,
                        trans_b,
                        rd0,
                        yd1,
                        xd1,
                        px, xd1,
                        py, yd1,
                        pr, yd1
                    );
                }
            }
        }
    }
}

#ifndef MAG_BLAS_SPECIALIZATION
#error "BLAS specialization undefined"
#endif
#ifndef MAG_BLAS_SPECIALIZATION_FEAT_REQUEST
#error "Feature request routine undefined"
#endif

#if defined(__x86_64__) || defined(_M_X64)
uint64_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST() {
    uint64_t caps = 1ull<<MAG_AMD64_CAP_SSE2; /* always required */
    #ifdef __AVX512F__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512F;
    #endif
    #ifdef __AVX512BW__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512BW;
    #endif
    #ifdef __AVX512CD__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512CD;
    #endif
    #ifdef __AVX512DQ__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512DQ;
    #endif
    #ifdef __AVX512ER__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512ER;
    #endif
    #ifdef __AVX512IFMA__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512IFMA;
    #endif
    #ifdef __AVX512PF__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512PF;
    #endif
    #ifdef __AVX512VBMI__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512VBMI;
    #endif
    #ifdef __AVX512VL__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512VL;
    #endif
    #ifdef __AVX512_4FMAPS__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_4FMAPS;
    #endif
    #ifdef __AVX512_4VNNIW__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_4VNNIW;
    #endif
    #ifdef __AVX512_FP16__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_FP16;
    #endif
    #ifdef __AVX512_BF16__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_BF16;
    #endif
    #ifdef __AVX512_BITALG__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_BITALG;
    #endif
    #ifdef __AVX512_VBMI2__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VBMI2;
    #endif
    #ifdef __AVX512_VNNI__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VNNI;
    #endif
    #ifdef __AVX512_VP2INTERSECT__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VP2INTERSECT;
    #endif
    #ifdef __AVX512_VPOPCNTDQ__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VPOPCNTDQ;
    #endif
    #ifdef __AVX__
        caps |= 1ull<<MAG_AMD64_CAP_AVX;
    #endif
    #ifdef __AVX2__
        caps |= 1ull<<MAG_AMD64_CAP_AVX2;
    #endif
    #ifdef __AVXVNNI__
       caps |= 1ull<<MAG_AMD64_CAP_AVXVNNI;
    #endif
    #ifdef __AVXVNNIINT8__
        caps |= 1ull<<MAG_AMD64_CAP_AVXVNNIINT8;
    #endif
    #ifdef __AVXVNNIINT16__
        caps |= 1ull<<MAG_AMD64_CAP_AVXVNNIINT16;
    #endif
    #ifdef __BMI__
        caps |= 1ull<<MAG_AMD64_CAP_BMI;
    #endif
    #ifdef __BMI2__
        caps |= 1ull<<MAG_AMD64_CAP_BMI2;
    #endif
    #ifdef __F16C__
        caps |= 1ull<<MAG_AMD64_CAP_F16C;
    #endif
    #ifdef __FMA__
        caps |= 1ull<<MAG_AMD64_CAP_FMA;
    #endif
    #ifdef __GFNI__
        caps |= 1ull<<MAG_AMD64_CAP_GFNI;
    #endif
    #ifdef __PCLMUL__
        caps |= 1ull<<MAG_AMD64_CAP_PCLMUL;
    #endif
    #ifdef __RDRND__
        caps |= 1ull<<MAG_AMD64_CAP_RDRND;
    #endif
    #ifdef __RDSEED__
        caps |= 1ull<<MAG_AMD64_CAP_RDSEED;
    #endif
    #ifdef __RDTSCP__
        caps |= 1ull<<MAG_AMD64_CAP_RDTSCP;
    #endif
    #ifdef __SHA__
        caps |= 1ull<<MAG_AMD64_CAP_SHA;
    #endif
    #ifdef __SSE3__
        caps |= 1ull<<MAG_AMD64_CAP_SSE3;
    #endif
    #ifdef __SSE4_1__
        caps |= 1ull<<MAG_AMD64_CAP_SSE4_1;
    #endif
    #ifdef __SSE4_2__
        caps |= 1ull<<MAG_AMD64_CAP_SSE4_2;
    #endif
    #ifdef __SSSE3__
        caps |= 1ull<<MAG_AMD64_CAP_SSSE3;
    #endif
    #ifdef __VAES__
        caps |= 1ull<<MAG_AMD64_CAP_VAES;
    #endif
    #ifdef __VPCLMULQDQ__
        caps |= 1ull<<MAG_AMD64_CAP_VPCLMULQDQ;
    #endif
    #ifdef __XSAVE__
        caps |= 1ull<<MAG_AMD64_CAP_XSAVE;
    #endif
    return caps;
}

#elif defined(__aarch64__)

uint64_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST(void) {
    uint64_t caps = 1u<<MAG_ARM64_CAP_NEON; /* Always required on arm64. */
    #ifdef __ARM_FEATURE_DOTPROD
        caps |= 1u<<MAG_ARM64_CAP_DOTPROD;
    #endif
    #ifdef __ARM_FEATURE_MATMUL_INT8
        caps |= 1u<<MAG_ARM64_CAP_I8MM;
    #endif
    #ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
        caps |= 1u<<MAG_ARM64_CAP_F16SCA;
    #endif
    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        caps |= 1u<<MAG_ARM64_CAP_F16VEC;
    #endif
    #ifdef __ARM_FEATURE_BF16
        caps |= 1u<<MAG_ARM64_CAP_BF16;
    #endif
    #ifdef __ARM_FEATURE_SVE
        caps |= 1u<<MAG_ARM64_CAP_SVE;
    #endif
    #ifdef __ARM_FEATURE_SVE2
        caps |= 1u<<MAG_ARM64_CAP_SVE2;
    #endif
    return caps;
}

#endif

static void (*const init_kernels[MAG_IOP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*, mag_kernel_context_t* ctx) = {
    [MAG_IOP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_IOP_BROADCAST] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_broadcast_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_broadcast_e5m10,
    },
    [MAG_IOP_RAND_UNIFORM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_rand_uniform_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_rand_uniform_e5m10,
    },
    [MAG_IOP_RAND_NORMAL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_rand_normal_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_rand_normal_e5m10,
    },
};

static void (*const forward_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*, mag_kernel_context_t* ctx) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_clone,
        [MAG_DTYPE_E5M10] = &mag_blas_clone,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mean_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mean_e5m10,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_min_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_min_e5m10,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_max_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_max_e5m10,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sum_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sum_e5m10,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_abs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_abs_e5m10,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_neg_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_neg_e5m10,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_log_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_log_e5m10,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqr_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqr_e5m10,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqrt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqrt_e5m10,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sin_e5m10,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_cos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_cos_e5m10,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_step_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_step_e5m10,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_exp_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_exp_e5m10,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_e5m10,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_dv_e5m10,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_e5m10,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_dv_e5m10,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_hard_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_hard_sigmoid_e5m10,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_e5m10,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_dv_e5m10,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_e5m10,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_dv_e5m10,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_e5m10,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_dv_e5m10,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_e5m10,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_dv_e5m10,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_E8M23] = &mag_blas_add_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_add_e5m10,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sub_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sub_e5m10,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mul_e5m10,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_div_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_div_e5m10,
    },
    [MAG_OP_ADDS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_adds_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_adds_e5m10,
    },
    [MAG_OP_SUBS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_subs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_subs_e5m10,
    },
    [MAG_OP_MULS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_muls_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_muls_e5m10,
    },
    [MAG_OP_DIVS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_divs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_divs_e5m10,
    },
    [MAG_OP_POWS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_pows_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_pows_e5m10,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_matmul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_matmul_e5m10,
    },
};

static uint32_t (*const pre_forward_kernels[MAG_OP__NUM])(mag_kernel_context_t*) = {
    [MAG_OP_NOP] = NULL,
    [MAG_OP_CLONE] = NULL,
    [MAG_OP_VIEW] = NULL,
    [MAG_OP_TRANSPOSE] = NULL,
    [MAG_OP_PERMUTE] = NULL,
    [MAG_OP_MEAN] = NULL,
    [MAG_OP_MIN] = NULL,
    [MAG_OP_MAX] = NULL,
    [MAG_OP_SUM] = NULL,
    [MAG_OP_ABS] = NULL,
    [MAG_OP_NEG] = NULL,
    [MAG_OP_LOG] = NULL,
    [MAG_OP_SQR] = NULL,
    [MAG_OP_SQRT] = NULL,
    [MAG_OP_SIN] = NULL,
    [MAG_OP_COS] = NULL,
    [MAG_OP_STEP] = NULL,
    [MAG_OP_EXP] = NULL,
    [MAG_OP_SOFTMAX] = NULL,
    [MAG_OP_SOFTMAX_DV] = NULL,
    [MAG_OP_SIGMOID] = NULL,
    [MAG_OP_SIGMOID_DV] = NULL,
    [MAG_OP_HARD_SIGMOID] = NULL,
    [MAG_OP_SILU] = NULL,
    [MAG_OP_SILU_DV] = NULL,
    [MAG_OP_TANH] = NULL,
    [MAG_OP_TANH_DV] = NULL,
    [MAG_OP_RELU] = NULL,
    [MAG_OP_RELU_DV] = NULL,
    [MAG_OP_GELU] = NULL,
    [MAG_OP_GELU_DV] = NULL,
    [MAG_OP_ADD] = NULL,
    [MAG_OP_SUB] = NULL,
    [MAG_OP_MUL] = NULL,
    [MAG_OP_DIV] = NULL,
    [MAG_OP_ADDS] = NULL,
    [MAG_OP_SUBS] = NULL,
    [MAG_OP_MULS] = NULL,
    [MAG_OP_DIVS] = NULL,
    [MAG_OP_POWS] = NULL,
    [MAG_OP_MATMUL] = NULL,
};

static void (*const post_forward_kernels[MAG_OP__NUM])(mag_kernel_context_t*) = {
    [MAG_OP_NOP] = NULL,
    [MAG_OP_CLONE] = NULL,
    [MAG_OP_VIEW] = NULL,
    [MAG_OP_TRANSPOSE] = NULL,
    [MAG_OP_PERMUTE] = NULL,
    [MAG_OP_MEAN] = NULL,
    [MAG_OP_MIN] = NULL,
    [MAG_OP_MAX] = NULL,
    [MAG_OP_SUM] = NULL,
    [MAG_OP_ABS] = NULL,
    [MAG_OP_NEG] = NULL,
    [MAG_OP_LOG] = NULL,
    [MAG_OP_SQR] = NULL,
    [MAG_OP_SQRT] = NULL,
    [MAG_OP_SIN] = NULL,
    [MAG_OP_COS] = NULL,
    [MAG_OP_STEP] = NULL,
    [MAG_OP_EXP] = NULL,
    [MAG_OP_SOFTMAX] = NULL,
    [MAG_OP_SOFTMAX_DV] = NULL,
    [MAG_OP_SIGMOID] = NULL,
    [MAG_OP_SIGMOID_DV] = NULL,
    [MAG_OP_HARD_SIGMOID] = NULL,
    [MAG_OP_SILU] = NULL,
    [MAG_OP_SILU_DV] = NULL,
    [MAG_OP_TANH] = NULL,
    [MAG_OP_TANH_DV] = NULL,
    [MAG_OP_RELU] = NULL,
    [MAG_OP_RELU_DV] = NULL,
    [MAG_OP_GELU] = NULL,
    [MAG_OP_GELU_DV] = NULL,
    [MAG_OP_ADD] = NULL,
    [MAG_OP_SUB] = NULL,
    [MAG_OP_MUL] = NULL,
    [MAG_OP_DIV] = NULL,
    [MAG_OP_ADDS] = NULL,
    [MAG_OP_SUBS] = NULL,
    [MAG_OP_MULS] = NULL,
    [MAG_OP_DIVS] = NULL,
    [MAG_OP_POWS] = NULL,
    [MAG_OP_MATMUL] = NULL,
};

static void (*const backward_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*, mag_kernel_context_t* ctx) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_clone,
        [MAG_DTYPE_E5M10] = &mag_blas_clone,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mean_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mean_e5m10,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_min_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_min_e5m10,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_max_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_max_e5m10,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sum_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sum_e5m10,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_abs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_abs_e5m10,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_neg_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_neg_e5m10,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_log_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_log_e5m10,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqr_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqr_e5m10,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqrt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqrt_e5m10,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sin_e5m10,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_cos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_cos_e5m10,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_step_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_step_e5m10,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_exp_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_exp_e5m10,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_e5m10,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_dv_e5m10,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_e5m10,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_dv_e5m10,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_hard_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_hard_sigmoid_e5m10,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_e5m10,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_dv_e5m10,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_e5m10,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_dv_e5m10,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_e5m10,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_dv_e5m10,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_e5m10,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_dv_e5m10,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_E8M23] = &mag_blas_add_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_add_e5m10,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sub_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sub_e5m10,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mul_e5m10,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_div_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_div_e5m10,
    },
    [MAG_OP_ADDS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_adds_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_adds_e5m10,
    },
    [MAG_OP_SUBS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_subs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_subs_e5m10,
    },
    [MAG_OP_MULS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_muls_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_muls_e5m10,
    },
    [MAG_OP_DIVS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_divs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_divs_e5m10,
    },
    [MAG_OP_POWS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_pows_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_pows_e5m10,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_matmul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_matmul_e5m10,
    },
};

static uint32_t (*const pre_backward_kernels[MAG_OP__NUM])(mag_kernel_context_t*) = {
    [MAG_OP_NOP] = NULL,
    [MAG_OP_CLONE] = NULL,
    [MAG_OP_VIEW] = NULL,
    [MAG_OP_TRANSPOSE] = NULL,
    [MAG_OP_PERMUTE] = NULL,
    [MAG_OP_MEAN] = NULL,
    [MAG_OP_MIN] = NULL,
    [MAG_OP_MAX] = NULL,
    [MAG_OP_SUM] = NULL,
    [MAG_OP_ABS] = NULL,
    [MAG_OP_NEG] = NULL,
    [MAG_OP_LOG] = NULL,
    [MAG_OP_SQR] = NULL,
    [MAG_OP_SQRT] = NULL,
    [MAG_OP_SIN] = NULL,
    [MAG_OP_COS] = NULL,
    [MAG_OP_STEP] = NULL,
    [MAG_OP_EXP] = NULL,
    [MAG_OP_SOFTMAX] = NULL,
    [MAG_OP_SOFTMAX_DV] = NULL,
    [MAG_OP_SIGMOID] = NULL,
    [MAG_OP_SIGMOID_DV] = NULL,
    [MAG_OP_HARD_SIGMOID] = NULL,
    [MAG_OP_SILU] = NULL,
    [MAG_OP_SILU_DV] = NULL,
    [MAG_OP_TANH] = NULL,
    [MAG_OP_TANH_DV] = NULL,
    [MAG_OP_RELU] = NULL,
    [MAG_OP_RELU_DV] = NULL,
    [MAG_OP_GELU] = NULL,
    [MAG_OP_GELU_DV] = NULL,
    [MAG_OP_ADD] = NULL,
    [MAG_OP_SUB] = NULL,
    [MAG_OP_MUL] = NULL,
    [MAG_OP_DIV] = NULL,
    [MAG_OP_ADDS] = NULL,
    [MAG_OP_SUBS] = NULL,
    [MAG_OP_MULS] = NULL,
    [MAG_OP_DIVS] = NULL,
    [MAG_OP_POWS] = NULL,
    [MAG_OP_MATMUL] = NULL,
};

static void (*const post_backward_kernels[MAG_OP__NUM])(mag_kernel_context_t*) = {
    [MAG_OP_NOP] = NULL,
    [MAG_OP_CLONE] = NULL,
    [MAG_OP_VIEW] = NULL,
    [MAG_OP_TRANSPOSE] = NULL,
    [MAG_OP_PERMUTE] = NULL,
    [MAG_OP_MEAN] = NULL,
    [MAG_OP_MIN] = NULL,
    [MAG_OP_MAX] = NULL,
    [MAG_OP_SUM] = NULL,
    [MAG_OP_ABS] = NULL,
    [MAG_OP_NEG] = NULL,
    [MAG_OP_LOG] = NULL,
    [MAG_OP_SQR] = NULL,
    [MAG_OP_SQRT] = NULL,
    [MAG_OP_SIN] = NULL,
    [MAG_OP_COS] = NULL,
    [MAG_OP_STEP] = NULL,
    [MAG_OP_EXP] = NULL,
    [MAG_OP_SOFTMAX] = NULL,
    [MAG_OP_SOFTMAX_DV] = NULL,
    [MAG_OP_SIGMOID] = NULL,
    [MAG_OP_SIGMOID_DV] = NULL,
    [MAG_OP_HARD_SIGMOID] = NULL,
    [MAG_OP_SILU] = NULL,
    [MAG_OP_SILU_DV] = NULL,
    [MAG_OP_TANH] = NULL,
    [MAG_OP_TANH_DV] = NULL,
    [MAG_OP_RELU] = NULL,
    [MAG_OP_RELU_DV] = NULL,
    [MAG_OP_GELU] = NULL,
    [MAG_OP_GELU_DV] = NULL,
    [MAG_OP_ADD] = NULL,
    [MAG_OP_SUB] = NULL,
    [MAG_OP_MUL] = NULL,
    [MAG_OP_DIV] = NULL,
    [MAG_OP_ADDS] = NULL,
    [MAG_OP_SUBS] = NULL,
    [MAG_OP_MULS] = NULL,
    [MAG_OP_DIVS] = NULL,
    [MAG_OP_POWS] = NULL,
    [MAG_OP_MATMUL] = NULL,
};

void MAG_BLAS_SPECIALIZATION(mag_kernel_registry_t* kernels) {
    for (unsigned i=0; i < MAG_IOP__NUM; ++i)
        for (unsigned j=0; j < MAG_DTYPE__NUM; ++j)
            kernels->init[i][j] = init_kernels[i][j];

    for (unsigned i=0; i < MAG_OP__NUM; ++i) {
        for (unsigned j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->fwd[i][j] = forward_kernels[i][j];
            kernels->bwd[i][j] = backward_kernels[i][j];
        }
        kernels->fwd_pre[i] = pre_forward_kernels[i];
        kernels->fwd_post[i] = post_forward_kernels[i];
        kernels->bwd_pre[i] = pre_backward_kernels[i];
        kernels->bwd_post[i] = post_backward_kernels[i];
    }
}

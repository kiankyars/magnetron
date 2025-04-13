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

#define mag_e8m23p(t) ((const mag_e8m23_t*)(t)->storage->base)
#define mag_e8m23p_mut(t) ((mag_e8m23_t*)(t)->storage->base)
#define mag_e5m10p(t) ((const mag_e5m10_t*)(t)->storage->base)
#define mag_e5m10p_mut(t) ((mag_e5m10_t*)(t)->storage->base)

static void MAG_HOTPROC mag_vector_cast_mag_e8m23_cvt_e5m10(int64_t n, const mag_e8m23_t* __restrict src, mag_e5m10_t* __restrict dst) {
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < n; i += 4) {
            float32x4_t v = vld1q_f32(src+i);
            vst1_u16((uint16_t*)dst+i, vcvt_f16_f32(v));
        }
    #endif
    for (; i < n; ++i) {
        dst[i] = mag_e8m23_cvt_e5m10(src[i]);
    }
}

static void MAG_HOTPROC mag_vector_cast_mag_e5m10_cvt_e8m23(int64_t n, const mag_e5m10_t* __restrict src, mag_e8m23_t* __restrict dst) {
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < n; i += 4) {
            uint16x4_t v = vld1_u16((const uint16_t*)src+i);
            vst1q_f32(dst+i, vcvt_f32_f16(v));
        }
    #endif
    for (; i < n; ++i) {
        dst[i] = mag_e5m10_cvt_e8m23(src[i]);
    }
}

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
    mag_prng_gen_uniform_vec_e8m23(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n-1; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_e8m23_t* u1 = o+i;
        mag_e8m23_t* u2 = o+i+1;
        mag_e8m23_t mag = std*sqrtf(-2.0f*logf(*u1));
        mag_e8m23_t y0 = mag*cosf(MAG_TAU**u2) + mean;
        mag_e8m23_t y1 = mag*sinf(MAG_TAU**u2) + mean;
        *u1 = y0;
        *u2 = y1;
    }
    if (n & 1) {  /* Handle odd numel */
        mag_e8m23_t u[2];
        mag_prng_gen_uniform_vec_e8m23(prng, u, sizeof(u)/sizeof(*u), 0.0f, 1.0f);
        o[n-1] = std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean;
    }
}

/* Generate N normal (Gauss) distributed floats. */
static void MAG_HOTPROC mag_prng_gen_normal_vec_e5m10(mag_prng_state_t* prng, mag_e5m10_t* o, int64_t n, mag_e8m23_t mean, mag_e8m23_t std) {
    mag_prng_gen_uniform_vec_e5m10(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_e8m23_t u1 = mag_e5m10_cvt_e8m23(o[i]);
        mag_e8m23_t u2 = mag_e5m10_cvt_e8m23(o[i+1]);
        mag_e8m23_t mag = std*sqrtf(-2.0f*logf(u1));
        mag_e8m23_t y0 = mag*cosf(MAG_TAU*u2) + mean;
        mag_e8m23_t y1 = mag*sinf(MAG_TAU*u2) + mean;
        o[i] = mag_e8m23_cvt_e5m10(y0);
        o[i+1] = mag_e8m23_cvt_e5m10(y1);
    }
    if (n & 1) {  /* Handle odd numel */
        mag_e8m23_t u[2];
        mag_prng_gen_uniform_vec_e8m23(prng, u, sizeof(u)/sizeof(*u), 0.0f, 1.0f);
        o[n-1] = mag_e8m23_cvt_e5m10(std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean);
    }
}

static void mag_blas_nop(const mag_compute_payload_t* payload) { (void)payload; }

static inline int64_t mag_offset_from_flat(const mag_tensor_t* t, int64_t idx) {
    int64_t off = 0;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t coord = idx % t->shape[d];
        idx /= t->shape[d];
        off += coord * t->strides[d];
    }
    return off;
}

static void mag_blas_clone(const mag_compute_payload_t* payload) {
    mag_tensor_t*  r  = payload->node;
    const mag_tensor_t* x  = r->op_inputs[0];
    mag_e8m23_t* br = mag_e8m23p_mut(r);
    const mag_e8m23_t* bx = mag_e8m23p(x);
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static void mag_blas_init_broadcast_e8m23(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e8m23_t xi = mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    if (xi == 0.0f) {
        memset(b_r, 0, mag_tensor_get_data_size(r));
        return;
    }
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_broadcast_e5m10(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e5m10_t xi = mag_e8m23_cvt_e5m10(mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]));
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_rand_uniform_e8m23(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e8m23_t min = mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_e8m23_t max = mag_opp_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e8m23(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_uniform_e5m10(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e8m23_t min = mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_e8m23_t max = mag_opp_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e5m10(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_normal_e8m23(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e8m23_t mean = mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_e8m23_t stddev = mag_opp_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_e8m23_t* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e8m23(payload->local_prng, b_r, numel, mean, stddev);
}

static void mag_blas_init_rand_normal_e5m10(const mag_compute_payload_t* payload) {
    mag_tensor_t* r = payload->node;
    mag_e8m23_t mean = mag_opp_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_e8m23_t stddev = mag_opp_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_e5m10_t* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e5m10(payload->local_prng, b_r, numel, mean, stddev);
}

#define mag_cpu_blas_impl_binary(T, OP) \
    static void MAG_HOTPROC mag_blas_##OP##_##T( \
        const mag_compute_payload_t* payload) { \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        const mag_tensor_t* y = r->op_inputs[1]; \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        const mag_##T##_t* by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t start = ti * chunk; \
        int64_t end = mag_xmin(start + chunk, total); \
        for (int64_t idx = start; idx < end; idx++) { \
            int64_t tmp = idx; \
            int64_t roff = 0; \
            int64_t xoff = 0; \
            int64_t yoff = 0; \
            for (int64_t d=r->rank-1; d >= 0; --d) { \
                int64_t coord = tmp % r->shape[d]; \
                tmp /= r->shape[d]; \
                int64_t xcoord = (x->shape[d] == 1) ? 0 : coord; \
                int64_t ycoord = (y->shape[d] == 1) ? 0 : coord; \
                roff += coord  * r->strides[d]; \
                xoff += xcoord * x->strides[d]; \
                yoff += ycoord * y->strides[d]; \
            } \
            br[roff] = mag_##T##_s##OP(bx[xoff], by[yoff]); \
        } \
    }

#define mag_cpu_blas_impl_unary(T, FUNC) \
    static void MAG_HOTPROC mag_blas_##FUNC##_##T(const mag_compute_payload_t* payload) { \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t start = ti * chunk; \
        int64_t end = mag_xmin(start + chunk, total); \
        int64_t ra = r->rank; \
        for (int64_t idx=start; idx < end; ++idx) { \
            int64_t roff = mag_offset_from_flat(r, idx); \
            int64_t tmp = idx; \
            int64_t xoff = 0; \
            for (int64_t d=ra-1; d >= 0; --d) { \
                int64_t coord = tmp % r->shape[d]; \
                tmp /= r->shape[d]; \
                int64_t xcoord = (x->shape[d] == 1) ? 0 : coord; \
                xoff += xcoord * x->strides[d]; \
            } \
            br[roff] = mag_##T##_s##FUNC(bx[xoff]); \
        } \
    }

#define mag_cpu_blas_impl_unary_scalar(T, FUNC) \
    static void MAG_HOTPROC mag_blas_##FUNC##s_##T(const mag_compute_payload_t* payload) { \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        mag_e8m23_t xi = mag_opp_unpack_e8m23_or_panic(r->op_params[0]); \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        const mag_##T##_t* bx = mag_##T##p(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1) / tc; \
        int64_t start = ti * chunk; \
        int64_t end = mag_xmin(start + chunk, total); \
        int64_t ra =  r->rank; \
        for (int64_t idx=start; idx < end; ++idx) { \
            int64_t roff = mag_offset_from_flat(r, idx); \
            int64_t tmp = idx; \
            int64_t xoff = 0; \
            for (int64_t d=ra-1; d >= 0; --d) { \
                int64_t coord = tmp % r->shape[d]; \
                tmp /= r->shape[d]; \
                int64_t xcoord = (x->shape[d] == 1) ? 0 : coord; \
                xoff += xcoord * x->strides[d]; \
            } \
            br[roff] = mag_##T##_s##FUNC(bx[xoff], xi); \
        } \
    }

#define mag_cpu_blas_impl_reduce(T, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_blas_##FUNC##_##T(const mag_compute_payload_t* payload) { \
        mag_tensor_t* r = payload->node; \
        const mag_tensor_t* x = r->op_inputs[0]; \
        if (payload->thread_idx != 0) return; \
        const mag_##T##_t* bx = mag_##T##p(x); \
        mag_##T##_t* br = mag_##T##p_mut(r); \
        ACC_T acc = (INIT_EXPR); \
        for (int64_t i=0; i < x->numel; ++i) { \
            int64_t off = mag_offset_from_flat(x, i); \
            UPDATE_STMT; \
        } \
        FINAL_STMT; \
    }

#define mag_e8m23_sadd(x, y) ((x)+(y))
#define mag_e8m23_ssub(x, y) ((x)-(y))
#define mag_e8m23_smul(x, y) ((x)*(y))
#define mag_e8m23_sdiv(x, y) ((x)/(y))
#define mag_e8m23_spow(x, y) (powf((x), (y)))
#define mag_e8m23_sabs(x) (fabs(x))
#define mag_e8m23_sneg(x) (-(x))
#define mag_e8m23_slog(x) (logf(x))
#define mag_e8m23_ssqr(x) ((x)*(x))
#define mag_e8m23_ssqrt(x) (sqrtf(x))
#define mag_e8m23_ssin(x) (sinf(x))
#define mag_e8m23_scos(x) (cosf(x))
#define mag_e8m23_sstep(x) ((x) > 0.0f ? 1.0f : 0.0f)
#define mag_e8m23_sexp(x) (expf(x))
#define mag_e8m23_ssoftmax(x) (expf(x))
#define mag_e8m23_ssoftmax_dv(x) (expf(x))
#define mag_e8m23_ssigmoid(x) (1.0f / (1.0f + expf(-(x))))
#define mag_e8m23_ssigmoid_dv(x) (mag_e8m23_ssigmoid(x) * (1.0f - mag_e8m23_ssigmoid(x)))
#define mag_e8m23_shard_sigmoid(x) (1.0f / (1.0f + expf(-(x))))
#define mag_e8m23_ssilu(x) (x * mag_e8m23_ssigmoid(x))
#define mag_e8m23_ssilu_dv(x) (mag_e8m23_ssigmoid(x) + x * mag_e8m23_ssigmoid(x))
#define mag_e8m23_stanh(x) (tanhf(x))
#define mag_e8m23_stanh_dv(x) (1.0f - mag_e8m23_stanh(x) * mag_e8m23_stanh(x))
#define mag_e8m23_srelu(x) ((x) > 0.0f ? (x) : 0.0f)
#define mag_e8m23_srelu_dv(x) ((x) > 0.0f ? 1.0f : 0.0f)
#define mag_e8m23_sgelu(x) ((x) * 0.5f * (1.0f + mag_e8m23_stanh(x)))
#define mag_e8m23_sgelu_dv(x) (0.5f * (1.0f + mag_e8m23_stanh(x)) + 0.5f * (x) * (1.0f - mag_e8m23_stanh(x) * mag_e8m23_stanh(x)))

mag_cpu_blas_impl_unary(e8m23, abs)
mag_cpu_blas_impl_unary(e8m23, neg)
mag_cpu_blas_impl_unary(e8m23, log)
mag_cpu_blas_impl_unary(e8m23, sqr)
mag_cpu_blas_impl_unary(e8m23, sqrt)
mag_cpu_blas_impl_unary(e8m23, sin)
mag_cpu_blas_impl_unary(e8m23, cos)
mag_cpu_blas_impl_unary(e8m23, step)
mag_cpu_blas_impl_unary(e8m23, exp)
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
mag_cpu_blas_impl_binary(e8m23, add)
mag_cpu_blas_impl_binary(e8m23, sub)
mag_cpu_blas_impl_binary(e8m23, mul)
mag_cpu_blas_impl_binary(e8m23, div)
mag_cpu_blas_impl_reduce( \
    e8m23, sum, mag_e11m52_t, 0.0, \
    acc += (mag_e11m52_t)bx[off];, \
    *br = (mag_e8m23_t)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, mean, mag_e11m52_t, 0.0, \
    acc += (mag_e11m52_t)bx[off];, \
    acc /= (mag_e11m52_t)x->numel; *br = (mag_e8m23_t)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, min, mag_e8m23_t, INFINITY, \
    acc = fminf(acc, bx[off]);, \
    *br = acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, max, mag_e8m23_t, -INFINITY, \
    acc = fmaxf(acc, bx[off]);, \
    *br = acc; )

static void MAG_HOTPROC mag_blas_softmax_e8m23(const mag_compute_payload_t* payload) {
        mag_tensor_t* r = payload->node;
        const mag_tensor_t* x = r->op_inputs[0];
        mag_e8m23_t* br = mag_e8m23p_mut(r);
        const mag_e8m23_t* bx = mag_e8m23p(x);
        int64_t last_dim = r->shape[r->rank-1];
        int64_t num_rows = r->numel / last_dim;
        int64_t tc = payload->thread_num;
        int64_t ti = payload->thread_idx;
        int64_t rows_per_thread = (num_rows + tc - 1)/tc;
        int64_t start_row = ti * rows_per_thread;
        int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
        for (int64_t row = start_row; row < end_row; ++row) {
            const mag_e8m23_t* row_in = bx + row * last_dim;
            mag_e8m23_t* row_out = br + row * last_dim;
            mag_e8m23_t max_val = row_in[0];
            for (int64_t i = 1; i < last_dim; ++i) {
                if (row_in[i] > max_val) {
                    max_val = row_in[i];
                }
            }
            mag_e8m23_t sum = 0.0f;
            for (int64_t i=0; i < last_dim; ++i) {
                row_out[i] = expf(row_in[i] - max_val);
                sum += row_out[i];
            }
            for (int64_t i=0; i < last_dim; ++i) {
                row_out[i] /= sum;
            }
        }
    }

#define mag_e5m10_sadd(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_sadd(mag_e5m10_cvt_e8m23(x), mag_e5m10_cvt_e8m23(y)))
#define mag_e5m10_ssub(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_ssub(mag_e5m10_cvt_e8m23(x), mag_e5m10_cvt_e8m23(y)))
#define mag_e5m10_smul(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_smul(mag_e5m10_cvt_e8m23(x), mag_e5m10_cvt_e8m23(y)))
#define mag_e5m10_sdiv(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_sdiv(mag_e5m10_cvt_e8m23(x), mag_e5m10_cvt_e8m23(y)))
#define mag_e5m10_spow(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_spow(mag_e5m10_cvt_e8m23(x), mag_e5m10_cvt_e8m23(y)))
#define mag_e5m10_sabs(x) mag_e8m23_cvt_e5m10(mag_e8m23_sabs(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sneg(x) mag_e8m23_cvt_e5m10(mag_e8m23_sneg(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_slog(x) mag_e8m23_cvt_e5m10(mag_e8m23_slog(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssqr(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssqr(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssqrt(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssqrt(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssin(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssin(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_scos(x) mag_e8m23_cvt_e5m10(mag_e8m23_scos(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sstep(x) mag_e8m23_cvt_e5m10(mag_e8m23_sstep(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sexp(x) mag_e8m23_cvt_e5m10(mag_e8m23_sexp(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssoftmax(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssoftmax(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssoftmax_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssoftmax_dv(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssigmoid(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssigmoid(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssigmoid_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssigmoid_dv(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_shard_sigmoid(x) mag_e8m23_cvt_e5m10(mag_e8m23_shard_sigmoid(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssilu(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssilu(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssilu_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssilu_dv(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_stanh(x) mag_e8m23_cvt_e5m10(mag_e8m23_stanh(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_stanh_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_stanh_dv(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_srelu(x) mag_e8m23_cvt_e5m10(mag_e8m23_srelu(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_srelu_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_srelu_dv(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sgelu(x) mag_e8m23_cvt_e5m10(mag_e8m23_sgelu(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sgelu_dv(x) mag_e8m23_cvt_e5m10(mag_e8m23_sgelu_dv(mag_e5m10_cvt_e8m23(x)))

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
//mag_cpu_blas_impl_unary_scalar(e5m10, add)
//mag_cpu_blas_impl_unary_scalar(e5m10, sub)
//mag_cpu_blas_impl_unary_scalar(e5m10, mul)
//mag_cpu_blas_impl_unary_scalar(e5m10, div)
//mag_cpu_blas_impl_unary_scalar(e5m10, pow)
mag_cpu_blas_impl_binary(e5m10, add)
mag_cpu_blas_impl_binary(e5m10, sub)
mag_cpu_blas_impl_binary(e5m10, mul)
mag_cpu_blas_impl_binary(e5m10, div)

#undef mag_cpu_blas_impl_unary_scalar
#undef mag_cpu_blas_impl_unary
#undef mag_cpu_blas_impl_binary

#define VLA(type, name, size) \
type* name = (type*)(*mag_alloc)(NULL, (size) * sizeof(type))

static int64_t mag_offset_rmn(const mag_tensor_t* t, int64_t batch, int64_t i, int64_t j) {
    int64_t ra = t->rank;
    int64_t off = 0;
    if (ra == 3) {
        off += batch * t->strides[0];
        off += i     * t->strides[1];
        off += j     * t->strides[2];
    } else if (ra == 2) {
        off += i * t->strides[0];
        off += j * t->strides[1];
    } else {                        /* rank 1: j == 0 */
        off += i * t->strides[0];
    }
    return off;
}

static void MAG_HOTPROC mag_blas_matmul_e8m23(const mag_compute_payload_t* payload) {
    if (payload->thread_idx != 0) return;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    const mag_tensor_t* y = r->op_inputs[1];
    mag_e8m23_t* br = mag_e8m23p_mut(r);
    const mag_e8m23_t* bx = mag_e8m23p(x);
    const mag_e8m23_t* by = mag_e8m23p(y);
    int64_t M, N, K;
    if (r->rank == 1) {
        M = r->shape[0];
        N = 1;
    } else {
        M = r->shape[r->rank - 2];
        N = r->shape[r->rank - 1];
    }
    K = x->shape[x->rank - 1];
    int64_t batch = (r->rank == 3) ? r->shape[0] : 1;
    int64_t bx_batch = (x->rank == 3) ? x->shape[0] : 1;
    int64_t by_batch = (y->rank == 3) ? y->shape[0] : 1;
    bool x_row = mag_tensor_is_contiguous(x) && x->strides[x->rank - 1] == 1;
    bool y_row = mag_tensor_is_contiguous(y) && (y->rank == 1 || y->strides[y->rank - 1] == 1);
    bool r_row = mag_tensor_is_contiguous(r) && (r->rank == 1 || r->strides[r->rank - 1] == 1);
    VLA(mag_e8m23_t, xbuf, M * K);
    VLA(mag_e8m23_t, ybuf, K * N);
    VLA(mag_e8m23_t, rbuf, M * N);
    for (int64_t b=0; b < batch; ++b) {
        int64_t xb = bx_batch == 1 ? 0 : b;
        int64_t yb = by_batch == 1 ? 0 : b;
        const mag_e8m23_t* px = bx + mag_offset_rmn(x, xb, 0, 0);
        const mag_e8m23_t* py = by + mag_offset_rmn(y, yb, 0, 0);
        mag_e8m23_t* pr = br + mag_offset_rmn(r,  b,  0, 0);
        const mag_e8m23_t* A = px;
        if (!x_row) { /* pack / broadcast X if needed */
            for (int64_t i=0; i < M; ++i)
                for (int64_t k=0; k < K; ++k)
                    xbuf[i*K + k] = px[mag_offset_rmn(x, xb, i, k)];
            A = xbuf;
        }
        const mag_e8m23_t* B = py;
        if (!y_row) { /* pack / broadcast Y if needed */
            for (int64_t k=0; k < K; ++k)
                for (int64_t n=0; n < N; ++n)
                    ybuf[k*N + n] = (y->rank == 1) ? py[k] : py[mag_offset_rmn(y, yb, k, n)];
            B = ybuf;
        }
        mag_e8m23_t* C = r_row ? pr : rbuf;
        for (int64_t i=0; i < M; ++i) { /* Standard SGEMM */
            for (int64_t n=0; n < N; ++n) {
                mag_e11m52_t acc = 0.0;
                for (int64_t k=0; k < K; ++k) {
                    acc += (mag_e11m52_t)A[i*K + k] *
                           (mag_e11m52_t)B[k*N + n];
                }
                C[i*N + n] = (mag_e8m23_t)acc;
            }
        }
        if (!r_row) { /* Scatter back R if needed */
            for (int64_t i=0; i < M; ++i)
                for (int64_t n=0; n < N; ++n)
                    pr[mag_offset_rmn(r, b, i, n)] = rbuf[i*N + n];
        }
    }
    (*mag_alloc)(xbuf, 0);
    (*mag_alloc)(ybuf, 0);
    (*mag_alloc)(rbuf, 0);
}

static void MAG_HOTPROC  mag_blas_repeat_back_e8m23(const mag_compute_payload_t* payload) {
    if (payload->thread_idx != 0) return;
    mag_tensor_t* r = payload->node;
    const mag_tensor_t* x = r->op_inputs[0];
    mag_e8m23_t* br = mag_e8m23p_mut(r);
    const mag_e8m23_t* bx = mag_e8m23p(x);
    for (int64_t i=0; i < r->numel; ++i)
        br[mag_offset_from_flat(r, i)] = 0;
    int64_t rx = r->rank;
    int64_t xx = x->rank;
    int64_t shift = xx - rx;
    for (int64_t flat=0; flat < x->numel; ++flat) {
        int64_t tmp = flat;
        int64_t xoff = 0;
        int64_t roff = 0;
        for (int64_t d = xx-1; d >= 0; --d) {
            int64_t coord = tmp % x->shape[d];
            tmp /= x->shape[d];
            xoff += coord * x->strides[d];
            int64_t rd = d - shift;
            if (rd >= 0) {
                int64_t rcoord = coord % r->shape[rd];
                roff += rcoord * r->strides[rd];
            }
        }
        br[roff] += bx[xoff];
    }
}


static void MAG_HOTPROC mag_blas_repeat_back_e5m10(const mag_compute_payload_t* payload) {
    mag_panic("NYI");
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

static void (*const mag_blas_lut_init_kernels[MAG_IOP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*) = {
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

static void (*const mag_blas_lut_forward_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*) = {
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
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_min_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_max_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sum_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
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
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_SUBS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_subs_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MULS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_muls_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_DIVS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_divs_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_POWS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_pows_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_matmul_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_blas_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_repeat_back_e5m10,
    },
};

static void (*const mag_blas_lut_backward_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_compute_payload_t*) = {
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
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_min_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_max_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sum_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
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
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_SUBS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_subs_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MULS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_muls_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_DIVS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_divs_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_POWS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_pows_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_matmul_e8m23,
        [MAG_DTYPE_E5M10] = NULL,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_blas_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_repeat_back_e5m10,
    },
};

mag_static_assert(MAG_DTYPE__NUM <= 255);
#define mag_dt_perm(x,y) ((((x)&255)<<8)+((y)&255))
static void MAG_HOTPROC mag_blas_vector_cast_stub(size_t nb, const void* src, mag_dtype_t src_t, void* dst, mag_dtype_t dst_t) {
    mag_assert2(dst_t != src_t); /* src and dst types must differ */
    int64_t nbs = mag_dtype_meta_of(src_t)->size;
    int64_t nbd = mag_dtype_meta_of(dst_t)->size;
    mag_assert2(((uintptr_t)src&(nbs-1)) == 0);     /* src must be aligned */
    mag_assert2(((uintptr_t)dst&(nbd-1)) == 0);     /* dst must be aligned */
    mag_assert2((nb&(nbs-1)) == 0);                 /* size must be aligned */
    int64_t n = (int64_t)nb/nbs;                    /* Byte to elem granularity. */
    switch (mag_dt_perm(src_t, dst_t)) {
        case mag_dt_perm(MAG_DTYPE_E8M23, MAG_DTYPE_E5M10): mag_vector_cast_mag_e8m23_cvt_e5m10(n, src, dst); return;
        case mag_dt_perm(MAG_DTYPE_E5M10, MAG_DTYPE_E8M23): mag_vector_cast_mag_e5m10_cvt_e8m23(n, src, dst); return;
        default: mag_panic("invalid vector cast dtypes %s -> %s", mag_dtype_meta_of(src_t)->name, mag_dtype_meta_of(dst_t)->name);
    }
}
#undef mag_dt_perm

void MAG_BLAS_SPECIALIZATION(mag_kernel_registry_t* kernels) {
    for (int i=0; i < MAG_IOP__NUM; ++i)
        for (int j=0; j < MAG_DTYPE__NUM; ++j)
            kernels->init[i][j] = mag_blas_lut_init_kernels[i][j];
    for (int i=0; i < MAG_OP__NUM; ++i) {
        for (int j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->fwd[i][j] = mag_blas_lut_forward_kernels[i][j];
            kernels->bwd[i][j] = mag_blas_lut_backward_kernels[i][j];
        }
    }
    kernels->vector_cast = &mag_blas_vector_cast_stub;
}

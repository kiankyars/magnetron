/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
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

static MAG_AINLINE mag_E5M10 mag_e8m23_cvt_e5m10(mag_E8M23 x) {
    uint16_t r;
    #ifdef __F16C__
        #ifdef _MSC_VER
            r = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0);
        #else
            r = _cvtss_sh(x, 0);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        union {
            __fp16 f;
            uint16_t u;
        } castor = {.f=(__fp16)x};
        r = castor.u;
    #else
        union {
            uint32_t u;
            mag_E8M23 f;
        } reinterpret;
        mag_E8M23 base = fabs(x)*0x1.0p+112f*0x1.0p-110f;
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
    return (mag_E5M10){.bits=r};
}

static MAG_AINLINE mag_E8M23 mag_e5m10_cvt_e8m23(mag_E5M10 x) {
    #ifdef __F16C__
        #ifdef _MSC_VER
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x.bits)));
        #else
            return _cvtsh_ss(x.bits);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        union {
            __fp16 f;
            uint16_t u;
        } castor = {.u=x.bits};
        return castor.f;
    #else
        union {
            uint32_t u;
            mag_E8M23 f;
        } reinterpret;
        uint32_t w = (uint32_t)x.bits<<16;
        uint32_t sign = w & 0x80000000u;
        uint32_t two_w = w+w;
        uint32_t offs = 0xe0u<<23;
        uint32_t t1 = (two_w>>4) + offs;
        uint32_t t2 = (two_w>>17) | (126u<<23);
        reinterpret.u = t1;
        mag_E8M23 norm_x = reinterpret.f*0x1.0p-112f;
        reinterpret.u = t2;
        mag_E8M23 denorm_x = reinterpret.f-0.5f;
        uint32_t denorm_cutoff = 1u<<27;
        uint32_t r = sign | (two_w < denorm_cutoff
            ? (reinterpret.f = denorm_x, reinterpret.u)
            : (reinterpret.f = norm_x, reinterpret.u));
        reinterpret.u = r;
        return reinterpret.f;
    #endif
}

#define mag_e8m23p(t) ((const mag_E8M23*)(t)->storage->base)
#define mag_e8m23p_mut(t) ((mag_E8M23*)(t)->storage->base)
#define mag_e5m10p(t) ((const mag_E5M10*)(t)->storage->base)
#define mag_e5m10p_mut(t) ((mag_E5M10*)(t)->storage->base)

static void MAG_HOTPROC mag_vector_cast_mag_e8m23_cvt_e5m10(int64_t n, const mag_E8M23* _Nonnull __restrict src, mag_E5M10* _Nonnull __restrict dst) {
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < n; i += 4) {
            float32x4_t v = vld1q_f32(src+i);
            vst1_f16((__fp16*)dst+i, vcvt_f16_f32(v));
        }
    #endif
    for (; i < n; ++i) {
        dst[i] = mag_e8m23_cvt_e5m10(src[i]);
    }
}

static void MAG_HOTPROC mag_vector_cast_mag_e5m10_cvt_e8m23(int64_t n, const mag_E5M10* _Nonnull __restrict src, mag_E8M23* _Nonnull __restrict dst) {
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < n; i += 4) {
            float16x4_t v = vld1_f16((const __fp16*)src+i);
            vst1q_f32(dst+i, vcvt_f32_f16(v));
        }
    #endif
    for (; i < n; ++i) {
        dst[i] = mag_e5m10_cvt_e8m23(src[i]);
    }
}

/* Generate N uniform canonical floats in [0, 1) using active algorithm and rescale to [min, max]. */
static void MAG_AINLINE mag_prng_gen_uniform_vec_e8m23(mag_PRNGState* _Nonnull prng, mag_E8M23* _Nonnull o, int64_t n, mag_E8M23 min, mag_E8M23 max) {
    mag_E8M23 rescale_uniform = max - min;
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
                o[ii] = min + rescale_uniform * (1.f/(mag_E8M23)(1<<23)*((mag_E8M23)(y>>9) + 0.5f)); /* Generate canonical and rescale. */
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
                o[ii] = min + rescale_uniform * (1.f/(mag_E8M23)(1<<23)*((mag_E8M23)(y>>9) + 0.5f)); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N uniform canonical floats in [0, 1) using active algorithm and rescale to [min, max]. */
static void MAG_AINLINE mag_prng_gen_uniform_vec_e5m10(mag_PRNGState* _Nonnull prng, mag_E5M10* _Nonnull o, int64_t n, mag_E8M23 min, mag_E8M23 max) {
    mag_E8M23 rescale_uniform = max - min;
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
                o[ii] = mag_e8m23_cvt_e5m10(min + rescale_uniform*(1.f/(mag_E8M23)(1<<23)*((mag_E8M23)(y>>9) + 0.5f))); /* Generate canonical and rescale. */
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
                o[ii] = mag_e8m23_cvt_e5m10(min + rescale_uniform*(1.f/(mag_E8M23)(1<<23)*((mag_E8M23)(y>>9) + 0.5f))); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N normal (Gauss) distributed floats. */
static void MAG_HOTPROC mag_prng_gen_normal_vec_e8m23(mag_PRNGState* _Nonnull prng, mag_E8M23* _Nonnull o, int64_t n, mag_E8M23 mean, mag_E8M23 std) {
    mag_prng_gen_uniform_vec_e8m23(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n-1; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_E8M23* u1 = o+i;
        mag_E8M23* u2 = o+i+1;
        mag_E8M23 mag = std*sqrtf(-2.0f*logf(*u1));
        mag_E8M23 y0 = mag*cosf(MAG_TAU**u2) + mean;
        mag_E8M23 y1 = mag*sinf(MAG_TAU**u2) + mean;
        *u1 = y0;
        *u2 = y1;
    }
    if (n & 1) {  /* Handle odd numel */
        mag_E8M23 u[2];
        mag_prng_gen_uniform_vec_e8m23(prng, u, sizeof(u)/sizeof(*u), 0.0f, 1.0f);
        o[n-1] = std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean;
    }
}

/* Generate N normal (Gauss) distributed floats. */
static void MAG_HOTPROC mag_prng_gen_normal_vec_e5m10(mag_PRNGState* _Nonnull prng, mag_E5M10* _Nonnull o, int64_t n, mag_E8M23 mean, mag_E8M23 std) {
    mag_prng_gen_uniform_vec_e5m10(prng, o, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < n; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_E8M23 u1 = mag_e5m10_cvt_e8m23(o[i]);
        mag_E8M23 u2 = mag_e5m10_cvt_e8m23(o[i+1]);
        mag_E8M23 mag = std*sqrtf(-2.0f*logf(u1));
        mag_E8M23 y0 = mag*cosf(MAG_TAU*u2) + mean;
        mag_E8M23 y1 = mag*sinf(MAG_TAU*u2) + mean;
        o[i] = mag_e8m23_cvt_e5m10(y0);
        o[i+1] = mag_e8m23_cvt_e5m10(y1);
    }
    if (n & 1) {  /* Handle odd numel */
        mag_E8M23 u[2];
        mag_prng_gen_uniform_vec_e8m23(prng, u, sizeof(u)/sizeof(*u), 0.0f, 1.0f);
        o[n-1] = mag_e8m23_cvt_e5m10(std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean);
    }
}

static mag_E8M23 MAG_HOTPROC mag_vdot_e8m23(int64_t numel, const mag_E8M23* _Nonnull restrict x, const mag_E8M23* _Nonnull restrict y) {
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
        mag_E8M23 sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
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
        mag_E8M23 sum = _mm512_reduce_add_ps(*acc);
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
        mag_E8M23 sum = _mm_cvtss_f32(v0);
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
            mag_E8M23 sum = _mm_cvtss_f32(*acc);
        #else
            __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(*acc, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            mag_E8M23 sum = _mm_cvtss_f32(sums);
        #endif
        for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
        return sum;
    #else
        mag_E11M52 r = 0.0;
        for (int64_t i=0; i < numel; ++i) r += (mag_E11M52)x[i] * (mag_E11M52)y[i];
        return (mag_E8M23)r;
    #endif
}

static void mag_blas_nop(const mag_CPUKernelPayload* _Nonnull payload) { (void)payload; }

static inline int64_t mag_offset_from_flat(const mag_Tensor* _Nonnull t, int64_t idx) {
    int64_t off = 0;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t coord = idx % t->shape[d];
        idx /= t->shape[d];
        off += coord * t->strides[d];
    }
    return off;
}

static void mag_blas_clone_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor*  r  = payload->node;
    const mag_Tensor* x  = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static void mag_blas_clone_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x  = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static void mag_blas_init_broadcast_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 xi = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    if (xi == 0.0f) {
        memset(b_r, 0, mag_tensor_get_data_size(r));
        return;
    }
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_broadcast_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E5M10 xi = mag_e8m23_cvt_e5m10(mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]));
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static void mag_blas_init_rand_uniform_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 min = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 max = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e8m23(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_uniform_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 min = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 max = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_uniform_vec_e5m10(payload->local_prng, b_r, numel, min, max);
}

static void mag_blas_init_rand_normal_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 mean = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 stddev = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e8m23(payload->local_prng, b_r, numel, mean, stddev);
}

static void mag_blas_init_rand_normal_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 mean = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 stddev = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_prng_gen_normal_vec_e5m10(payload->local_prng, b_r, numel, mean, stddev);
}

#define mag_cpu_blas_impl_binary(T, DT, OP) \
    static void MAG_HOTPROC mag_blas_##OP##_##T( \
        const mag_CPUKernelPayload* _Nonnull payload) { \
        mag_Tensor* r = payload->node; \
        const mag_Tensor* x = r->op_inputs[0]; \
        const mag_Tensor* y = r->op_inputs[1]; \
        mag_##DT* br = mag_##T##p_mut(r); \
        const mag_##DT* bx = mag_##T##p(x); \
        const mag_##DT* by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t start = ti * chunk; \
        int64_t end = mag_xmin(start + chunk, total); \
        int64_t rx = r->rank - x->rank; \
        int64_t ry = r->rank - y->rank; \
        for (int64_t idx = start; idx < end; ++idx) { \
            int64_t tmp = idx; \
            int64_t roff = 0, xoff = 0, yoff = 0; \
            for (int64_t d = r->rank - 1; d >= 0; --d) { \
                int64_t dim = r->shape[d]; \
                int64_t coord = tmp % dim; \
                tmp /= dim;  \
                roff += coord * r->strides[d];  \
                int64_t dx = d - rx; \
                if (dx >= 0 && x->shape[dx] > 1) { \
                    xoff += coord * x->strides[dx]; \
                } \
                int64_t dy = d - ry; \
                if (dy >= 0 && y->shape[dy] > 1) { \
                    yoff += coord * y->strides[dy]; \
                } \
            } \
            mag_bnd_chk(bx+xoff, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(by+yoff, by, mag_tensor_get_data_size(y)); \
            mag_bnd_chk(br+roff, br, mag_tensor_get_data_size(r)); \
            br[roff] = mag_##T##_s##OP(bx[xoff], by[yoff]); \
        } \
    }

#define mag_cpu_blas_impl_unary(T, DT, FUNC) \
    static void MAG_HOTPROC mag_blas_##FUNC##_##T(const mag_CPUKernelPayload* _Nonnull payload) { \
        mag_Tensor* r = payload->node; \
        const mag_Tensor* x = r->op_inputs[0]; \
        mag_##DT* br = mag_##T##p_mut(r); \
        const mag_##DT* bx = mag_##T##p(x); \
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
            mag_bnd_chk(bx+xoff, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+roff, br, mag_tensor_get_data_size(r)); \
            br[roff] = mag_##T##_s##FUNC(bx[xoff]); \
        } \
    }

#define mag_cpu_blas_impl_unary_scalar(T, DT, FUNC) \
    static void MAG_HOTPROC mag_blas_##FUNC##s_##T(const mag_CPUKernelPayload* _Nonnull payload) { \
        mag_Tensor* r = payload->node; \
        const mag_Tensor* x = r->op_inputs[0]; \
        mag_E8M23 xi = mag_op_param_unpack_e8m23_or_panic(r->op_params[0]); \
        mag_##DT* br = mag_##T##p_mut(r); \
        const mag_##DT* bx = mag_##T##p(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1) / tc; \
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
            mag_bnd_chk(bx+xoff, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+roff, br, mag_tensor_get_data_size(r)); \
            br[roff] = mag_##T##_ss##FUNC(bx[xoff], xi); \
        } \
    }

#define mag_cpu_blas_impl_reduce(T, DT, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_blas_##FUNC##_##T(const mag_CPUKernelPayload* _Nonnull payload) { \
        mag_Tensor* r = payload->node; \
        const mag_Tensor* x = r->op_inputs[0]; \
        if (payload->thread_idx != 0) return; \
        const mag_##DT* bx = mag_##T##p(x); \
        mag_##DT* br = mag_##T##p_mut(r); \
        ACC_T acc = (INIT_EXPR); \
        for (int64_t i=0; i < x->numel; ++i) { \
            int64_t off = mag_offset_from_flat(x, i); \
            mag_bnd_chk(bx+off, bx, mag_tensor_get_data_size(x)); \
            UPDATE_STMT; \
        } \
        FINAL_STMT; \
    }

#define mag_e8m23_sadd(x, y) ((x)+(y))
#define mag_e8m23_ssub(x, y) ((x)-(y))
#define mag_e8m23_smul(x, y) ((x)*(y))
#define mag_e8m23_sdiv(x, y) ((x)/(y))
#define mag_e8m23_spow(x, y) (powf((x), (y)))
#define mag_e8m23_ssadd(x, y) ((x)+(y))
#define mag_e8m23_sssub(x, y) ((x)-(y))
#define mag_e8m23_ssmul(x, y) ((x)*(y))
#define mag_e8m23_ssdiv(x, y) ((x)/(y))
#define mag_e8m23_sspow(x, y) (powf((x), (y)))
#define mag_e8m23_sabs(x) (fabs(x))
#define mag_e8m23_ssgn(x) ((x) > 0.f ? 1.f : ((x) < 0.f ? -1.f : 0.f))
#define mag_e8m23_sneg(x) (-(x))
#define mag_e8m23_slog(x) (logf(x))
#define mag_e8m23_ssqr(x) ((x)*(x))
#define mag_e8m23_ssqrt(x) (sqrtf(x))
#define mag_e8m23_ssin(x) (sinf(x))
#define mag_e8m23_scos(x) (cosf(x))
#define mag_e8m23_sstep(x) ((x) > 0.0f ? 1.0f : 0.0f)
#define mag_e8m23_sexp(x) (expf(x))
#define mag_e8m23_sfloor(x) (floorf(x))
#define mag_e8m23_sceil(x) (ceilf(x))
#define mag_e8m23_sround(x) (roundf(x))
#define mag_e8m23_ssoftmax_dv(x) (expf(x))
#define mag_e8m23_ssigmoid(x) (1.0f / (1.0f + expf(-(x))))
#define mag_e8m23_ssigmoid_dv(x) (mag_e8m23_ssigmoid(x) * (1.0f - mag_e8m23_ssigmoid(x)))
#define mag_e8m23_shard_sigmoid(x)( fminf(1.0f, fmaxf(0.0f, ((x) + 3.0f) / 6.0f)))
#define mag_e8m23_ssilu(x) (x * mag_e8m23_ssigmoid(x))
#define mag_e8m23_ssilu_dv(x) (mag_e8m23_ssigmoid(x) + x * mag_e8m23_ssigmoid(x))
#define mag_e8m23_stanh(x) (tanhf(x))
#define mag_e8m23_stanh_dv(x) (1.0f - mag_e8m23_stanh(x) * mag_e8m23_stanh(x))
#define mag_e8m23_srelu(x) ((x) > 0.0f ? (x) : 0.0f)
#define mag_e8m23_srelu_dv(x) ((x) > 0.0f ? 1.0f : 0.0f)
#define mag_e8m23_sgelu(x) ((x) * 0.5f * (1.0f + mag_e8m23_stanh(x)))
#define mag_e8m23_sgelu_dv(x) (0.5f * (1.0f + mag_e8m23_stanh(x)) + 0.5f * (x) * (1.0f - mag_e8m23_stanh(x) * mag_e8m23_stanh(x)))

mag_cpu_blas_impl_unary(e8m23, E8M23, abs)
mag_cpu_blas_impl_unary(e8m23, E8M23, sgn)
mag_cpu_blas_impl_unary(e8m23, E8M23, neg)
mag_cpu_blas_impl_unary(e8m23, E8M23, log)
mag_cpu_blas_impl_unary(e8m23, E8M23, sqr)
mag_cpu_blas_impl_unary(e8m23, E8M23, sqrt)
mag_cpu_blas_impl_unary(e8m23, E8M23, sin)
mag_cpu_blas_impl_unary(e8m23, E8M23, cos)
mag_cpu_blas_impl_unary(e8m23, E8M23, step)
mag_cpu_blas_impl_unary(e8m23, E8M23, exp)
mag_cpu_blas_impl_unary(e8m23, E8M23, floor)
mag_cpu_blas_impl_unary(e8m23, E8M23, ceil)
mag_cpu_blas_impl_unary(e8m23, E8M23, round)
mag_cpu_blas_impl_unary(e8m23, E8M23, softmax_dv)
mag_cpu_blas_impl_unary(e8m23, E8M23, sigmoid)
mag_cpu_blas_impl_unary(e8m23, E8M23, sigmoid_dv)
mag_cpu_blas_impl_unary(e8m23, E8M23, hard_sigmoid)
mag_cpu_blas_impl_unary(e8m23, E8M23, silu)
mag_cpu_blas_impl_unary(e8m23, E8M23, silu_dv)
mag_cpu_blas_impl_unary(e8m23, E8M23, tanh)
mag_cpu_blas_impl_unary(e8m23, E8M23, tanh_dv)
mag_cpu_blas_impl_unary(e8m23, E8M23, relu)
mag_cpu_blas_impl_unary(e8m23, E8M23, relu_dv)
mag_cpu_blas_impl_unary(e8m23, E8M23, gelu)
mag_cpu_blas_impl_unary(e8m23, E8M23, gelu_dv)
mag_cpu_blas_impl_unary_scalar(e8m23, E8M23, add)
mag_cpu_blas_impl_unary_scalar(e8m23, E8M23, sub)
mag_cpu_blas_impl_unary_scalar(e8m23, E8M23, mul)
mag_cpu_blas_impl_unary_scalar(e8m23, E8M23, div)
mag_cpu_blas_impl_unary_scalar(e8m23, E8M23, pow)
mag_cpu_blas_impl_binary(e8m23, E8M23, add)
mag_cpu_blas_impl_binary(e8m23, E8M23, sub)
mag_cpu_blas_impl_binary(e8m23, E8M23, mul)
mag_cpu_blas_impl_binary(e8m23, E8M23, div)
mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, sum, mag_E11M52, 0.0, \
    acc += (mag_E11M52)bx[off];, \
    *br = (mag_E8M23)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, mean, mag_E11M52, 0.0, \
    acc += (mag_E11M52)bx[off];, \
    acc /= (mag_E11M52)x->numel; *br = (mag_E8M23)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, min, mag_E8M23, INFINITY, \
    acc = fminf(acc, bx[off]);, \
    *br = acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, max, mag_E8M23, -INFINITY, \
    acc = fmaxf(acc, bx[off]);, \
    *br = acc; )

static void MAG_HOTPROC mag_blas_softmax_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t last_dim = r->shape[r->rank-1];
    int64_t num_rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rows_per_thread = (num_rows + tc - 1)/tc;
    int64_t start_row = ti * rows_per_thread;
    int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
    for (int64_t row = start_row; row < end_row; ++row) {
        const mag_E8M23* row_in = bx + row * last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
        mag_E8M23* row_out = br + row * last_dim;
        mag_E8M23 max_val = row_in[0]; /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            if (row_in[i] > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = row_in[i];
            }
        }
        mag_E8M23 sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
            row_out[i] = expf(row_in[i] - max_val); /* -max for numerical stability */
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
#define mag_e5m10_ssadd(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_sadd(mag_e5m10_cvt_e8m23(x), y))
#define mag_e5m10_sssub(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_ssub(mag_e5m10_cvt_e8m23(x), y))
#define mag_e5m10_ssmul(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_smul(mag_e5m10_cvt_e8m23(x), y))
#define mag_e5m10_ssdiv(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_sdiv(mag_e5m10_cvt_e8m23(x), y))
#define mag_e5m10_sspow(x, y) mag_e8m23_cvt_e5m10(mag_e8m23_spow(mag_e5m10_cvt_e8m23(x), y))
#define mag_e5m10_sabs(x) mag_e8m23_cvt_e5m10(mag_e8m23_sabs(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssgn(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssgn(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sneg(x) mag_e8m23_cvt_e5m10(mag_e8m23_sneg(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_slog(x) mag_e8m23_cvt_e5m10(mag_e8m23_slog(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssqr(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssqr(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssqrt(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssqrt(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_ssin(x) mag_e8m23_cvt_e5m10(mag_e8m23_ssin(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_scos(x) mag_e8m23_cvt_e5m10(mag_e8m23_scos(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sstep(x) mag_e8m23_cvt_e5m10(mag_e8m23_sstep(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sexp(x) mag_e8m23_cvt_e5m10(mag_e8m23_sexp(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sfloor(x) mag_e8m23_cvt_e5m10(mag_e8m23_sfloor(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sceil(x) mag_e8m23_cvt_e5m10(mag_e8m23_sceil(mag_e5m10_cvt_e8m23(x)))
#define mag_e5m10_sround(x) mag_e8m23_cvt_e5m10(mag_e8m23_sround(mag_e5m10_cvt_e8m23(x)))
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

mag_cpu_blas_impl_unary(e5m10, E5M10, abs)
mag_cpu_blas_impl_unary(e5m10, E5M10, sgn)
mag_cpu_blas_impl_unary(e5m10, E5M10, neg)
mag_cpu_blas_impl_unary(e5m10, E5M10, log)
mag_cpu_blas_impl_unary(e5m10, E5M10, sqr)
mag_cpu_blas_impl_unary(e5m10, E5M10, sqrt)
mag_cpu_blas_impl_unary(e5m10, E5M10, sin)
mag_cpu_blas_impl_unary(e5m10, E5M10, cos)
mag_cpu_blas_impl_unary(e5m10, E5M10, step)
mag_cpu_blas_impl_unary(e5m10, E5M10, exp)
mag_cpu_blas_impl_unary(e5m10, E5M10, floor)
mag_cpu_blas_impl_unary(e5m10, E5M10, ceil)
mag_cpu_blas_impl_unary(e5m10, E5M10, round)
mag_cpu_blas_impl_unary(e5m10, E5M10, softmax_dv)
mag_cpu_blas_impl_unary(e5m10, E5M10, sigmoid)
mag_cpu_blas_impl_unary(e5m10, E5M10, sigmoid_dv)
mag_cpu_blas_impl_unary(e5m10, E5M10, hard_sigmoid)
mag_cpu_blas_impl_unary(e5m10, E5M10, silu)
mag_cpu_blas_impl_unary(e5m10, E5M10, silu_dv)
mag_cpu_blas_impl_unary(e5m10, E5M10, tanh)
mag_cpu_blas_impl_unary(e5m10, E5M10, tanh_dv)
mag_cpu_blas_impl_unary(e5m10, E5M10, relu)
mag_cpu_blas_impl_unary(e5m10, E5M10, relu_dv)
mag_cpu_blas_impl_unary(e5m10, E5M10, gelu)
mag_cpu_blas_impl_unary(e5m10, E5M10, gelu_dv)
mag_cpu_blas_impl_unary_scalar(e5m10, E5M10, add)
mag_cpu_blas_impl_unary_scalar(e5m10, E5M10, sub)
mag_cpu_blas_impl_unary_scalar(e5m10, E5M10, mul)
mag_cpu_blas_impl_unary_scalar(e5m10, E5M10, div)
mag_cpu_blas_impl_unary_scalar(e5m10, E5M10, pow)
mag_cpu_blas_impl_binary(e5m10, E5M10, add)
mag_cpu_blas_impl_binary(e5m10, E5M10, sub)
mag_cpu_blas_impl_binary(e5m10, E5M10, mul)
mag_cpu_blas_impl_binary(e5m10, E5M10, div)

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, sum, mag_E8M23, 0.0f, \
    acc += mag_e5m10_cvt_e8m23(bx[off]);, \
    *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, mean, mag_E8M23, 0.0f, \
    acc += mag_e5m10_cvt_e8m23(bx[off]);, \
    acc /= (mag_E8M23)x->numel; *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, min, mag_E8M23, INFINITY, \
    acc = fminf(acc, mag_e5m10_cvt_e8m23(bx[off]));, \
    *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, max, mag_E8M23, -INFINITY, \
    acc = fmaxf(acc, mag_e5m10_cvt_e8m23(bx[off]));, \
    *br = mag_e8m23_cvt_e5m10(acc); )

static void MAG_HOTPROC mag_blas_softmax_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
        mag_Tensor* r = payload->node;
        const mag_Tensor* x = r->op_inputs[0];
        mag_E5M10* br = mag_e5m10p_mut(r);
        const mag_E5M10* bx = mag_e5m10p(x);
        int64_t last_dim = r->shape[r->rank-1];
        int64_t num_rows = r->numel / last_dim;
        int64_t tc = payload->thread_num;
        int64_t ti = payload->thread_idx;
        int64_t rows_per_thread = (num_rows + tc - 1)/tc;
        int64_t start_row = ti * rows_per_thread;
        int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
        for (int64_t row = start_row; row < end_row; ++row) {
            const mag_E5M10* row_in = bx + row * last_dim;
            mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
            mag_E5M10* row_out = br + row * last_dim;
            mag_E8M23 max_val = mag_e5m10_cvt_e8m23(row_in[0]);  /* Max val is computed for numerical stability */
            for (int64_t i=1; i < last_dim; ++i) {
                mag_E8M23 fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
                if (fp32_row > max_val) {
                    mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                    max_val = fp32_row;
                }
            }
            mag_E8M23 sum = 0.0f;
            for (int64_t i=0; i < last_dim; ++i) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
                mag_E8M23 fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
                row_out[i] = mag_e8m23_cvt_e5m10(expf(fp32_row - max_val)); /* -max for numerical stability */
                sum += fp32_row;
            }
            for (int64_t i=0; i < last_dim; ++i) {
                row_out[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(row_out[i]) / sum);
            }
        }
    }

#undef mag_cpu_blas_impl_unary_scalar
#undef mag_cpu_blas_impl_unary
#undef mag_cpu_blas_impl_binary

#define VLA(type, name, size) \
type* name = (type*)(*mag_alloc)(NULL, (size) * sizeof(type))

static int64_t mag_offset_rmn(const mag_Tensor* _Nonnull t, int64_t batch, int64_t i, int64_t j) {
    int64_t ts0 = t->strides[0];
    int64_t ts1 = t->strides[1];
    int64_t ts2 = t->strides[2];
    int64_t ra = t->rank;
    int64_t off = 0;
    if (ra == 3) {
        off += batch*ts0;
        off += i*ts1;
        off += j*ts2;
    } else if (ra == 2) {
        off += i*ts0;
        off += j*ts1;
    } else {
        off += i*ts0;
    }
    return off;
}

static MAG_AINLINE mag_E8M23* _Nonnull mag_mm_pack_x_e8m23(mag_E8M23* _Nonnull xbuf, int64_t M, int64_t N, int64_t K, int64_t xb, const mag_Tensor* _Nonnull x, const mag_E8M23* _Nonnull px) {
    for (int64_t i=0; i < M; ++i) {
        for (int64_t k=0; k < K; ++k) {
            size_t j = mag_offset_rmn(x, xb, i, k);
            mag_bnd_chk(px+j, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(xbuf + i*K + k, xbuf, M*K*sizeof(*xbuf));
            xbuf[i*K + k] = px[j];
        }
    }
    return xbuf;
}

static MAG_AINLINE mag_E8M23* _Nonnull mag_mm_pack_y_e8m23(mag_E8M23* _Nonnull ybuf, int64_t K, int64_t N, int64_t yb, const mag_Tensor* _Nonnull y, const mag_E8M23* _Nonnull py) {
    if (y->rank == 1) {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[k];
            }
        }
    } else {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[mag_offset_rmn(y, yb, k, n)];
            }
        }
    }
    return ybuf;
}

static MAG_AINLINE void mag_mm_scatter_back_r_e8m23(const mag_E8M23* _Nonnull rbuf, mag_E8M23* _Nonnull pr, const mag_Tensor* _Nonnull r, int64_t b, int64_t M, int64_t N) {
    for (int64_t i=0; i < M; ++i)
        for (int64_t n=0; n < N; ++n)
            pr[mag_offset_rmn(r, b, i, n)] = rbuf[i*N + n];
}

static void MAG_HOTPROC mag_blas_matmul_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
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
    bool x_row = mag_tensor_is_contiguous(x) && x->strides[x->rank-1] == 1;
    bool r_row = mag_tensor_is_contiguous(r) && (r->rank == 1 || r->strides[r->rank-1] == 1);
    mag_E8M23* xbuf = x_row ? NULL : (*mag_alloc)(NULL, M*K*sizeof(*xbuf)); /*MK*/
    mag_E8M23* ybuf = (*mag_alloc)(NULL, K*N*sizeof(*ybuf)); /*KN*/
    mag_E8M23* rbuf = r_row ? NULL : (*mag_alloc)(NULL, M*N*sizeof(*rbuf)); /*MN*/
    for (int64_t b=0; b < batch; ++b) {
        int64_t xb = bx_batch == 1 ? 0 : b;
        int64_t yb = by_batch == 1 ? 0 : b;
        const mag_E8M23* px = bx + mag_offset_rmn(x, xb, 0, 0);
        const mag_E8M23* py = by + mag_offset_rmn(y, yb, 0, 0);
        mag_E8M23* pr = br + mag_offset_rmn(r, b,  0, 0);
        const mag_E8M23* restrict A = px;
        if (!x_row) A = mag_mm_pack_x_e8m23(xbuf, M, N, K, xb, x, px);              /* Pack and broadcast X if needed */
        const mag_E8M23* restrict B = mag_mm_pack_y_e8m23(ybuf, K, N, yb, y, py);   /* Y is always packed to provide contiguous access for the microkernel. ybuf[n*K + k] holds element (k, n) – contiguous per column */
        mag_E8M23* restrict C = r_row ? pr : rbuf;
        /* Packed SGEMM */
        for (int64_t i=0; i < M; ++i) {
            const mag_E8M23* restrict a_row = A + i*K;
            for (int64_t n=0; n < N; ++n) {
                const mag_E8M23* restrict b_col = B + n*K; /* a_row and b_col are both contiguous now */
                C[i*N + n] = mag_vdot_e8m23(K, b_col, a_row); /* SIMD dotprod */
            }
        }
        if (!r_row) mag_mm_scatter_back_r_e8m23(rbuf, pr, r, b, M, N); /* Scatter back R if needed */
    }
    if (xbuf) (*mag_alloc)(xbuf, 0);
    if (ybuf) (*mag_alloc)(ybuf, 0);
    if (mag_alloc) (*mag_alloc)(rbuf, 0);
}

static void MAG_HOTPROC mag_blas_matmul_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
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
    bool x_row = mag_tensor_is_contiguous(x) && x->strides[x->rank-1] == 1;
    bool y_row = mag_tensor_is_contiguous(y) && (y->rank == 1 || y->strides[y->rank-1] == 1);
    bool r_row = mag_tensor_is_contiguous(r) && (r->rank == 1 || r->strides[r->rank-1] == 1);
    VLA(mag_E5M10, xbuf, M * K);
    VLA(mag_E5M10, ybuf, K * N);
    VLA(mag_E5M10, rbuf, M * N);
    for (int64_t b=0; b < batch; ++b) {
        int64_t xb = bx_batch == 1 ? 0 : b;
        int64_t yb = by_batch == 1 ? 0 : b;
        const mag_E5M10* px = bx + mag_offset_rmn(x, xb, 0, 0);
        const mag_E5M10* py = by + mag_offset_rmn(y, yb, 0, 0);
        mag_E5M10* pr = br + mag_offset_rmn(r,  b,  0, 0);
        const mag_E5M10* A = px;
        if (!x_row) { /* pack / broadcast X if needed */
            for (int64_t i=0; i < M; ++i) {
                for (int64_t k=0; k < K; ++k) {
                    size_t j = mag_offset_rmn(x, xb, i, k);
                    mag_bnd_chk(px+j, bx, mag_tensor_get_data_size(x));
                    mag_bnd_chk(xbuf + i*K + k, xbuf, M*K*sizeof(*xbuf));
                    xbuf[i*K + k] = px[j];
                }
            }
            A = xbuf;
        }
        const mag_E5M10* B = py;
        if (!y_row) { /* pack / broadcast Y if needed */
            for (int64_t k=0; k < K; ++k) {
                for (int64_t n=0; n < N; ++n) {
                    ybuf[k*N + n] = y->rank == 1 ? py[k] : py[mag_offset_rmn(y, yb, k, n)];
                }
            }
            B = ybuf;
        }
        mag_E5M10* C = r_row ? pr : rbuf;
        for (int64_t i=0; i < M; ++i) { /* Standard SGEMM */
            for (int64_t n=0; n < N; ++n) {
                mag_E8M23 acc = 0.0f;
                for (int64_t k=0; k < K; ++k) {
                    acc += mag_e5m10_cvt_e8m23(A[i*K + k])*mag_e5m10_cvt_e8m23(B[k*N + n]);
                }
                C[i*N + n] = mag_e8m23_cvt_e5m10(acc);
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

static void MAG_HOTPROC  mag_blas_repeat_back_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
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

static void MAG_HOTPROC mag_blas_repeat_back_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    for (int64_t i=0; i < r->numel; ++i)
        br[mag_offset_from_flat(r, i)].bits = 0;
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
        br[roff] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(br[roff]) + mag_e5m10_cvt_e8m23(bx[xoff]));
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

#elif defined(__aarch64__) || defined(_M_ARM64)

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

static void (*_Nonnull const mag_blas_lut_init_kernels[MAG_IOP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull) = {
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

static void (*_Nonnull const mag_blas_lut_eval_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_clone_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_clone_e5m10,
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
    [MAG_OP_SGN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sgn_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sgn_e5m10,
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
    [MAG_OP_FLOOR] = {
        [MAG_DTYPE_E8M23] = &mag_blas_floor_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_floor_e5m10,
    },
    [MAG_OP_CEIL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_ceil_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_ceil_e5m10,
    },
    [MAG_OP_ROUND] = {
        [MAG_DTYPE_E8M23] = &mag_blas_round_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_round_e5m10,
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
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_blas_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_repeat_back_e5m10,
    },
};

mag_static_assert(MAG_DTYPE__NUM <= 255);
#define mag_dt_perm(x,y) ((((x)&255)<<8)+((y)&255))
static void MAG_HOTPROC mag_blas_vector_cast_stub(size_t nb, const void* _Nonnull src, mag_DType src_t, void* _Nonnull dst, mag_DType dst_t) {
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

void MAG_BLAS_SPECIALIZATION(mag_CPUKernelRegistry* _Nonnull kernels) {
    for (int i=0; i < MAG_IOP__NUM; ++i)
        for (int j=0; j < MAG_DTYPE__NUM; ++j)
            kernels->init[i][j] = mag_blas_lut_init_kernels[i][j];
    for (int i=0; i < MAG_OP__NUM; ++i) {
        for (int j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->fwd[i][j] = mag_blas_lut_eval_kernels[i][j];
        }
    }
    kernels->vector_cast = &mag_blas_vector_cast_stub;
}

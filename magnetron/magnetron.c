/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

/*
**
**
** ### To add a new operation:
** 1. Add the operation to the mag_op_def macro, which defines all operations, with all information needed.
** 2. Write a validation routine, or use an existing one (e.g. 'mag_validate_op_binary').
** 3. Add the validation routine to the 'routines' table in 'mag_op_get_validator_routine', at the op index.
** 4. Write a result tensor constructor routine or use an existing one (e.g. 'mag_result_constructor_routine_isomorph').
** 5. Add the result tensor constructor routine to the 'routines' table in 'mag_op_get_result_constructor_routine', at the op index.
** 6. Write a BLAS computation routine.
** 7. Add the BLAS computation routine to the 'dispatch_lut' table in 'mag_blas_compute_dispatch_table_default', at the op index.
*/

/*
** Here ⊕ denotes a binary or unary operator.
** Note that an operators can or cannot support any of those forms. This must be specified in mag_op_def.
**
** Operators can have two forms:
**
** 1. R = A ⊕ B
**  Result is a new tensor of shape of A and contains element-wise result of A ⊕ B, where A and B are tensors.
**
** 2. R = A ⊕= B
**  Result is a view tensor of shape of A and contains element-wise result of A ⊕= B, where A and B are tensors. (Safes 1 allocation)
*/

#include "magnetron.h"
#include "magnetron_internal.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <time.h>
#include <float.h>
#include <ctype.h>
#include <errno.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#ifdef NDEBUG
#define MAG_LOG_DEFAULT_ENABLE 0
#else
#define MAG_LOG_DEFAULT_ENABLE 1
#endif
bool mag_log_enabled = MAG_LOG_DEFAULT_ENABLE; /* Read from multiple threads, allowed to be written from main thread once at start. */
#undef MAG_LOG_DEFAULT_ENABLE

void mag_set_log_mode(bool enabled) {
    mag_log_enabled = enabled;
}

MAG_NORET MAG_COLDPROC MAG_EXPORT void mag_panic(const char* msg, ...) {
    fprintf(stdout, "%s", MAG_CC_RED);
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fprintf(stdout, "%s", MAG_CC_RESET);
    fputc('\n', stdout);
    fflush(stdout);
    abort();
}

static void* mag_os_alloc_stub(void* blk, size_t size) {
    if (!size) {
        free(blk);
        return NULL;
    }
    if(!blk) {
        blk = malloc(size);
        mag_assert(blk, "Failed to allocate %.03fKiB memory", (double)size/(double)(1<<10));
        return blk;
    }
    void* block = realloc(blk, size);
    mag_assert(blk, "Failed to reallocate %.03fKiB memory", (double)size/(double)(1<<10));
    return block;
}

void* (*mag_alloc)(void* blk, size_t size) = &mag_os_alloc_stub;

void* mag_alloc_aligned(size_t size, size_t align) {
    mag_assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    void* p = (*mag_alloc)(NULL, size+sizeof(void*)+align-1);
    uintptr_t pp = ((uintptr_t)p+sizeof(void*)+align-1)&-align;
    ((void**)pp)[-1] = p;
    return (void*)pp;
}

void mag_free_aligned(void* blk) {
    (*mag_alloc)(((void**)blk)[-1], 0);
}

/* Include STB libraries and override their allocator with ours. */
#define STBI_STATIC
#define STBI_MALLOC(sz) ((*mag_alloc)(NULL, (sz)))
#define STBI_FREE(ptr) ((*mag_alloc)((ptr), 0))
#define STBI_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz)))
#define STBIW_MALLOC(sz) ((*mag_alloc)(NULL, (sz)))
#define STBIW_FREE(ptr) ((*mag_alloc)((ptr), 0))
#define STBIW_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz)))
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#if defined(__x86_64__) || defined(_M_X64)
#define MAG_X86_64_CPUID_0H 0
#define MAG_X86_64_CPUID_1H 1
#define MAG_X86_64_CPUID_2H 2
#define MAG_X86_64_CPUID_7H 3
#define MAG_X86_64_CPUID_80000001H 4
#define MAG_X86_64_CPUID_80000007H 5
#define MAG_X86_64_CPUID_16H 6
#define MAG_X86_64_CPUID_7H_1H 7
#define MAG_X86_64_CPUID_EAX 0
#define MAG_X86_64_CPUID_EBX 1
#define MAG_X86_64_CPUID_ECX 2
#define MAG_X86_64_CPUID_EDX 3

#define mag_x86_64_feature_def(_, __) /* Enumerator | CPUDID Leaf | Register | Bit Index */\
    _(AVX                  ,    1H,        ECX,     28)__\
    _(AVX2                 ,    7H,        EBX,      5)__\
    _(AVXVNNI              ,    7H_1H,     EAX,      4)__\
    _(AVXVNNIINT8          ,    7H_1H,     EDX,      4)__\
    _(AVXVNNIINT16         ,    7H_1H,     EDX,     10)__\
    _(AVX512BW             ,    7H,        EBX,     30)__\
    _(AVX512CD             ,    7H,        EBX,     28)__\
    _(AVX512DQ             ,    7H,        EBX,     17)__\
    _(AVX512ER             ,    7H,        EBX,     27)__\
    _(AVX512F              ,    7H,        EBX,     16)__\
    _(AVX512IFMA           ,    7H,        EBX,     21)__\
    _(AVX512PF             ,    7H,        EBX,     26)__\
    _(AVX512VBMI           ,    7H,        ECX,      1)__\
    _(AVX512VL             ,    7H,        EBX,     31)__\
    _(AVX512_4FMAPS        ,    7H,        EDX,      3)__\
    _(AVX512_4VNNIW        ,    7H,        EDX,      2)__\
    _(AVX512_FP16          ,    7H,        EDX,     23)__\
    _(AVX512_BF16          ,    7H_1H,     EAX,      5)__\
    _(AVX512_BITALG        ,    7H,        ECX,     12)__\
    _(AVX512_VBMI2         ,    7H,        ECX,      6)__\
    _(AVX512_VNNI          ,    7H,        ECX,     11)__\
    _(AVX512_VP2INTERSECT  ,    7H,        EDX,      8)__\
    _(AVX512_VPOPCNTDQ     ,    7H,        ECX,     14)__\
    _(BMI                  ,    7H,        EBX,      3)__\
    _(BMI2                 ,    7H,        EBX,      8)__\
    _(F16C                 ,    1H,        ECX,     29)__\
    _(FMA                  ,    1H,        ECX,     12)__\
    _(FPU                  ,    1H,        EDX,      0)__\
    _(GFNI                 ,    7H,        ECX,      8)__\
    _(IA64                 ,    1H,        EDX,     30)__\
    _(MMX                  ,    1H,        EDX,     23)__\
    _(OSXSAVE              ,    1H,        ECX,     27)__\
    _(PCLMUL               ,    1H,        ECX,      1)__\
    _(RDRND                ,    1H,        ECX,     30)__\
    _(RDSEED               ,    7H,        EBX,     18)__\
    _(RDTSCP               ,    80000001H, EDX,     27)__\
    _(SHA                  ,    7H,        EBX,     29)__\
    _(SSE                  ,    1H,        EDX,     25)__\
    _(SSE2                 ,    1H,        EDX,     26)__\
    _(SSE3                 ,    1H,        ECX,      0)__\
    _(SSE4_1               ,    1H,        ECX,     19)__\
    _(SSE4_2               ,    1H,        ECX,     20)__\
    _(SSSE3                ,    1H,        ECX,      9)__\
    _(VAES                 ,    7H,        ECX,      9)__\
    _(VME                  ,    1H,        EDX,      1)__\
    _(VMX                  ,    1H,        ECX,      5)__\
    _(VPCLMULQDQ           ,    7H,        ECX,     10)__\
    _(XSAVE                ,    1H,        ECX,     26)__\
    _(HYBRID_CPU           ,    7H,        EDX,     15)__

#define _(enumerator, leaf, reg, bit) MAG_X86_64_FEATURE_##enumerator
typedef enum mag_x86_64_feature_t {
    mag_x86_64_feature_def(_, MAG_SEP)
    MAG_X86_64_FEATURE__COUNT
} mag_x86_64_feature_t;
#undef _
#define _(enumerator, leaf, reg, bit) #enumerator
static const char* const mag_x86_64_feature_names[MAG_X86_64_FEATURE__COUNT] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&MAG_X86_64_CPUID_##leaf)
static const uint8_t mag_x86_64_feature_leaves[MAG_X86_64_FEATURE__COUNT] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&MAG_X86_64_CPUID_##reg)
static const uint8_t mag_x86_64_feature_regs[MAG_X86_64_FEATURE__COUNT] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (1u<<(bit))
static const uint32_t msml__x86_64_feature_masks[MAG_X86_64_FEATURE__COUNT] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#undef mag_x86_64_feature_def
#endif

void* (*mag_get_alloc_fn(void))(void* blk, size_t size) {
    return mag_alloc;
}

void mag_set_alloc_fn(void* (*alloc)(void* blk, size_t size)) {
    mag_assert2(alloc);
    mag_alloc = alloc;
}

void mag_humanize_memory_size(size_t n, double* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (double)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (double)n/(double)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (double)n/(double)(1<<20);
        *unit = "MiB";
    } else {
        *out = (double)n/(double)(1<<30);
        *unit = "GiB";
    }
}

static void MAG_COLDPROC mag_print_separator(FILE* f) {
    f = f ? f : stdout;
    char sep[100+1];
    for (size_t i=0; i < (sizeof(sep)/sizeof(*sep))-1; ++i) sep[i] = '-';
    sep[sizeof(sep)/sizeof(*sep)-1] = '\0';
    fprintf(f, "%s\n", sep);
}

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)
static void mag_fmt_dims(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    memset(*buf, 0, sizeof(*buf));
    char* p = *buf;
    mag_assert2(p+rank*21+3 < *buf+MAG_FMT_DIM_BUF_SIZE);
    *p++ = '(';
    for (int64_t i=0; i < rank; ++i) {
        p += snprintf(p, 21, "%" PRIi64, (*dims)[i]);
        if (i < rank-1) {
            *p++ = ',';
            *p++ = ' ';
        }
    }
    *p++ = ')';
    *p = '\0';
}

#ifdef _WIN32
#include <wchar.h>
extern __declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int cp,
    unsigned long flags,
    const char* str,
    int cbmb,
    wchar_t* widestr,
    int cchwide
);
extern __declspec(dllimport) int __stdcall WideCharToMultiByte(
    unsigned int cp,
    unsigned long flags,
    const wchar_t* widestr,
    int cchwide,
    char* str,
    int cbmb,
    const char* defchar,
    int* used_default
);
#endif

static FILE* mag_fopen(const char* file, const char* mode) {
    mag_assert(file && *file && mode && *mode, "Invalid file name or mode");
    FILE* f = NULL;
    #ifdef _WIN32
        wchar_t w_mode[64];
        wchar_t w_file[1024];
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, file, -1, w_file, sizeof(w_file)/sizeof(*w_file)) == 0) return NULL;
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, w_mode, sizeof(w_mode)/sizeof(*w_mode)) == 0) return NULL;
        #if defined(_MSC_VER) && _MSC_VER >= 1400
           if (_wfopen_s(&f, w_file, w_mode) != 0)
               return NULL;
        #else
           f = _wfopen(w_file, w_mode);
        #endif
    #elif defined(_MSC_VER) && _MSC_VER >= 1400
        if (fopen_s(&f, filename, mode) != 0) return NULL;
    #else
        f = fopen(file, mode);
    #endif
    return f;
}

static inline uintptr_t mag_thread_id(void) {
    uintptr_t tid;
    #if defined(_MSC_VER) && defined(_M_X64)
        tid = __readgsqword(48);
    #elif defined(_MSC_VER) && defined(_M_IX86)
        tid = __readfsdword(24);
    #elif defined(_MSC_VER) && defined(_M_ARM64)
        tid = __getReg(18);
    #elif defined(__i386__)
        __asm__ __volatile__("movl %%gs:0, %0" : "=r" (tid));  /* x86-32 WIN32 uses %GS */
    #elif defined(__MACH__) && defined(__x86_64__)
        __asm__ __volatile__("movq %%gs:0, %0" : "=r" (tid));  /* x86.64 OSX uses %GS */
    #elif defined(__x86_64__)
        __asm__ __volatile__("movq %%fs:0, %0" : "=r" (tid));  /* x86-64 Linux and BSD uses %FS */
    #elif defined(__arm__)
        __asm__ __volatile__("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r" (tid));
    #elif defined(__aarch64__) && defined(__APPLE__)
        __asm__ __volatile__("mrs %0, tpidrro_el0" : "=r" (tid));
    #elif defined(__aarch64__)
        __asm__ __volatile__("mrs %0, tpidr_el0" : "=r" (tid));
    #elif defined(__powerpc64__)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        register uintptr_t tp __asm__ ("r13");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #endif
    #elif defined(__powerpc__)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        register uintptr_t tp __asm__ ("r2");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #endif
    #elif defined(__s390__) && defined(__GNUC__)
        tid = (uintptr_t)__builtin_thread_pointer();
    #elif defined(__riscv)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        __asm__ ("mv %0, tp" : "=r" (tid));
    #endif
    #else
    #error "Unsupported magnetron platform"
    #endif
    return tid;
}

static uint64_t mag_hpc_clock_ns(void) { /* High precision clock in nanoseconds. */
    #ifdef _WIN32
        static LONGLONG t_freq;
        static LONGLONG t_boot;
        static bool t_init = false;
        if (!t_init) { /* Reduce chance of integer overflow when uptime is high. */
            LARGE_INTEGER li;
            QueryPerformanceFrequency(&li);
            t_freq = li.QuadPart;
            QueryPerformanceCounter(&li);
            t_boot = li.QuadPart;
            t_init = true;
        }
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return ((li.QuadPart - t_boot)*1000000000) / t_freq;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec*1000000000 + (uint64_t)ts.tv_nsec;
    #endif
}
static uint64_t mag_hpc_clock_elapsed_ns(uint64_t start) { /* High precision clock elapsed time in microseconds. */
    return (uint64_t)llabs((int64_t)mag_hpc_clock_ns() - (int64_t)start);
}
static double mag_hpc_clock_elapsed_ms(uint64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (double)mag_hpc_clock_elapsed_ns(start) / 1e6;
}
#define mag_clock_cycles() ((uint64_t)clock())
#define mag_cycles_per_ms() ((uint64_t)CLOCKS_PER_SEC/1000)

typedef uint32_t mag_bitset_t;
mag_static_assert(sizeof(mag_bitset_t) == 4);
#define mag_bitset_size(n) (((n)+((4<<3)-1))>>5)
#define mag_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define mag_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define mag_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define mag_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32) && defined(__ARM_FEATURE_CRYPTO)
static uint64x2_t MAG_AINLINE mag_clmul_lo_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    register uint64x2_t r;
    __asm__ __volatile__(
        "pmull %0.1q, %2.1d, %3.1d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c) : "w"(a), "w"(b)
    );
    return r;
}
static uint64x2_t MAG_AINLINE mag_clmul_hi_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    register uint64x2_t r;
    __asm__ __volatile__(
        "pmull2 %0.1q, %2.2d, %3.2d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c) : "w"(a), "w"(b)
    );
    return r;
}
#elif defined(__x86_64__) || defined(_M_X64)
static uint32_t mag_xnmodp(uint64_t n) { /* x^n mod P, in log(n) time */
    uint64_t stack = ~(uint64_t)1;
    uint32_t low;
    for (; n > 191; n = (n>>1) - 16) stack = (stack<<1) + (n & 1);
    stack = ~stack;
    uint32_t acc = 0x80000000 >> (n & 31);
    for (n >>= 5; n; --n) acc = _mm_crc32_u32(acc, 0);
    while (low = stack & 1, stack >>= 1) {
        __m128i x = _mm_cvtsi32_si128(acc);
        uint64_t y = _mm_cvtsi128_si64(_mm_clmulepi64_si128(x, x, 0));
        acc = _mm_crc32_u64(0, y << low);
    }
    return acc;
}
static __m128i MAG_AINLINE mag_clmul_scalar(uint32_t a, uint32_t b) {
    return _mm_clmulepi64_si128(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b), 0);
}
static __m128i MAG_AINLINE mag_crc_shift(uint32_t crc, size_t sz) {
    return mag_clmul_scalar(crc, mag_xnmodp((sz<<3) - 33));
}
#endif

static uint32_t mag_crc32c(const void* buffer, size_t size) { /* Compute CRC32 checksum with CRC32c polynomial. */
    if (mag_unlikely(!buffer || !size)) return 0;
    const uint8_t* buf = (const uint8_t*)buffer;
    #if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32) && defined(__ARM_FEATURE_CRYPTO)
        uint32_t crc = ~0;
        for (; size && ((uintptr_t)buf & 7); --size) crc = __crc32cb(crc, *buf++);
        if (((uintptr_t)buf & 8) && size >= 8) {
            crc = __crc32cd(crc, *(const uint64_t*)buf);
            buf += 8;
            size -= 8;
        }
        if (size >= 192) { /* First vector chunk. */
            uint64x2_t x0 = vld1q_u64((const uint64_t*)buf), y0;
            uint64x2_t x1 = vld1q_u64((const uint64_t*)(buf+16)), y1;
            uint64x2_t x2 = vld1q_u64((const uint64_t*)(buf+32)), y2;
            uint64x2_t x3 = vld1q_u64((const uint64_t*)(buf+48)), y3;
            uint64x2_t x4 = vld1q_u64((const uint64_t*)(buf+64)), y4;
            uint64x2_t x5 = vld1q_u64((const uint64_t*)(buf+80)), y5;
            uint64x2_t x6 = vld1q_u64((const uint64_t*)(buf+96)), y6;
            uint64x2_t x7 = vld1q_u64((const uint64_t*)(buf+112)), y7;
            uint64x2_t x8 = vld1q_u64((const uint64_t*)(buf+128)), y8;
            uint64x2_t x9 = vld1q_u64((const uint64_t*)(buf+144)), y9;
            uint64x2_t x10 = vld1q_u64((const uint64_t*)(buf+160)), y10;
            uint64x2_t x11 = vld1q_u64((const uint64_t*)(buf+176)), y11;
            uint64x2_t k;
            { static const uint64_t MAG_ALIGN(16) k_[] = {0xa87ab8a8, 0xab7aff2a}; k = vld1q_u64(k_); }
            x0 = veorq_u64((uint64x2_t){crc, 0}, x0);
            buf += 192;
            size -= 192;
            while (size >= 192) { /* Work loop. */
                y0 = mag_clmul_lo_e(x0, k, vld1q_u64((const uint64_t*)buf)), x0 = mag_clmul_hi_e(x0, k, y0);
                y1 = mag_clmul_lo_e(x1, k, vld1q_u64((const uint64_t*)(buf+16))), x1 = mag_clmul_hi_e(x1, k, y1);
                y2 = mag_clmul_lo_e(x2, k, vld1q_u64((const uint64_t*)(buf+32))), x2 = mag_clmul_hi_e(x2, k, y2);
                y3 = mag_clmul_lo_e(x3, k, vld1q_u64((const uint64_t*)(buf+48))), x3 = mag_clmul_hi_e(x3, k, y3);
                y4 = mag_clmul_lo_e(x4, k, vld1q_u64((const uint64_t*)(buf+64))), x4 = mag_clmul_hi_e(x4, k, y4);
                y5 = mag_clmul_lo_e(x5, k, vld1q_u64((const uint64_t*)(buf+80))), x5 = mag_clmul_hi_e(x5, k, y5);
                y6 = mag_clmul_lo_e(x6, k, vld1q_u64((const uint64_t*)(buf+96))), x6 = mag_clmul_hi_e(x6, k, y6);
                y7 = mag_clmul_lo_e(x7, k, vld1q_u64((const uint64_t*)(buf+112))), x7 = mag_clmul_hi_e(x7, k, y7);
                y8 = mag_clmul_lo_e(x8, k, vld1q_u64((const uint64_t*)(buf+128))), x8 = mag_clmul_hi_e(x8, k, y8);
                y9 = mag_clmul_lo_e(x9, k, vld1q_u64((const uint64_t*)(buf+144))), x9 = mag_clmul_hi_e(x9, k, y9);
                y10 = mag_clmul_lo_e(x10, k, vld1q_u64((const uint64_t*)(buf+160))), x10 = mag_clmul_hi_e(x10, k, y10);
                y11 = mag_clmul_lo_e(x11, k, vld1q_u64((const uint64_t*)(buf+176))), x11 = mag_clmul_hi_e(x11, k, y11);
                buf += 192;
                size -= 192;
            }
            /* Reduce x0 ... x11 to just x0. */
            { static const uint64_t MAG_ALIGN(16) k_[] = {0xf20c0dfe, 0x493c7d27}; k = vld1q_u64(k_); }
            y0 = mag_clmul_lo_e(x0, k, x1), x0 = mag_clmul_hi_e(x0, k, y0);
            y2 = mag_clmul_lo_e(x2, k, x3), x2 = mag_clmul_hi_e(x2, k, y2);
            y4 = mag_clmul_lo_e(x4, k, x5), x4 = mag_clmul_hi_e(x4, k, y4);
            y6 = mag_clmul_lo_e(x6, k, x7), x6 = mag_clmul_hi_e(x6, k, y6);
            y8 = mag_clmul_lo_e(x8, k, x9), x8 = mag_clmul_hi_e(x8, k, y8);
            y10 = mag_clmul_lo_e(x10, k, x11), x10 = mag_clmul_hi_e(x10, k, y10);
            { static const uint64_t MAG_ALIGN(16) k_[] = {0x3da6d0cb, 0xba4fc28e}; k = vld1q_u64(k_); }
            y0 = mag_clmul_lo_e(x0, k, x2), x0 = mag_clmul_hi_e(x0, k, y0);
            y4 = mag_clmul_lo_e(x4, k, x6), x4 = mag_clmul_hi_e(x4, k, y4);
            y8 = mag_clmul_lo_e(x8, k, x10), x8 = mag_clmul_hi_e(x8, k, y8);
            { static const uint64_t MAG_ALIGN(16) k_[] = {0x740eef02, 0x9e4addf8}; k = vld1q_u64(k_); }
            y0 = mag_clmul_lo_e(x0, k, x4), x0 = mag_clmul_hi_e(x0, k, y0);
            x4 = x8;
            y0 = mag_clmul_lo_e(x0, k, x4), x0 = mag_clmul_hi_e(x0, k, y0);
            /* Reduce 128 bits to 32 bits, and multiply by x^32. */
            crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
            crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
        }
        for (; size >= 8; buf += 8, size -= 8) crc = __crc32cd(crc, *(const uint64_t*)buf);
        for (; size; --size) crc = __crc32cb(crc, *buf++);
        return ~crc;
    #elif defined(__x86_64__) || defined(_M_X64)
        uint32_t crc = ~0;
        for (; size && ((uintptr_t)buf & 7); --size) crc = _mm_crc32_u8(crc, *buf++);
        if (size >= 32) {
            size_t klen = ((size - 8) / 24)<<3;
            uint32_t crc1 = 0;
            uint32_t crc2 = 0;
            /* Main loop. */
            do {
                crc = _mm_crc32_u64(crc, *(const uint64_t*)buf);
                crc1 = _mm_crc32_u64(crc1, *(const uint64_t*)(buf + klen));
                crc2 = _mm_crc32_u64(crc2, *(const uint64_t*)(buf + (klen<<1)));
                buf += 8;
                size -= 24;
            } while (size >= 32);
            __m128i vc0 = mag_crc_shift(crc, (klen << 1) + 8);
            __m128i vc1 = mag_crc_shift(crc1, klen + 8);
            uint64_t vc = _mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);
            /* Final 8 bytes. */
            buf += klen<<1;
            crc = crc2;
            crc = _mm_crc32_u64(crc, *(const uint64_t*)buf ^ vc), buf += 8;
            size -= 8;
        }
        for (; size >= 8; buf += 8, size -= 8) crc = _mm_crc32_u64(crc, *(const uint64_t*)buf);
        for (; size; --size) crc = _mm_crc32_u8(crc, *buf++);
        return ~crc;
    #else
        static const uint32_t crc_lut[256] = {
            0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
            0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
            0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
            0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
            0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
            0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
            0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
            0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
            0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
            0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
            0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
            0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
            0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
            0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
            0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
            0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
            0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
            0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
            0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
            0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
            0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
            0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
            0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
            0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
            0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
            0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
            0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
            0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
            0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d, 0xef087a76, 0x1d63f975,
            0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
            0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
            0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
            0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
            0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
            0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
            0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
            0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
            0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
            0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
            0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
            0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
            0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
            0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
        };
        uint32_t crc = ~0u;
        for (size_t i=0; i < size; ++i)
            crc = (crc >> 8) ^ crc_lut[buf[i] ^ (crc & 0xff)];
        return ~crc;
    #endif
}

typedef enum mag_format_type {
    MAG_FMT_EOF, MAG_FMT_ERR, MAG_FMT_LIT, MAG_FMT_INT,
    MAG_FMT_UINT, MAG_FMT_NUM, MAG_FMT_STR, MAG_FMT_CHAR,
    MAG_FMT_PTR
} mag_format_type; /* Format types for formatted output */

typedef uint32_t mag_format_flags; /* Flags for formatting output */

/* Format flags */
#define MAG_FMT_F_LEFT  0x0100 /* Left-align the output */
#define MAG_FMT_F_PLUS  0x0200 /* Prefix positive numbers with a plus sign */
#define MAG_FMT_F_ZERO  0x0400 /* Pad with zeros instead of spaces */
#define MAG_FMT_F_SPACE 0x0800 /* Prefix a space for positive numbers */
#define MAG_FMT_F_ALT   0x1000 /* Alternate format flag */
#define MAG_FMT_F_UPPER 0x2000 /* Use uppercase letters for hex output */

/* Format subtypes (bits reused) */
#define MAG_FMT_T_HEX   0x0010 /* Hexadecimal format for unsigned integers */
#define MAG_FMT_T_OCT   0x0020 /* Octal format for unsigned integers */
#define MAG_FMT_T_FP_A  0x0000 /* 'a' format for floating-point numbers */
#define MAG_FMT_T_FP_E  0x0010 /* 'e' format for floating-point numbers */
#define MAG_FMT_T_FP_F  0x0020 /* 'f' format for floating-point numbers */
#define MAG_FMT_T_FP_G  0x0030 /* 'g' format for floating-point numbers */
#define MAG_FMT_T_QUOTED 0x0010 /* Quoted string format */

#define MAG_FMT_SH_WIDTH 16    /* Shift width for formatting */
#define MAG_FMT_SH_PREC  24    /* Shift precision for formatting */
#define MAG_FMT_TYPE(sf) ((mag_format_type)((sf) & 15))  /* Extract format type */
#define MAG_FMT_WIDTH(sf) (((sf) >> MAG_FMT_SH_WIDTH) & 255u) /* Extract width */
#define MAG_FMT_PREC(sf) ((((sf) >> MAG_FMT_SH_PREC) & 255u) - 1u) /* Extract precision */
#define MAG_FMT_FP(sf) (((sf) >> 4) & 3) /* Extract floating-point format */

/* Formats for conversion characters */
#define MAG_FMT_A (MAG_FMT_NUM|MAG_FMT_T_FP_A) /* 'a' format */
#define MAG_FMT_C (MAG_FMT_CHAR) /* 'c' format */
#define MAG_FMT_D (MAG_FMT_INT)  /* 'd' format */
#define MAG_FMT_E (MAG_FMT_NUM|MAG_FMT_T_FP_E) /* 'e' format */
#define MAG_FMT_F (MAG_FMT_NUM|MAG_FMT_T_FP_F) /* 'f' format */
#define MAG_FMT_G (MAG_FMT_NUM|MAG_FMT_T_FP_G) /* 'g' format */
#define MAG_FMT_I MAG_FMT_D /* 'i' format (same as 'd') */
#define MAG_FMT_O (MAG_FMT_UINT|MAG_FMT_T_OCT) /* 'o' format */
#define MAG_FMT_P (MAG_FMT_PTR) /* 'p' format */
#define MAG_FMT_Q (MAG_FMT_STR|MAG_FMT_T_QUOTED) /* Quoted string */
#define MAG_FMT_S (MAG_FMT_STR) /* 's' format */
#define MAG_FMT_U (MAG_FMT_UINT) /* 'u' format */
#define MAG_FMT_X (MAG_FMT_UINT|MAG_FMT_T_HEX) /* 'x' format */
#define MAG_FMT_G14 (MAG_FMT_G | ((14+1) << MAG_FMT_SH_PREC)) /* 'g' format with precision 14 */

static char* mag_fmt_f64(mag_format_flags sf, double n, char* p);

static bool MAG_AINLINE mag_imull64_ov(int64_t a, int64_t b, int64_t* out) { /* Performs c = a*b with overflow checking. Returns true on overflow, else false. */
    #ifdef _MSC_VER
    #ifdef _M_ARM64
        uint64_t high = __umulh(a, b);
        *out = a*b;
        return high != (*out>>63);
    #else
        int64_t high;
        int64_t low = _mul128(a, b, &high);
        int64_t sign = low >> 63;
        *out = low;
        return high != sign;
    #endif
    #else
    #if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
        return __builtin_smulll_overflow(a, b, (long long*)out);
    #else
        return __builtin_smull_overflow(a, b, out);
    #endif
    #endif
}

/* Generate n uniform random floats within [min, max]. */
static void mag_prng_generate_n(mag_ctx_t* ctx, float* out_gen, int64_t out_n, float min, float max) {
    float rescale_uniform = max - min;
    switch (ctx->prng_algorithm) {
        case MAG_PRNG_MERSENNE_TWISTER: {
            uint32_t* rem = &ctx->prng.mersenne.remaining;
            uint32_t* next = &ctx->prng.mersenne.next;
            uint32_t* state = ctx->prng.mersenne.state;
            for (int64_t ii=0; ii < out_n; ++ii) {
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
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        case MAG_PRNG_PCG: {
            uint64_t* state = &ctx->prng.pcg.state;
            uint64_t* inc = &ctx->prng.pcg.inc;
            for (int64_t ii=0; ii < out_n; ++ii) {
                uint64_t prev = *state;
                *state = prev*6364136223846793005ull + *inc;
                uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
                uint32_t rot = prev >> 59u;
                uint32_t y = (mixed>>rot) | (mixed << ((-rot)&31));
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        default:
            mag_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void mag_prng_init(mag_ctx_t* ctx, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull ^ (uintptr_t)ctx ^ (uintptr_t)&ctx; /* Default seed. */
    switch (ctx->prng_algorithm) {
        case MAG_PRNG_MERSENNE_TWISTER: {
            uint32_t* state = ctx->prng.mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            ctx->prng.mersenne.next = 0;
            ctx->prng.mersenne.remaining = 1;
        } break;
        case MAG_PRNG_PCG: {
            ctx->prng.pcg.state = seed ^ 0x853c49e6748fea9bull;
            ctx->prng.pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            mag_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
static bool mag_ctx_x86_64_cpu_has_feature(const mag_ctx_t* ctx, mag_x86_64_feature_t feature) {
    const uint8_t (*leafs)[49] = &mag_x86_64_feature_leaves;
    const uint8_t (*regs)[49] = &mag_x86_64_feature_regs;
    const uint32_t (*features)[8][4] = &ctx->sys.x86_64_cpu_features;
    const uint32_t (*masks)[49] = &msml__x86_64_feature_masks;
    return (*features)[(*leafs)[feature]][(*regs)[feature]] & (*masks)[feature];
}
#endif

static void mag_system_host_info_query(mag_ctx_t* ctx); /* Query host system information. */
static void mag_system_host_info_dump(mag_ctx_t* ctx) {
    mag_log_info("OS/Kernel: %s", ctx->sys.os_name);
    const char* cpu_arch = "?";
    #if defined(__x86_64__) || defined(_M_X64)
        cpu_arch = "x86-64";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        cpu_arch = "aarch64";
    #else
    #error "Unknwon CPU arch"
    #endif
    mag_log_info("CPU (%s): %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u", cpu_arch, ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
    #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        if (mag_log_enabled) {
            printf("CPU Features:");
            for (uint32_t i=0, k=0; i < MAG_X86_64_FEATURE__COUNT; ++i) {
                if (mag_ctx_x86_64_cpu_has_feature(ctx, i)) {
                    if ((k++ & 7) == 0) printf("\n\t");
                    printf("%s ", mag_x86_64_feature_names[i]);
                }
            }
            putchar('\n');
        }
    #endif
    double mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
    double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

/* Default image loader/saver implementation. */
static uint8_t* mag_default_image_load_impl(const char*, uint32_t(*)[3], mag_color_channels_t);
static void mag_default_image_load_free_fn_impl(uint8_t*);
static bool mag_default_image_save_impl(const char*, const uint8_t*, const uint32_t(*)[3]);

static MAG_COLDPROC void mag_ctx_dump_compiler_info(void) {
    const char* compiler_name = "Unknown";
    int compiler_version_major = 0, compiler_version_minor = 0;
    #ifdef __clang__
        compiler_name = "Clang";
        compiler_version_major = __clang_major__;
        compiler_version_minor = __clang_minor__;
    #elif defined(__GNUC__)
        compiler_name = "GCC";
        compiler_version_major = __GNUC__;
        compiler_version_minor = __GNUC_MINOR__;
    #elif defined(_MSC_VER)
        compiler_name = "MSVC";
        compiler_version_major = _MSC_VER / 100;
        compiler_version_minor = _MSC_VER % 100;
    #endif
    mag_log_info("magnetron v.%d.%d - " __DATE__ " " __TIME__ " - %s %d.%d", mag_version_major(MAG_VERSION), mag_version_minor(MAG_VERSION), compiler_name, compiler_version_major, compiler_version_minor);
}

mag_ctx_t* mag_ctx_create(mag_compute_device_type_t device) {
    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_compiler_info(); /* Dump compiler info. */

    /* Initialize context with default values or from context info. */
    mag_ctx_t* ctx = (mag_ctx_t*)(*mag_alloc)(NULL, sizeof(*ctx)); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init allocators */
    mag_fixed_intrusive_pool_init(&ctx->tensor_pool, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 4096);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */

    /* Query and print host system information. */
    mag_system_host_info_query(ctx);
    mag_system_host_info_dump(ctx);

    /* Configure configureable media processors */
    ctx->image_load_fn = &mag_default_image_load_impl;
    ctx->image_load_free_fn = &mag_default_image_load_free_fn_impl;
    ctx->image_save_fn = &mag_default_image_save_impl;

    /* Initialize PRNG state. */
    ctx->prng_algorithm = MAG_PRNG_MERSENNE_TWISTER;
    mag_prng_init(ctx, ctx->tr_id^(uintptr_t)ctx^(uintptr_t)&ctx); /* Initialize PRNG state. */

    /* Create selected compute device. */
    ctx->exec_mode = MAG_EXEC_MODE_EAGER;
    ctx->device_type = device;
    ctx->device = mag_init_dynamic_device(ctx, &ctx->device_type);
    mag_log_info("Compute device: %s", ctx->device->name);


    /* Print context initialization time. */
    mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

static void mag_tensor_destroy(mag_tensor_t* t);

void mag_ctx_destroy(mag_ctx_t* ctx) {
#if MAG_SANITIZE_RC /* Check for leaked tensors in RC tracking list and print them */
    mag_tensor_node_t** head = &ctx->rc_tracked;
    mag_tensor_node_t* curr = *head;
    uint32_t nleaked = 0;
    for (; curr; curr = curr->next, ++nleaked) {
        mag_log_error("Leaked tensor detected: %p, RCS: %u, RCW: %u, CTOR: %s", curr->tensor, curr->tensor->rcb.rc_strong, curr->tensor->rcb.rc_weak, curr->tensor->rcb.dtor ? "Y" : "N");
        mag_tensor_print(curr->tensor, true, false);
    }
    if (nleaked) mag_log_error("Leaked tensors detected: %u", nleaked);
    else mag_log_info("No leaked tensors detected.");
    curr = *head;
    while (curr) { /* Free tracking list */
        mag_tensor_node_t* tmp = curr;
        curr = curr->next;
        mag_tensor_destroy(tmp->tensor);
        (*mag_alloc)(tmp, 0);
    }
    *head = NULL;
#endif
    mag_fixed_intrusive_pool_print_info(&ctx->tensor_pool, "Tensor Hull Pool");
    mag_fixed_intrusive_pool_destroy(&ctx->tensor_pool);
    mag_destroy_dynamic_device(ctx->device); ctx->device = NULL;
    memset(ctx, 0, sizeof(*ctx));
    (*mag_alloc)(ctx, 0);
    ctx = NULL;
    mag_log_info("magnetron context destroyed.");
}

mag_exec_mode_t mag_ctx_get_exec_mode(const mag_ctx_t* ctx) { return ctx->exec_mode; }

void mag_ctx_set_exec_mode(mag_ctx_t* ctx, mag_exec_mode_t mode) {
    ctx->exec_mode = mode;
    mag_log_info("Execution mode set to: %s", mode == MAG_EXEC_MODE_EAGER ? "Eager" : "Deferred");
}

mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* ctx) { return ctx->prng_algorithm; }

void mag_ctx_set_prng_algorithm(mag_ctx_t* ctx, mag_prng_algorithm_t algorithm, uint64_t seed) {
    ctx->prng_algorithm = algorithm;
    mag_prng_init(ctx, seed); /* Reinitialize PRNG state with new seed. */
}

mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* ctx) { return ctx->device_type; }
const char* mag_ctx_get_compute_device_name(const mag_ctx_t* ctx) { return ctx->device->name; }
const char* mag_ctx_get_os_name(const mag_ctx_t* ctx) { return ctx->sys.os_name; }
const char* mag_ctx_get_cpu_name(const mag_ctx_t* ctx) { return ctx->sys.cpu_name; }
uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* ctx) { return ctx->sys.cpu_virtual_cores; }
uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* ctx) { return ctx->sys.cpu_physical_cores; }
uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* ctx) { return ctx->sys.cpu_sockets; }
uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* ctx) { return ctx->sys.phys_mem_total; }
uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* ctx) { return ctx->sys.phys_mem_free; }
bool mag_ctx_is_numa_system(const mag_ctx_t* ctx) { return false; /* TODO */ }
size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* ctx) { return 0; /* TODO */ }

static mag_intrusive_chunk* mag_fixed_pool_chunk_new(size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    size_t cap = blocks_per_chunk*block_size;
    uintptr_t size = 0;
    mag_pincr((void**)&size, sizeof(mag_intrusive_chunk), __alignof(mag_intrusive_chunk));
    mag_pincr((void**)&size, cap, block_align);
    void* base = (*mag_alloc)(NULL, size), *pos = base;
    mag_intrusive_chunk* chunk = mag_pincr(&pos, sizeof(mag_intrusive_chunk), __alignof(mag_intrusive_chunk));
    uint8_t* bot = mag_pincr(&pos, cap, block_align);
    *chunk = (mag_intrusive_chunk) {
        .bot = bot,
        .top = bot+cap,
        .next = NULL
    };
    return chunk;
}

void mag_fixed_intrusive_pool_init(mag_fixed_intrusive_pool* pool, size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    mag_assert2(blocks_per_chunk);
    block_size = mag_xmax(sizeof(void*), block_size); /* Ensure block size is at least sizeof(void*) to store intrusive free list. */
    mag_intrusive_chunk* chunk = mag_fixed_pool_chunk_new(block_size, block_align, blocks_per_chunk);
    *pool = (mag_fixed_intrusive_pool) {
        .block_size = block_size,
        .block_align = block_align,
        .blocks_per_chunk = blocks_per_chunk,
        .chunks = chunk,
        .chunk_head = chunk,
        .free_list = NULL,
        .num_freelist_hits = 0,
        .num_pool_hits = 0,
        .num_chunks = 1,
        .num_allocs = 0
    };
}

void* mag_fixed_intrusive_pool_malloc(mag_fixed_intrusive_pool* pool) {
    ++pool->num_allocs;
    if (mag_likely(pool->free_list)) { /* 1. Try to pop from free_list (fastest path) */
        ++pool->num_freelist_hits;
        void* blk = pool->free_list;
        pool->free_list = *(void**)blk; /* Next free block is stored at block [0..sizeof(void*)-1] */
        return blk;
    }
    mag_intrusive_chunk* chunk = pool->chunk_head;
    mag_assert2(chunk);
    uint8_t* top = chunk->top-pool->block_size;
    if (mag_likely(top >= chunk->bot)) {  /* 2. Allocate from the last pool if possible (fast path) */
        ++pool->num_pool_hits;
        chunk->top = top;
        return top;
    }
    mag_intrusive_chunk* new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk);     /* 3. Current chunk is exhausted, allocate new (slow path) */
    chunk->next = new_chunk;
    pool->chunk_head = new_chunk;
    new_chunk->top -= pool->block_size;
    ++pool->num_chunks;
    return new_chunk->top;
}

void mag_fixed_intrusive_pool_free(mag_fixed_intrusive_pool* pool, void* blk) { /* Push chunk into free list */
    *(void**)blk = pool->free_list;
    pool->free_list = blk;
}

void mag_fixed_intrusive_pool_destroy(mag_fixed_intrusive_pool* pool) {
    mag_intrusive_chunk* chunk = pool->chunks;
    while (chunk) {
        mag_intrusive_chunk* next = chunk->next;
        (*mag_alloc)(chunk, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

MAG_COLDPROC void mag_fixed_intrusive_pool_print_info(mag_fixed_intrusive_pool* pool, const char* name) {
    mag_log_info("Fixed Intrusive Pool: %s", name);
    mag_log_info(
        "\tBlock Size: %zu B, Block Align: %zu B, Blocks Per Chunk: %zu B",
        pool->block_size,
        pool->block_align,
        pool->blocks_per_chunk
    );
    mag_log_info(
        "\tChunks: %zu, Allocs: %zu, Freelist Hits: %zu, Num Pool Hits: %zu",
        (size_t)pool->num_chunks,
        (size_t)pool->num_allocs,
        (size_t)pool->num_freelist_hits,
        (size_t)pool->num_pool_hits
    );
    double mem_alloced, pool_mem;
    const char* mem_unit_alloced, *mem_unit_pool;
    mag_humanize_memory_size(pool->num_chunks*pool->blocks_per_chunk*pool->block_size, &mem_alloced, &mem_unit_alloced);
    mag_humanize_memory_size(pool->num_allocs*pool->block_size, &pool_mem, &mem_unit_pool);
    mag_log_info("\t Real Mem Allocated: %.03f %s, Total Pool Mem %.03f %s", mem_alloced, mem_unit_alloced, pool_mem, mem_unit_pool);
}

void mag_ctx_profile_start_recording(mag_ctx_t* ctx) {
    if (ctx->profiler_enabled) return;
    memset(ctx->op_perf_mons_total, 0, sizeof(ctx->op_perf_mons_total));
    ctx->profiler_enabled = true;
}

typedef struct mag_op_perf_record_t {
    mag_op_perf_info_t perf;
    mag_op_t op;
} mag_op_perf_record_t;

static int mag_cmp_perf_info(const void* x, const void* y) {
    const mag_op_perf_record_t* op1 = (const mag_op_perf_record_t *)x;
    const mag_op_perf_record_t* op2 = (const mag_op_perf_record_t *)y;
    if (op1->perf.elapsed_ns_acc < op2->perf.elapsed_ns_acc) return 1;
    if (op1->perf.elapsed_ns_acc > op2->perf.elapsed_ns_acc) return -1;
    return 0;
}

void mag_ctx_profile_stop_recording(mag_ctx_t* ctx, const char* export_csv_file) {
    mag_assert(ctx->profiler_enabled, "Profiler must be enabled to generate report");
    ctx->profiler_enabled = false;
    bool csv = export_csv_file && *export_csv_file;
    if (!csv) {
        mag_print_separator(stdout);
        printf("OS/Kernel: %s\n", ctx->sys.os_name);
        printf("CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u\n", ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
        #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        printf("CPU Features:");
        for (unsigned i=0, k=0; i < MAG_X86_64_FEATURE__COUNT; ++i) {
            if (mag_ctx_x86_64_cpu_has_feature(ctx, i)) {
                if (k++ % 8 == 0) printf("\n\t");
                printf("%s ", mag_x86_64_feature_names[i]);
            }
        }
        putchar('\n');
        #endif
        double mem_total, mem_free, mem_used;
        const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
        mag_humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
        mag_humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
        mag_humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
        double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
        printf("Physical memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)\n", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
        mag_print_separator(stdout);
        printf("%16s %16s %16s %16s %16s\n", "Operation", "Executions", "Usage (%)", "AVG Time (μs)", "Total Time (μs)");
    }
    mag_op_perf_record_t sorted[MAG_OP__NUM];
    uint64_t exec_total = 0;
    for (mag_op_t op=MAG_OP_NOP; op < MAG_OP__NUM; ++op) { /* Convert to sortable record. */
        sorted[op].op = op;
        sorted[op].perf = ctx->op_perf_mons_total[op];
        exec_total += sorted[op].perf.n_execs;
    }
    if (mag_unlikely(!exec_total) && !csv) {
        printf("\n! No operations profiled. Enable profiler and execute any operation to see results.\n");
        mag_print_separator(stdout);
        return;
    }
    qsort(sorted, MAG_OP__NUM, sizeof(*sorted), &mag_cmp_perf_info); /* Quicksort by time descending. */
    FILE* f = NULL;
    if (csv) {
        f = mag_fopen(export_csv_file, "wt");
        mag_assert(f, "Failed to open CSV file: %s", export_csv_file);
        fprintf(f, "Operation,Executions,Usage,AVG Time,Total Time\n"); /* CSV Header */
    }
    for (mag_op_t i=MAG_OP_NOP; i < MAG_OP__NUM; ++i) { /* Format sorted performance data */
        const mag_op_perf_record_t* info = sorted+i;
        const mag_op_perf_info_t* perf = &info->perf;
        if (!perf->n_execs) continue; /* Op never executed. */
        const char* op_name = mag_op_meta_of(info->op)->mnemonic;
        double perc_exec = (double)perf->n_execs/(double)exec_total * 100.0;
        char perc_exec_str[64];
        snprintf(perc_exec_str, sizeof(perc_exec_str), "%.1f", perc_exec);
        double avg_time = (double)perf->elapsed_ns_acc/1e3/(double)perf->n_execs;
        char avg_time_str[64];
        snprintf(avg_time_str, sizeof(avg_time_str), "%f", avg_time);
        double tot_time = (double)perf->elapsed_ns_acc/1e3;
        char tot_time_str[64];
        snprintf(tot_time_str, sizeof(tot_time_str), "%f", tot_time);
        if (csv) {
            fprintf(f, "%s,%" PRIu64 ",%s,%s,%s\n", op_name, perf->n_execs, perc_exec_str, avg_time_str, tot_time_str);
        } else {
            printf("%16s %16" PRIu64 " %16s%16s%16s\n", op_name, perf->n_execs, perc_exec_str, avg_time_str, tot_time_str);
        }
    }
    if (csv) fclose(f);
    else {
        putchar('\n');
        printf("Total operations profiled: %" PRIu64 "\n", exec_total);
        mag_print_separator(stdout);
    }
}

uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b) { return ((uint32_t)r<<16)|((uint32_t)g<<8)|(uint32_t)b; }
uint32_t mag_pack_color_f32(float r, float g, float b) {
    return (((uint32_t)(r*255.0f)&255)<<16)|(((uint32_t)(g*255.0f)&255)<<8)|((uint32_t)(b*255.0f)&255);
}

const char* mag_device_type_get_name(mag_compute_device_type_t op) {
    static const char* const names[MAG_COMPUTE_DEVICE_TYPE__NUM] = {
        [MAG_COMPUTE_DEVICE_TYPE_CPU] = "CPU",
        [MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA] = "CUDA GPU",
    };
    return names[op];
}

const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type) {
    static const mag_dtype_meta_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_F32] = {
            sizeof(float),
            "f32"
        },
    };
    return &infos[type];
}

/*
**  validation error print template
**
**
printf("SHORT ERROR DESCRIPTION"
        "ERROR: Failed to execute operation: %s.\n"
        "    - Input Tensor 1 '%s': MISMATCHES\n"
        "    - Input Tensor 2 '%s': MISMATCHES\n"
        "    Hint: ANY HINT FOR USR."
);
*/

static bool mag_validate_inputs(mag_op_t op, mag_tensor_t** inputs, uint32_t numin) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_unlikely(meta->argcount != numin || numin > MAG_MAX_INPUT_TENSORS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            meta->mnemonic, meta->argcount, numin
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->argcount; ++i) {
        if (mag_unlikely(!inputs[i])) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Input tensor %u is NULL.\n"
                "    Hint: Ensure all input tensors are valid and non-NULL.\n",
                meta->mnemonic, i
            );
            mag_print_separator(stderr);
            fputc('\n', stderr);
            fflush(stderr);
            return false;
        }
    }
    return true;
}

static bool mag_validate_op_params(mag_op_t op, const mag_op_param_t* params, uint32_t numparams) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (!meta->paramcount) return true; /* No parameters to validate. */
    if (mag_unlikely(meta->paramcount != numparams || numparams > MAG_MAX_OP_PARAMS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires at most %u parameters, but %u were provided.\n"
            "    Hint: Ensure the correct number of operation parameters are provided.\n",
            meta->mnemonic, MAG_MAX_OP_PARAMS, meta->paramcount
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    if (mag_unlikely(!params)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation parameters are NULL.\n"
            "    Hint: Ensure all operation parameters are valid and non-NULL.\n",
            meta->mnemonic
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->paramcount; ++i) {
        if (mag_unlikely(params[i].type != meta->param_types[i])) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Operation parameter %u type mismatch.\n"
                "    - Expected type id: %d\n"
                "    - Provided type id: %d\n"
                "    Hint: Ensure the correct parameter types are provided.\n",
                meta->mnemonic, i, meta->param_types[i], params[i].type
            );
        }
    }
    return true;
}

static bool mag_validate_shape_eq(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_shape_eq(a, b))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be equal.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

static bool mag_validate_shape_broadcastable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_can_broadcast(b, a))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    char broadcast_able_str[MAG_MAX_DIMS*2+4+1] = {0};
    char* p = broadcast_able_str;
    *p++ = '[';
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i) {
        *p++ = a->shape[i] % b->shape[i] == 0 ? 'Y' : 'N';
        *p++ = i < MAG_MAX_DIMS-1 ? ',' : ']';
    }
    *p = '\0';
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be broadcast-able.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Broadcast-able: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2,
        broadcast_able_str
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

#define mag_validate_expr_gen(expr, message, ...) \
    if (mag_unlikely(!(expr))) { \
        if (1) { \
           mag_log_error(message, ## __VA_ARGS__); \
        } \
        return false; \
    }

static bool mag_validate_op_unary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    if (mag_unlikely(!mag_validate_shape_eq(op, result, inputs[0]))) return false;
    return true;
}

static bool mag_validate_op_binary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    if (mag_unlikely(!mag_validate_shape_eq(op, result, inputs[0]))) return false;
    if (mag_unlikely(!mag_validate_shape_broadcastable(op, inputs[0], inputs[1]))) return false;
    mag_validate_expr_gen(mag_tensor_is_contiguous(result), "Result must be contiguous.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "First tensor must be contiguous.");
    return true;
}

static bool mag_validate_op_transpose(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return true;
}

static bool mag_validate_op_scalar(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "Mean"); /* TODO */
    mag_validate_expr_gen(result->shape[0] == 1, "Mean");
    mag_validate_expr_gen(result->shape[1] == inputs[0]->shape[1], "Mean");
    mag_validate_expr_gen(result->shape[2] == inputs[0]->shape[2], "Mean");
    mag_validate_expr_gen(result->shape[3] == inputs[0]->shape[3], "Mean");
    return true;
}

static bool mag_validate_op_matmul(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    mag_validate_expr_gen(inputs[0]->shape[1] == inputs[1]->shape[0], "Input tensor shapes must be compatible for matrix multiplication.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "First tensor must be contiguous.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[1]), "Second tensor must be contiguous.");
    mag_validate_expr_gen(inputs[1]->shape[2] % inputs[0]->shape[2] == 0, "Result tensor shape mismatch.");
    mag_validate_expr_gen(inputs[1]->shape[3] % inputs[0]->shape[3] == 0, "Result tensor shape mismatch.");
    return true;
}

static mag_tensor_t* mag_tensor_create(mag_ctx_t* ctx, mag_dtype_t type, const int64_t* dims, int64_t rank, mag_tensor_t* view, size_t view_offs);

static mag_tensor_t* mag_result_constructor_routine_isomorph(mag_tensor_t** inputs, const mag_op_param_t* params) {
    (void)params;
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, inputs[0]->rank, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_view(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    (void)params;
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, MAG_MAX_DIMS, inputs[0], 0);
}

static mag_tensor_t* mag_result_constructor_routine_scalar(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    int64_t shape[MAG_MAX_DIMS];
    *shape = 1;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)
        shape[i] = inputs[0]->shape[i];
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, shape, MAG_MAX_DIMS, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_transposed(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* transposed = mag_result_constructor_routine_view(inputs, params);
    mag_swap(int64_t, transposed->shape[0], transposed->shape[1]);
    mag_swap(int64_t, transposed->strides[0], transposed->strides[1]);
    return transposed;
}

static mag_tensor_t* mag_result_constructor_routine_permuted(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_assert2(params != NULL); /* TODO */
    mag_tensor_t* permuted = mag_result_constructor_routine_view(inputs, params);
    uint32_t axes[MAG_MAX_DIMS];
    for (uint32_t i = 0; i < MAG_MAX_DIMS; ++i) /* Unpack axes */
        axes[i] = params[i].x.u32;
    for (uint32_t i = 0; i < MAG_MAX_DIMS; ++i) { /* Check that all axes are unique */
        for (uint32_t j = i+1; j < MAG_MAX_DIMS; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %zu != %zu", axes[i], axes[j]);
    }
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i) { /* Permute shape and strides */
        mag_assert2(axes[i] >= 0 && axes[i] < MAG_MAX_DIMS);
        permuted->shape[axes[i]] = inputs[0]->shape[i];
        permuted->strides[axes[i]] = inputs[0]->strides[i];
    }
    return permuted;
}

static mag_tensor_t* mag_result_constructor_routine_matmul(mag_tensor_t** inputs,  const mag_op_param_t* params) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[MAG_MAX_DIMS];
    shape[0] = inputs[0]->shape[0]; /* M */
    shape[1] = inputs[1]->shape[1]; /* R */
    return mag_tensor_create(inputs[0]->ctx, MAG_DTYPE_F32, shape, 2, NULL, 0);
}

const mag_op_meta_t* mag_op_meta_of(mag_op_t type) {
    static const mag_op_meta_t infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .argcount = 0,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = NULL,
            .validator = NULL
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_view,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_transposed,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .argcount = 1,
            .paramcount = MAG_MAX_DIMS,
            .param_types = {
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
            },
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_permuted,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_MEAN] = {
            .mnemonic = "mean",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MIN] = {
            .mnemonic = "min",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MAX] = {
            .mnemonic = "max",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_SUM] = {
            .mnemonic = "sum",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_NEG] = {
            .mnemonic = "neg",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_LOG] = {
            .mnemonic = "log",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SQR] = {
            .mnemonic = "sqr",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SQRT] = {
            .mnemonic = "sqrt",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIN] = {
            .mnemonic = "sin",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_COS] = {
            .mnemonic = "cos",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_STEP] = {
            .mnemonic = "step",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SOFTMAX] = {
            .mnemonic = "softmax",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SOFTMAX_DV] = {
            .mnemonic = "softmax_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIGMOID] = {
            .mnemonic = "sigmoid",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIGMOID_DV] = {
            .mnemonic = "sigmoid_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_HARD_SIGMOID] = {
            .mnemonic = "hard_sigmoid",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SILU] = {
            .mnemonic = "silu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SILU_DV] = {
            .mnemonic = "silu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TANH] = {
            .mnemonic = "tanh",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TANH_DV] = {
            .mnemonic = "tanh_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_RELU] = {
            .mnemonic = "relu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_RELU_DV] = {
            .mnemonic = "relu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_GELU] = {
            .mnemonic = "gelu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_GELU_DV] = {
            .mnemonic = "gelu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_ADD] = {
            .mnemonic = "add",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_SUB] = {
            .mnemonic = "sub",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_MUL] = {
            .mnemonic = "mul",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_DIV] = {
            .mnemonic = "div",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_ADDS] = {
            .mnemonic = "adds",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SUBS] = {
            .mnemonic = "subs",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_MULS] = {
            .mnemonic = "muls",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_DIVS] = {
            .mnemonic = "divs",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_matmul,
            .validator = &mag_validate_op_matmul
        }
    };
    return infos+type;
}

#undef mag_validate_inputs

int64_t mag_tensor_data_size(const mag_tensor_t* t) { return t->numel*mag_dtype_meta_of(t->dtype)->size; }
int64_t mag_tensor_numel(const mag_tensor_t* t) { return t->numel; }
int64_t mag_tensor_num_rows(const mag_tensor_t* t) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    return d1*d2*d3*d4*d5;
}
int64_t mag_tensor_num_cols(const mag_tensor_t* t) { return *t->shape; }

#if MAG_SANITIZE_RC
    static void mag_tensor_sanitize_dtor(mag_tensor_t* t) {
        (void)t;
    }
#endif

static mag_tensor_t* mag_tensor_create(mag_ctx_t* ctx, mag_dtype_t type, const int64_t* dims, int64_t rank, mag_tensor_t* view, size_t view_offs) {
    uintptr_t tr_id = mag_thread_id();
    mag_assert(tr_id == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", tr_id, ctx->tr_id);
    mag_assert(dims != NULL && rank >= 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
    mag_assert2(view_offs == 0); /* NYI. TODO */
    if (view) {
        if (view->view_uplink) { /* Traverse view chain and accumulate offset */
            view_offs += view->view_offs;
            view = view->view_uplink;
        }
        mag_tensor_incref(view); /* Increment view tensor strong RC */
    }
    int64_t dts = mag_dtype_meta_of(type)->size;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) /* Calculate buffer size and check for overflow. */
        mag_assert2(dims[i] > 0 && !mag_imull64_ov(dims[i], numel, &numel)); /* Overflow in buffer size. Max: INT64_MAX. Reduce dimensions. */
    int64_t numbytes = numel*dts;
    mag_assert2(!view || !numbytes || numbytes + view_offs <= mag_tensor_data_size(view)); /* Slice must be within viewed tensor data range. *//* Allocate memory for tensor struct on CPU RAM. */
    mag_tensor_t* t = mag_fixed_intrusive_pool_malloc(&ctx->tensor_pool);
    memset(t, 0, sizeof(*t));
    *t = (mag_tensor_t) {
        .rcb = {
            .rc_strong = 0,
            .rc_weak = 0,
            #if MAG_SANITIZE_RC
                .dtor = &mag_tensor_sanitize_dtor
            #endif
        },
        .ctx = ctx,
        .rank = rank,
        .shape = {0},
        .strides = {0},
        .dtype = type,
        .storage = {0},
        .numel = numel,
        .flags = view ? MAG_TFLAG_VIEW : MAG_TFLAG_OWNER,
        .op = MAG_OP_NOP,
        .op_inputs = {0},
        .op_params = {{0}},
        .view_uplink = view,
        .view_offs = view_offs,
        .grad = NULL,
        .pmon = {0},
        .name = "",
        .ud = NULL
    };
    mag_tensor_incref(t); /* First strong RC=1 */
    /* Allocate device memory */
    mag_compute_device_t* dvc = ctx->device;
    void (*allocator)(mag_compute_device_t*, mag_storage_buffer_t*, size_t, size_t) = dvc->alloc_storage;
    if (view) t->storage = view->storage; /* Reference memory from view */
    else (*allocator)(dvc, &t->storage, numbytes, dts); /* Allocate new device memory */
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)    /* Copy dimensions and set unused to identity. */
        t->shape[i] = i < rank ? dims[i] : 1;
    *t->strides = 1;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)    /* Calculate strides and check for overflow. */
        mag_assert2(!mag_imull64_ov(t->strides[i-1], t->shape[i-1], t->strides+i));
#if MAG_SANITIZE_RC /* If tensor RC sanitize is enabled, insert into tracking list */
    mag_tensor_node_t** head = &ctx->rc_tracked;
    mag_tensor_node_t* node = (*mag_alloc)(NULL, sizeof(*node));
    *node = (mag_tensor_node_t) {
        .tensor = t,
        .next = *head
    };
    *head = node;
#endif
    return t;
}

static void mag_tensor_destroy(mag_tensor_t* t) {
    mag_ctx_t* ctx = t->ctx;
#if MAG_SANITIZE_RC  /* If tensor RC sanitize is enabled, invoke destructor and erase from tracking list */
    void (*dtor)(mag_tensor_t*) = t->rcb.dtor;  /* Invoke Debug destructor. */
    if (dtor) (*dtor)(t);
    mag_tensor_node_t** head = &ctx->rc_tracked;
    if (*head) {
        mag_tensor_node_t* curr = *head, *prev = NULL;
        if (curr->tensor == t) { /* Head itself holds key */
            *head = curr->next;
            (*mag_alloc)(curr, 0);
        } else {
            while (curr && curr->tensor != t) { /* Find node */
                prev = curr;
                curr = curr->next;
            }
            if (curr) { /* Found node */
                prev->next = curr->next;
                (*mag_alloc)(curr, 0); /* Free node */
            }
        }
    }
#endif
    if (t->flags & MAG_TFLAG_OWNER) { /* Free device memory if tensor owns it. */
        mag_compute_device_t* dvc = t->ctx->device;
        void (*dtor)(mag_compute_device_t*, mag_storage_buffer_t*) = dvc->free_storage;
        (*dtor)(dvc, &t->storage);
    }
    mag_fixed_intrusive_pool_free(&ctx->tensor_pool, t);
}

void mag_tensor_incref(mag_tensor_t* t) {
    mag_assert2(t->rcb.rc_strong < UINT32_MAX);
    ++t->rcb.rc_strong;
}

bool mag_tensor_decref(mag_tensor_t* t) {
    if (t->view_uplink) { /* If tensor is a view, decrement base RC and free tensor chain */
        mag_tensor_decref(t->view_uplink);
    }
    if (!--t->rcb.rc_strong) { /* Strong RC reaches zero, destroy. */
        mag_tensor_destroy(t);
        return true;
    }
    return false;
}

mag_tensor_t* mag_tensor_create_1d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1}, 1, NULL, 0);
}

mag_tensor_t* mag_tensor_create_2d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2}, 2, NULL, 0);
}

mag_tensor_t* mag_tensor_create_3d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3}, 3, NULL, 0);
}

mag_tensor_t* mag_tensor_create_4d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4}, 4, NULL, 0);
}

mag_tensor_t* mag_tensor_create_5d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5}, 5, NULL, 0);
}

mag_tensor_t* mag_tensor_create_6d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5, d6}, 6, NULL, 0);
}

static void MAG_HOTPROC mag_op_exec(mag_tensor_t* R, mag_compute_device_t* dvc, mag_graph_eval_order_t ord) {
    mag_perf_mon_t* pmon = &R->pmon;
    mag_op_perf_info_t (*pmon_ops)[MAG_OP__NUM] = &R->ctx->op_perf_mons_total;
    mag_op_perf_info_t* pmon_op = (*pmon_ops)+R->op;
    uint64_t start = R->ctx->profiler_enabled ? mag_hpc_clock_ns() : 0;    /* Profiling monitoring */
    void (*exec)(mag_compute_device_t*, mag_tensor_t*) = ord == MAG_GRAPH_EVAL_ORDER_FORWARD ? dvc->eager_exec_fwd : dvc->eager_exec_bwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
    if (!R->ctx->profiler_enabled) return; /* Profiling disabled. */
    pmon->elapsed_ns = mag_hpc_clock_elapsed_ns(start);
    pmon->elapsed_ns_acc += pmon->elapsed_ns;
    pmon_op->elapsed_ns_acc += pmon->elapsed_ns;
    ++pmon->n_execs;
    ++pmon_op->n_execs;
}

static mag_tensor_t* MAG_HOTPROC mag_tensor_operator(
    mag_ctx_t* ctx,
    mag_op_t op,
    bool inplace,
    mag_tensor_t** inputs,
    uint32_t numin,
    const mag_op_param_t* params,
    uint32_t numparams
) {
    /* Validate inputs and params first */
    mag_assert2(op != MAG_OP_NOP);
    mag_assert(inputs && mag_validate_inputs(op, inputs, numin), "Invalid input tensors for operation %s.", mag_op_meta_of(op)->mnemonic);
    mag_assert(mag_validate_op_params(op, params, numparams), "Invalid parameters for operation %s.", mag_op_meta_of(op)->mnemonic);

    const mag_op_meta_t* meta = mag_op_meta_of(op);
    mag_graph_eval_order_t gra = MAG_GRA_FWD; /* TODO */
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_op_param_t*) = meta->r_alloc;
    bool (*validate_op)(mag_op_t, mag_tensor_t*, mag_tensor_t**, const mag_op_param_t*) = meta->validator;
    mag_tensor_t* R = (inplace && numin && meta->inplace)                                                   /* Inplace requested? */
        ? mag_tensor_create(ctx, (*inputs)->dtype, (*inputs)->shape, (*inputs)->rank, *inputs, 0)  /* View R <- X for inplace aliasing op. */
        : (*r_alloc)(inputs, params);                                                                       /* Construct new result tensor. */
    if (mag_unlikely(!(*validate_op)(op, R, inputs, params))) return NULL;                                  /* Validation failed. */
    mag_tensor_t* grad = NULL;                                                                              /* ∇ᵦL = ∂L/∂B - Upper gradient tensor. */  /* TODO */
    if (gra == MAG_GRA_BWD && grad) {
        R->grad = R->grad                       /* ∇ₐL = ∑ᵢ (∂L/∂Bᵢ) ⋅ (∂Bᵢ/∂A) - Chain rule accumulate. */
            ? mag_add(R->grad, grad)      /* ∇ₐL <- ∇ₐL + ∇ᵦL */
            : mag_clone(grad);                  /* ∇ₐL <- ∂L/∂B */
    }
    mag_assert2(R->op == MAG_OP_NOP);
    R->op = op;                                                         /* Set operation for deferred execution mode. */
    for (uint32_t i=0; i < numin; ++i) {                             /* Set input tensors and flags. */
        R->op_inputs[i] = inputs[i];
    }
    if (params) memcpy(R->op_params, params, sizeof(*params));   /* Copy operation parameters */
    if (ctx->exec_mode == MAG_EXEC_MODE_EAGER) {                    /* In eager execution mode, we execute immediately. */
        mag_op_exec(R, ctx->device, gra);                           /* Execute the operation immediately. */
    }
    return R;
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_view(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_transpose(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t d4, uint32_t d5) {
    mag_op_param_t params[MAG_MAX_OP_PARAMS] = {
        {.type=MAG_OP_TPARAM_U32, .x.u32=d0},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d1},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d2},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d3},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d4},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d5}
    };
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, params, sizeof(params)/sizeof(*params));
}

mag_tensor_t* mag_mean(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_min(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_max(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sum(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_abs(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_abs_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_neg(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_neg_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_log(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_log_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqr(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqr_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqrt(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqrt_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sin(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sin_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_cos(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_cos_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_step(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_step_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_sub(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_mul(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }

mag_tensor_t* mag_adds(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_adds_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_subs(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_subs_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_muls(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_muls_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_divs(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_divs_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0);
}

static MAG_AINLINE void mag_tensor_virtual_to_physical_index(const mag_tensor_t* t, int64_t v_idx, int64_t(*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

static MAG_AINLINE int64_t mag_tensor_physical_to_virtual_index(const mag_tensor_t* t, const int64_t (*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_load_local_storage_group_arr(*p_idx, i);
    return s0*i0 + s1*i1 + s2*i2 + s3*i3 + s4*i4 + s5*i5;
}

mag_tensor_t* mag_tensor_get_arg(const mag_tensor_t* t, size_t slot) {
    mag_assert(slot < MAG_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MAG_MAX_INPUT_TENSORS);
    return t->op_inputs[slot];
}

void mag_tensor_set_arg(mag_tensor_t* t, size_t slot, mag_tensor_t* arg) {
    mag_assert(slot < MAG_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MAG_MAX_INPUT_TENSORS);
    mag_assert(t->op_inputs[slot] == NULL, "Argument at slot #%zu already set", slot);
    t->op_inputs[slot] = arg;
}

void mag_tensor_copy_buffer_from(mag_tensor_t* t, const void* data, size_t size) {
    mag_assert(size == (size_t) mag_tensor_data_size(t), "Buffer size mismatch: %zu != %lld", size, mag_tensor_data_size(t));
    mag_storage_buffer_t* sto = &t->storage;
    (*sto->cpy_host_device)(sto, 0, data, size);
}

void mag_tensor_fill(mag_tensor_t* t, float x) {
    if (x == 0.0f) {
        mag_storage_buffer_t* sto = &t->storage;
        (*sto->set)(sto, 0, 0); /* Zero out the buffer. */
        return;
    }
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);

    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            int64_t n = mag_tensor_numel(t);
            float* buf = (float*)t->storage.base;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: mag_panic("Unsupported DType: %d", t->dtype);
    }
}

void mag_tensor_fill_random_uniform(mag_tensor_t* t, float min, float max) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            int64_t n = mag_tensor_numel(t);
            float* buf = (float*)t->storage.base;
            mag_prng_generate_n(t->ctx, buf, n, min, max); /* Generate uniform random numbers. */
        } break;
        default: mag_panic("Unsupported DType: %d", t->dtype);
    }
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, float mean, float stddev) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            int64_t n = mag_tensor_numel(t);
            mag_assert((n & 1) == 0, "Number of elements must be even");
            float* buf = (float*)t->storage.base;
            mag_prng_generate_n(t->ctx, buf, n, 0.0f, 1.0f); /* Generate uniform random numbers. */
            for (int64_t i=0; i < n; i += 2) { /* Map uniform to normal distribution using Box-Muller transform. */
                float* u1 = buf+i;
                float* u2 = buf+i+1;
                float mag = stddev*sqrtf(-2.0f*logf(*u1));
                float y0 = mag*cosf((float)(2.0*M_PI)*(*u2)) + mean;
                float y1 = mag*sinf((float)(2.0*M_PI)*(*u2)) + mean;
                *u1 = y0;
                *u2 = y1;
            }
        } break;
        default: mag_panic("Unsupported DType: %d", t->dtype);
    }
}

uint64_t mag_tensor_get_packed_refcounts(const mag_tensor_t* t) {
    return (uint64_t)t->rcb.rc_strong|((uint64_t)t->rcb.rc_weak << 32);
}

void mag_tensor_retain(mag_tensor_t* t) {
    ++t->rcb.rc_strong;
}

size_t mag_tensor_get_memory_usage(const mag_tensor_t* t) {
    return sizeof(*t) + mag_tensor_data_size(t);
}

static void mag_print_tensor_recursive(FILE* f, const mag_tensor_t* t, int64_t (*idx)[MAG_MAX_DIMS], const int64_t (*stri)[MAG_MAX_DIMS], int64_t curr_dim, int64_t total_dims, int indent) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    int64_t dim_size = t->shape[curr_dim];
    mag_load_local_storage_group_arr(*stri, s);
    if (curr_dim == total_dims - 1) {
        fprintf(f, "%*s[", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            mag_load_local_storage_group_arr(*idx, i);
            float val = *((const float*)t->storage.base + i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5);
            char fmt_buf[128];
            *mag_fmt_f64(MAG_FMT_G14, (double)val, fmt_buf) = '\0';
            fprintf(f, "%s", fmt_buf);
            if (i < dim_size - 1) {
                fprintf(f, " ");
            }
        }
        fprintf(f, "]");
    } else {
        fprintf(f, "%*s[\n", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            mag_print_tensor_recursive(f, t, idx, stri, curr_dim + 1, total_dims, indent + 1);
            if (i < dim_size - 1) {
                fprintf(f, ",\n");
            } else {
                fprintf(f, "\n");
            }
        }
        fprintf(f, "%*s]\n", indent, "");
    }
}


void mag_tensor_print(const mag_tensor_t* t, bool with_header, bool with_data) {
    mag_assert(t->dtype == MAG_DTYPE_F32, "Tensor must be F32");
    mag_assert2(with_header || with_data);
    mag_load_local_storage_group(t, x_d, shape);
    mag_load_local_storage_group(t, x_s, strides);
    FILE* f = stdout;
    if (with_header) {
        double buf_size_cvt = 0.0;
        const char* buf_size_unit = NULL;
        mag_humanize_memory_size(mag_tensor_get_memory_usage(t), &buf_size_cvt, &buf_size_unit);
        char shape[MAG_FMT_DIM_BUF_SIZE];
        char strides[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_dims(&shape, &t->shape, t->rank);
        mag_fmt_dims(&strides, &t->strides, MAG_MAX_DIMS);
        static const char* flag_abbrs = "OVGE";
        mag_assert2(strlen(flag_abbrs) == MAG_TFLAG_LEN);
        char flags[MAG_TFLAG_LEN+1] = {0};
        for (uint32_t i=0, k=0; i < MAG_TFLAG_LEN; ++i)
            if (t->flags & (1 << i))
                flags[k++] = flag_abbrs[i];
        flags[MAG_TFLAG_LEN] = '\0';
        fprintf(f, "Tensor '%s', DType: %s, Rank: %" PRIi64 ", Elements: %" PRIi64 ", Shape: %s, Strides: %s, Mem: %.03f %s, Flags: %s (%x)\n",
            t->name,
            mag_dtype_meta_of(t->dtype)->name,
            t->rank,
            mag_tensor_numel(t),
            shape,
            strides,
            buf_size_cvt,
            buf_size_unit,
            flags,
            t->flags
        );
    }
    if (with_data) {
        int64_t strides[MAG_MAX_DIMS];
        strides[MAG_MAX_DIMS-1] = 1;
        for (int32_t i = MAG_MAX_DIMS-2; i >= 0; --i)    // TODO: Fix this
            strides[i] = strides[i+1] * t->shape[i+1];
        int64_t idx[MAG_MAX_DIMS] = {0};
        mag_print_tensor_recursive(f, t, &idx, &strides, 0, t->rank, 0);
    }
}

void mag_tensor_set_name(mag_tensor_t* t, const char* name) {
    strncpy(t->name, name, MAG_MAX_TENSOR_NAME_LEN);
    t->name[MAG_MAX_TENSOR_NAME_LEN-1] = '\0';
}

void mag_tensor_fmt_name(mag_tensor_t* t, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(t->name, sizeof(t->name), fmt, args);
    va_end(args);
}

const char* mag_tensor_get_name(const mag_tensor_t* t) {
    return t->name;
}

int64_t mag_tensor_rank(const mag_tensor_t* t) { return t->rank; }
const int64_t* mag_tensor_shape(const mag_tensor_t* t) { return t->shape; }
const int64_t* mag_tensor_strides(const mag_tensor_t* t) { return t->strides; }
mag_dtype_t mag_tensor_dtype(const mag_tensor_t* t) { return t->dtype; }

void* mag_tensor_data_ptr(const mag_tensor_t* t) {
    return (void*)t->storage.base;
}

bool mag_tensor_is_scalar(const mag_tensor_t* t) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_vector(const mag_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_matrix(const mag_tensor_t* t) {
    #pragma GCC unroll 4
    for (uint32_t i=2; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_volume(const mag_tensor_t* t) {
    #pragma GCC unroll 3
    for (uint32_t i=3; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_shape_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    return memcmp(a->shape, b->shape, sizeof(a->shape)) == 0;
}

bool mag_tensor_are_strides_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    return memcmp(a->strides, b->strides, sizeof(a->strides)) == 0;
}

bool mag_tensor_can_broadcast(const mag_tensor_t* a, const mag_tensor_t* b) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)
        if ((b->shape[i] % a->shape[i]) != 0)
            return false;
    return true;
}

bool mag_tensor_is_transposed(const mag_tensor_t* t) { return t->strides[0] > t->strides[1]; }

bool mag_tensor_is_permuted(const mag_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=0; i < MAG_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool mag_tensor_is_contiguous(const mag_tensor_t* t) {
    return *t->strides == 1;
}

float mag_tensor_get_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            float r;
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_device_host)(sto, sizeof(r)*(d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5), &r, sizeof(r));
            return r;
        }
        default: mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

void mag_tensor_set_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_host_device)(sto, sizeof(x)*(d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5), &x, sizeof(x));
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

float mag_tensor_get_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        return mag_tensor_get_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            float r;
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_device_host)(sto, sizeof(r)*v_idx, &r, sizeof(r));
            return r;
        }
        default:
            mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

void mag_tensor_set_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx, float x) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        mag_tensor_set_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], x);
        return;
    }
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_host_device)(sto, sizeof(x)*v_idx, &x, sizeof(x));
        } break;
        default:
            mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

bool mag_tensor_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->numel != b->numel) return false;
    /*int64_t n = mag_tensor_num_elements(a); TODO
    switch (a->dtype) {
        case MAG_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i) {
                if (buf_a[i] != buf_b[i]) {
                    return false;
                }
            }
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_info_of(a->dtype)->name);
    }*/
    return false;
}

bool mag_tensor_is_close(const mag_tensor_t* a, const mag_tensor_t* b, float eps, double* percent_eq) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->numel != b->numel) return false;
    /*eps = eps < 0.0f ? FLT_EPSILON : eps; TODO
    int64_t n = mag_tensor_num_elements(a);
    int64_t n_eq = 0;
    switch (a->dtype) {
        case MAG_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i)   |x - y| <= ε     ∀ x, y ∈ A, B
                if (fabsf(buf_a[i] - buf_b[i]) <= eps) ++n_eq;
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_info_of(a->dtype)->name);
    }
    if (percent_eq) *percent_eq = (double)n_eq / (double)n * 100.0;
    return n_eq == n;*/
    return false;
}

void mag_tensor_img_draw_box(mag_tensor_t* t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb) {
    mag_assert(t->rank == 3, "Tensor must be 3D image tensor");
    mag_assert2(x2 > x1 && y2 > y1 && x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0);
    float* buf = mag_tensor_data_ptr(t);
    int32_t w = (int32_t)mag_tensor_image_width(t);
    int32_t h = (int32_t)mag_tensor_image_height(t);
    int32_t c = (int32_t)mag_tensor_image_channels(t);
    mag_assert2(w && h && c == 3);
    float r = (float)((rgb>>16)&0xff) / 255.0f;
    float g = (float)((rgb>>8)&0xff) / 255.0f;
    float b = (float)(rgb&0xff) / 255.0f;
    wi = mag_xmax(1, wi);
    for (int32_t i=0; i < wi; ++i) {
        int32_t xx1 = x1+i;
        int32_t yy1 = y1+i;
        int32_t xx2 = x2-i;
        int32_t yy2 = y2-i;
        if (mag_unlikely(xx1 >= w)) xx1 = w-1;
        if (mag_unlikely(xx2 >= w)) xx2 = w-1;
        if (mag_unlikely(yy1 >= h)) yy1 = h-1;
        if (mag_unlikely(yy2 >= h)) yy2 = h-1;
        for (int32_t j=xx1; j <= xx2; ++j) {
            float* r1 = buf + j + yy1*w + 0*w*h;
            float* r2 = buf + j + yy2*w + 0*w*h;
            float* g1 = buf + j + yy1*w + 1*w*h;
            float* g2 = buf + j + yy2*w + 1*w*h;
            float* b1 = buf + j + yy1*w + 2*w*h;
            float* b2 = buf + j + yy2*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_data_size(t));
            *r1 = *r2 = r;
            *g1 = *g2 = g;
            *b1 = *b2 = b;
        }
        for (int32_t j = yy1; j <= yy2; ++j) {
            float* r1 = buf + xx1 + j*w + 0*w*h;
            float* r2 = buf + xx2 + j*w + 0*w*h;
            float* g1 = buf + xx1 + j*w + 1*w*h;
            float* g2 = buf + xx2 + j*w + 1*w*h;
            float* b1 = buf + xx1 + j*w + 2*w*h;
            float* b2 = buf + xx2 + j*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_data_size(t));
            *r1 = *r2 = r;
            *g1 = *g2 = g;
            *b1 = *b2 = b;
        }
    }
}

static bool mag_glyph(uint32_t c, uint32_t x, uint32_t y) {
    c -= 33, --x;
    if (mag_unlikely(c > 93 || x > 6 || y > 13)) return false;
    uint32_t i = 98*c + 7*y + x;
    return (("0@P01248@00120000P49B0000000000000:DXlW2UoDX@10008@h;IR4n@R<Y?48000PYDF"
             "PP011J:U1000<T8QQQDAR4a50000@P012000000000000222448@P024@010028P0148@PP011100000"
             "ABELDU410000000048@l7124000000000000000H`01100000000n10000000000000000006<0000@P"
             "P011224488@00000`CXHY:=:D8?0000004<DT01248@000000l4:444444h700000`C8@Ph02D8?0000"
             "008HX89b?8@P000000n58`7@P05b300000`CP0O25:D8?00000POPP0112248000000l4:D8?Q25b300"
             "000`CX@Ql1244700000000H`0000<H00000000`P1000H`0110000044444@014@0000000n100PO000"
             "0000004@014@@@@@0000h948@@@@00120000`G`l5;F\\Lf0n100000l4:DXOQ25:400000hCX@Qn4:D"
             "X?000000?Q248@P0Ql000000N49DX@Q25i100000hGP01N48@PO00000PO124hAP012000000l4:@PLQ"
             "25b3000008DX@Qn5:DX@000000748@P0124L00000001248@P25b3000008DT456D8AT@00000P01248"
             "@P01n10000017G=IbP1364000008dXAU:U:E\\H000000?Q25:DX@Ql000000n4:DX?1248000000`CX"
             "@Q2U:E4GP0000P?Q25jCR8Q2100000l4:@0?P05b300000l71248@P01200000P@Q25:DX@Ql0000002"
             "5:D89BT`P1000004<HbT9[:BT800000P@QT8QQ49Q210000013:B4548@P000000h7888888@PO00000"
             "7248@P01248`10P0148P0148P0148000h01248@P0124>000015A000000000000000000000000h?00"
             "04@010000000000000000l0bGX@aL10000124XcX@Q25j300000000?Q248@8?000008@Pl5:DX@aL10"
             "000000`CX@o24`70000`AP01N48@P0100000000l5:DX@aL12T70124XcX@Q25:40000@P0P348@P01>"
             "00000240HP01248@P0a101248@T47B4940000HP01248@P01L00000000oBV<IbT910000000hCX@Q25"
             ":400000000?Q25:D8?00000000j<:DX@Qn48@00000`GX@Q25c58@P0000P>S248@P000000000l48P7"
             "@Pn0000048@`31248@030000000P@Q25:D<G0000000025:T49<H000000004<HbTE5920000000P@QT"
             "`@BX@0000000025:DX@aL12T70000h744444h70000PS01248>P0124`1001248@P01248@P0007@P01"
             "24`@P01R30000000S9S10000000"[i/6]-'0')>>(i%6))&1;
}

void mag_tensor_img_draw_text(mag_tensor_t* t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* txt) { /* TODO: Implement font scaling, size is ignored currently */
    mag_assert(t->rank == 3, "Tensor must be a 3D image tensor");
    mag_assert2(x >= 0 && y >= 0 && size >= 8 && txt && *txt);
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    float* buf = (float*)t->storage.base;
    int32_t w = (int32_t)mag_tensor_image_width(t);
    int32_t h = (int32_t)mag_tensor_image_height(t);
    int32_t c = (int32_t)mag_tensor_image_channels(t);
    mag_assert2(w && h && c == 3);
    float* pr = buf;
    float* pg = buf + w*h;
    float* pb = buf + w*h*2;
    float r = (float)((rgb>>16)&0xff) / 255.0f;
    float g = (float)((rgb>>8)&0xff) / 255.0f;
    float b = (float)(rgb&0xff) / 255.0f;
    int32_t ly = y;
    for (int32_t lx = x; *txt; lx = (*txt == '\n' ? x : lx+8), ly = (*txt == '\n' ? ly+14 : ly), txt++) {
        if (mag_unlikely(!isprint(*txt))) continue;
        for (int32_t yy = 0; yy < 14; ++yy) {
            for (int32_t xx = 0; xx < 8; ++xx) {
                if (!mag_glyph(*txt, xx, yy)) continue;
                int32_t px = lx + xx;
                int32_t py = ly + yy;
                if (mag_unlikely(px >= w || py >= h)) continue;
                int32_t ii = py*w + px;
                pr[ii] = r;
                pg[ii] = g;
                pb[ii] = b;
            }
        }
    }
}

mag_ctx_t* mag_tensor_get_ctx(const mag_tensor_t* t) { return t->ctx; }
void* mag_tensor_get_user_data(const mag_tensor_t* t) { return t->ud; }
void mag_tensor_set_user_data(mag_tensor_t* t, void* ud) { t->ud = ud; }

#ifdef __APPLE__
    static bool mag_sysctl_mib01(uint8_t (*out)[256], size_t* o_len, int mib0, int mib1) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        int name[2] = {mib0, mib1};
        size_t len = 0;
        if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), NULL, &len, NULL, 0))) return false; /* Get length */
        if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static bool mag_sysctl_key(uint8_t (*out)[256], size_t* o_len, const char* key) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        size_t len = 0;
        if (mag_unlikely(sysctlbyname(key, NULL, &len, NULL, 0))) return false; /* Get length */
        if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (mag_unlikely(sysctlbyname(key, *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static uint64_t mag_sysctl_unpack_int(const uint8_t (*in)[256], size_t len) { /* Unpack sysctl data */
        switch (len) {
            case sizeof(uint16_t): { uint16_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint32_t): { uint32_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint64_t): { uint64_t r; memcpy(&r, *in, sizeof(r)); return r; }
            default: return 0;
        }
    }
#else
    static bool mag_cpuinfo_parse_value(const char* key, char (*out)[128]) {
        FILE* cpuinfo = mag_fopen("/proc/cpuinfo", "rt");
        if (mag_unlikely(!cpuinfo)) return false;
        size_t key_len = strlen(key);
        char line[128];
        while (fgets(line, sizeof(line), cpuinfo)) {
            size_t line_len = strlen(line);
            if (line_len > 0 && line[line_len-1] == '\n') line[line_len-1] = '\0';
            if (strncmp(line, key, key_len) == 0 && (isspace((unsigned char)line[key_len]) || line[key_len] == ':')) {
                char* colon = strchr(line, ':');
                if (!colon) continue;
                char* value = colon+1;
                while (isspace((unsigned char)*value)) ++value;
                char* end = value + strlen(value);
                for (; end > value && isspace((unsigned char)*(end-1)); --end);
                *end = '\0';
                size_t value_len = llabs(end-value);
                if (mag_unlikely(!value_len || value_len >= sizeof(*out))) {
                    fclose(cpuinfo);
                    return false;
                }
                snprintf(*out, sizeof(*out), "%s", value);
                fclose(cpuinfo);
                return true;
            }
        }
        fclose(cpuinfo);
        return false;
    }
    static uint64_t mag_parse_meminfo_value(const char* line) {
        const char *p = strchr(line, ':');
        if (mag_unlikely(!p)) return 0;
        ++p;
        p += strspn(p, " \t");
        errno = 0;
        char* end;
        uint64_t value = strtoull(p, &end, 10);
        if (mag_unlikely(errno != 0 || p == end)) return 0;
        return value<<10;
    }
#endif

#ifdef __linux__
static void mag_trim_quotes(char* in) {
    if (in == NULL || *in == '\0') return;
    size_t len = strlen(in);
    if (in[len - 1] == '"') {
        in[len - 1] = '\0';
        len--;
    }
    if (in[0] == '"') {
        memmove(in, in + 1, len);
    }
}
#endif

static void MAG_COLDPROC mag_system_host_info_query_os_name(char (*out_os_name)[128]) { /* Get OS name */
    #ifdef _WIN32
        
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_KERN, KERN_VERSION) && len && *tmp)) {
            char* colon = strchr((const char*)tmp, ':');
            if (colon) *colon = '\0';
            snprintf(*out_os_name, sizeof(*out_os_name), "%s", (const char*)tmp);
        }
    #elif defined (__linux__)
        FILE* f = mag_fopen("/etc/os-release", "r");
        if (!f) {
            f = mag_fopen("/usr/lib/os-release", "r");
            if (!f) {
                f = mag_fopen("/etc/lsb-release", "r");
                if (mag_unlikely(!f)) return;
                char line[256];
                while (fgets(line, sizeof(line), f) != NULL) {
                    size_t len = strlen(line);
                    if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
                    if (strncmp(line, "DISTRIB_ID", sizeof("DISTRIB_ID")-1) == 0) {
                        char* equals_sign = strchr(line, '=');
                        if (equals_sign && *(equals_sign+1)) {
                            strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                            (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                        }
                    } else if (strncmp(line, "DISTRIB_DESCRIPTION", sizeof("DISTRIB_DESCRIPTION")-1) == 0) {
                        char* equals_sign = strchr(line, '=');
                        if (equals_sign && *(equals_sign+1)) {
                            char* start_quote = strchr(equals_sign+1, '"');
                            if (start_quote) {
                                char* end_quote = strchr(start_quote+1, '"');
                                if (end_quote) {
                                    size_t desc_len = end_quote-start_quote-1;
                                    if (desc_len >= sizeof(*out_os_name)) desc_len = sizeof(*out_os_name)-1;
                                    strncpy(*out_os_name, start_quote+1, desc_len);
                                    (*out_os_name)[desc_len] = '\0';
                                } else {
                                    strncpy(*out_os_name, start_quote+1, sizeof(*out_os_name)-1);
                                    (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                                }
                            } else {
                                strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                            }
                        }
                    }
                }
                fclose(f);
                return;
            }
        }
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (strncmp(line, "NAME", sizeof("NAME")-1) == 0) {
            char* equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign + 1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        } else if (strncmp(line, "PRETTY_NAME", sizeof("PRETTY_NAME")-1) == 0) {
            char* equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        }
    }
    fclose(f);
    mag_trim_quotes(*out_os_name);
    #endif
}

static void MAG_COLDPROC mag_system_host_info_query_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
    #ifdef _WIN32
        HKEY key;
        if (mag_unlikely(RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key))) return;
        char tmp[64+1] = {0};
        DWORD len = sizeof(tmp);
        if (mag_unlikely(RegQueryValueExA(key, "ProcessorNameString", NULL, NULL, (LPBYTE)tmp, &len))) return;
        if (mag_likely(strlen(tmp))) tmp[strlen(tmp)-1] = '\0';
        snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", tmp);
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.brand_string") && len && *tmp))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", (const char*)tmp);
    #else
        char cpu_name[128];
        if (mag_likely((mag_cpuinfo_parse_value("model name", &cpu_name) && *cpu_name) || (mag_cpuinfo_parse_value("Model", &cpu_name) && *cpu_name)))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", cpu_name);
    #endif
}

static void MAG_COLDPROC mag_system_host_info_query_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
    #ifdef _WIN32
        DWORD size = 0;
        GetLogicalProcessorInformation(NULL, &size);
        if (mag_unlikely(!size)) return;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = (*mag_alloc)(NULL, size);
        if (mag_unlikely(!GetLogicalProcessorInformation(info, &size))) goto end;
        for (DWORD i=0; i < size/sizeof(*info); ++i) {
            switch (info[i].Relationship) {
                default: continue;
                case RelationProcessorPackage: ++*out_sockets; continue;
                case RelationProcessorCore: {
                    ++*out_physical;
                    uintptr_t m = (uintptr_t)info[i].ProcessorMask;
                    m = m - ((m>>1) & 0x5555555555555555);
                    m = (m & 0x3333333333333333) + ((m>>2) & 0x3333333333333333);
                    *out_virtual += (((m + (m>>4)) & 0xf0f0f0f0f0f0f0f) * 0x101010101010101)>>56;
                } continue;
            }
        }
        end: (*mag_alloc)(info, 0);
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.thread_count") && len))
            *out_virtual = mag_sysctl_unpack_int(&tmp, len);
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.core_count") && len))
            *out_physical = mag_sysctl_unpack_int(&tmp, len);
        if (mag_likely(mag_sysctl_key(&tmp, &len, "hw.packages") && len))
            *out_sockets = mag_sysctl_unpack_int(&tmp, len);
    #else
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        FILE* cpuinfo = mag_fopen("/proc/cpuinfo", "r");
        if (mag_unlikely(!cpuinfo)) return;
        uint32_t physical_ids[MAG_MAX_CPUS];
        uint32_t core_ids[MAG_MAX_CPUS];
        uint32_t package_ids[MAG_MAX_CPUS];
        uint32_t cpu_count = 0;
        uint32_t package_count = 0;
        uint32_t current_physical_id = 0;
        uint32_t current_core_id = 0;
        bool got_physical_id = false;
        bool got_core_id = false;
        char line[256];
        while (fgets(line, sizeof(line), cpuinfo) != NULL) {
            if (strncmp(line, "physical id", sizeof("physical id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_physical_id = (uint32_t)strtoul(ptr, NULL, 10); got_physical_id = true; }
                }
            } else if (strncmp(line, "core id", sizeof("core id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_core_id = (uint32_t)strtoul(ptr, NULL, 10); got_core_id = true; }
                }
            } else if (*line == '\n') {
                if (got_physical_id && got_core_id) {
                    bool is_unique = true;
                    for (int32_t i = 0; i < cpu_count; ++i) if (physical_ids[i] == current_physical_id && core_ids[i] == current_core_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (cpu_count < MAG_MAX_CPUS) {
                            physical_ids[cpu_count] = current_physical_id;
                            core_ids[cpu_count] = current_core_id;
                            ++cpu_count;
                        } else break;
                    }
                    is_unique = true;
                    for (int32_t i = 0; i < package_count; ++i) if (package_ids[i] == current_physical_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (package_count < MAG_MAX_CPUS) package_ids[package_count++] = current_physical_id;
                        else break;
                    }
                }
                got_physical_id = false;
                got_core_id = false;
            }
        }
        fclose(cpuinfo);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : 0;
        if (!cpu_count && *out_virtual) cpu_count = *out_virtual;
        *out_physical = mag_xmax(1, cpu_count);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : *out_physical;
        *out_sockets = mag_xmax(1, package_count);
    #endif
}

static void MAG_COLDPROC mag_system_host_info_query_memory(uint64_t* out_phys_mem_total, uint64_t* out_phys_mem_free) { /* Get physical memory */
    #ifdef _WIN32
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (mag_likely(GlobalMemoryStatusEx(&mem))) {
            *out_phys_mem_total = mem.ullTotalPhys;
            *out_phys_mem_free = mem.ullAvailPhys;
        }
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_HW, HW_MEMSIZE) && len))
            *out_phys_mem_total = mag_sysctl_unpack_int(&tmp, len);
        struct vm_statistics64 stats;
        natural_t count = HOST_VM_INFO64_COUNT;
        if (mag_likely(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)(&stats), &count) == KERN_SUCCESS))
            *out_phys_mem_free = stats.free_count * getpagesize();
    #else
        FILE* meminfo = mag_fopen("/proc/meminfo", "r");
        if (mag_unlikely(!meminfo)) return;
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemTotal:", sizeof("MemTotal:")-1) == 0)
                *out_phys_mem_total = mag_parse_meminfo_value(line);
            else if (strncmp(line, "MemAvailable:", sizeof("MemAvailable:")-1) == 0)
                *out_phys_mem_free = mag_parse_meminfo_value(line);
        }
        fclose(meminfo);
    #endif
}

#if defined(__x86_64__) || defined(_M_X64)
    static void mag_cpuid(uint32_t leaf, int32_t sub, uint32_t* oeax, uint32_t* oebx, uint32_t* oecx, uint32_t* oedx) {
        #ifdef _MSC_VER
            int regs[4];
            if (sub != -1) __cpuidex(regs, leaf, sub);
            else __cpuid(regs, leaf);
            *oeax = regs[0], *oebx = regs[1], *oecx = regs[2], *oedx = regs[3];
        #else
            uint32_t eax, ebx, ecx, edx;
            if (sub != -1) __cpuid_count(leaf, sub, eax, ebx, ecx, edx);
            else __cpuid(leaf, eax, ebx, ecx, edx);
            *oeax = eax, *oebx = ebx, *oecx = ecx, *oedx = edx;
        #endif
    }
    static uint64_t MAG_AINLINE mag_xgetbv(void) { /* Query extended control register value. */
        #ifdef _MSC_VER
            return _xgetbv(0);
        #else
            uint32_t lo, hi;
            __asm__ __volatile__("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
            return (uint64_t)lo | ((uint64_t)hi << 32);
        #endif
    }
    #define mag_cpy_regs(id) \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EAX] = eax; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EBX] = ebx; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_ECX] = ecx; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EDX] = edx
    static void MAG_COLDPROC mag_system_info_query_x86_64_cpu_features(uint32_t (*features)[8][4]) {
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        uint32_t max_basic_leaf, max_extended_leaf;
        mag_cpuid(0, -1, &eax, &ebx, &ecx, &edx);
        mag_cpy_regs(0H);
        max_basic_leaf = eax;
        mag_cpuid(0x80000000u, -1, &eax, &ebx, &ecx, &edx);
        max_extended_leaf = eax;
        if (max_basic_leaf >= 1u) {
            mag_cpuid(1, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(1H);
        }
        if (max_basic_leaf >= 2u) {
            mag_cpuid(2u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(2H);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 0, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(7H);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(7H_1H);
        }
        if (max_basic_leaf >= 0x16u) {
            mag_cpuid(0x16u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(16H);
        }
        if (max_extended_leaf >= 0x80000001u) {
            mag_cpuid(0x80000001u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(80000001H);
        }
        if (max_extended_leaf >= 0x80000007u) {
            mag_cpuid(0x80000007u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(80000007H);
        }
        bool cpu_avx_support = ((*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] & 0x10000000u) != 0;
        bool cpu_osxsave_support = ((*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] & 0x8000000u) != 0;
        if (cpu_avx_support && cpu_osxsave_support) {
            uint64_t xcr0 = mag_xgetbv();
            if ((xcr0 & 0x6) != 0x6u) {
                (*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            }
            if ((xcr0 & 0xe0) != 0xe0u) { /* OS does not support AVX-512, clear AVX512 */
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0xdc230000u;
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_ECX] &= ~0x5842u;
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EDX] &= ~0x10cu;
                (*features)[MAG_X86_64_CPUID_7H_1H][MAG_X86_64_CPUID_EAX] &= ~0x20u;
            }
        } else {
            (*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0xdc230000u; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_ECX] &= ~0x5842u; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EDX] &= ~0x10cu; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H_1H][MAG_X86_64_CPUID_EAX] &= ~0x20u; /* Clear AVX512 */
        }
    }
    #undef mag_cpy_regs
#endif

static void MAG_COLDPROC mag_system_host_info_query(mag_ctx_t* ctx) {
    mag_system_host_info_query_os_name(&ctx->sys.os_name);
    mag_system_host_info_query_cpu_name(&ctx->sys.cpu_name);
    mag_system_host_info_query_cpu_cores(&ctx->sys.cpu_virtual_cores, &ctx->sys.cpu_physical_cores, &ctx->sys.cpu_sockets);
    mag_system_host_info_query_memory(&ctx->sys.phys_mem_total, &ctx->sys.phys_mem_free);
    #if defined(__x86_64__) || defined(_M_X64)
        mag_system_info_query_x86_64_cpu_features(&ctx->sys.x86_64_cpu_features);
    #endif
    if (mag_unlikely(!*ctx->sys.os_name)) snprintf(ctx->sys.os_name, sizeof(ctx->sys.os_name), "Unknown");
    if (mag_unlikely(!*ctx->sys.cpu_name)) snprintf(ctx->sys.cpu_name, sizeof(ctx->sys.cpu_name), "Unknown");
}

static MAG_AINLINE void mag_sto_write_u32_le(uint8_t** p, uint32_t x) {
    x = mag_bswap32(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static MAG_AINLINE void mag_sto_write_u64_le(uint8_t** p, uint64_t x) {
    x = mag_bswap64(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static MAG_AINLINE uint32_t mag_sto_read_u32_le(const uint8_t** p) {
    uint32_t x;
    memcpy(&x, *p, sizeof(x));
    x = mag_bswap32(x);
    *p += sizeof(x);
    return x;
}

static MAG_AINLINE uint64_t mag_sto_read_u64_le(const uint8_t** p) {
    uint64_t x;
    memcpy(&x, *p, sizeof(x));
    x = mag_bswap64(x);
    *p += sizeof(x);
    return x;
}

#define MAG_STO_MAGIC "magtron!"
mag_static_assert(sizeof(MAG_STO_MAGIC)-1 == sizeof(uint64_t));
#define MAG_STO_FILE_HEADER_SIZE ((sizeof(MAG_STO_MAGIC)-1) + sizeof(uint32_t)*3)
#define MAG_STO_TENSOR_HEADER_SIZE (MAG_MAX_TENSOR_NAME_LEN + sizeof(uint32_t) + sizeof(int64_t)*MAG_MAX_DIMS)
mag_static_assert(MAG_MAX_TENSOR_NAME_LEN % 8 == 0);
mag_static_assert(MAG_DTYPE__NUM <= 0xff);
mag_static_assert(MAG_MAX_DIMS <= 0xff);
#define mag_sto_sanitize(exp, ret) do { if (mag_unlikely(!(exp))) { mag_log_error("magnetron storage sanitize error: " #exp); return (ret); } } while (0)

static bool mag_sto_write_file_header( /* Write file header -  file header must be same in every version. */
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    uint32_t num_tensors,
    uint32_t ud
) {
    mag_sto_sanitize(*p + MAG_STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t mag_magic;
    memcpy(&mag_magic, MAG_STO_MAGIC, sizeof(mag_magic));
    mag_sto_write_u64_le(p, mag_magic);
    mag_sto_write_u32_le(p, version);
    mag_sto_write_u32_le(p, num_tensors);
    mag_sto_write_u32_le(p, ud);
    return mag_likely(*p - start == MAG_STO_FILE_HEADER_SIZE);
}

static bool mag_sto_read_file_header(  /* Read file header - file header must be same in every version. */
    const uint8_t** p,
    const uint8_t* end,
    uint32_t* version,
    uint32_t* num_tensors,
    uint32_t* ud
) {
    mag_sto_sanitize(*p + MAG_STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t mag_magic = mag_sto_read_u64_le(p);
    mag_sto_sanitize(memcmp(&mag_magic, MAG_STO_MAGIC, sizeof(mag_magic)) == 0, false);
    *version = mag_sto_read_u32_le(p);
    *num_tensors = mag_sto_read_u32_le(p);
    *ud = mag_sto_read_u32_le(p);
    return mag_likely(*p - start == MAG_STO_FILE_HEADER_SIZE);
}

static bool mag_sto_write_tensor_header(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    const char (*name)[MAG_MAX_TENSOR_NAME_LEN],
    mag_tensor_flags_t flags,
    mag_dtype_t dtype,
    int64_t rank,
    const int64_t (*shape)[MAG_MAX_DIMS]
) {
    mag_sto_sanitize(*p + MAG_STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            memcpy(name_u64, *name, sizeof(*name));
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i)   /* Write name as multiple u64 */
                mag_sto_write_u64_le(p, name_u64[i]);
            uint32_t aux = 0;   /* Pack small fields into aux field */
            aux |= (flags & 0xff) << 16;
            aux |= (dtype & 0xff) << 8;
            aux |= (rank & 0xff);
            mag_sto_write_u32_le(p, aux);     /* Write aux field */
            for (size_t i=0; i < MAG_MAX_DIMS; ++i) {      /* Write shape */
                mag_sto_sanitize((*shape)[i] >= 1 && (*shape)[i] < INT64_MAX, false);
                mag_sto_write_u64_le(p, (uint64_t)(*shape)[i]);
            }
        } break;
        default: return false;
    }
    return mag_likely(*p - start == MAG_STO_TENSOR_HEADER_SIZE);
}

static bool mag_sto_read_tensor_header(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    char (*name)[MAG_MAX_TENSOR_NAME_LEN],
    mag_tensor_flags_t* flags,
    mag_dtype_t* dtype,
    int64_t* rank,
    int64_t (*shape)[MAG_MAX_DIMS]
) {
    mag_sto_sanitize(*p + MAG_STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i) /* Read name as multiple u64 */
                name_u64[i] = mag_sto_read_u64_le(p);
            memcpy(name, name_u64, sizeof(*name));
            (*name)[sizeof(*name)-1] = '\0';
            uint32_t aux = mag_sto_read_u32_le(p); /* Read aux field */
            *flags = (mag_tensor_flags_t)((aux >> 16) & 0xff);
            *dtype = (mag_dtype_t)((aux >> 8) & 0xff);
            *rank = (int64_t)(aux & 0xff);
            mag_sto_sanitize((*flags & ~((1u<<MAG_TFLAG_LEN)-1)) == 0, false); /* Check fields */
            mag_sto_sanitize(*dtype >= 0 && *dtype < MAG_DTYPE__NUM, false);
            mag_sto_sanitize(*rank >= 1 && *rank <= MAG_MAX_DIMS, false);
            for (size_t i=0; i < MAG_MAX_DIMS; ++i) {  /* Read shape */
                uint64_t u64 = mag_sto_read_u64_le(p);
                mag_sto_sanitize(u64>= 1 && u64 <= (uint64_t)INT64_MAX, false);
                (*shape)[i] = (int64_t)u64;
            }
            mag_sto_sanitize(INT64_MAX/(*shape)[1] > (*shape)[0], false); /* Check for shape overflow */
            mag_sto_sanitize(INT64_MAX/(*shape)[2] > (*shape)[0]*(*shape)[1], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[3] > (*shape)[0]*(*shape)[1]*(*shape)[2], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[4] > (*shape)[0]*(*shape)[1]*(*shape)[2]*(*shape)[3], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[5] > (*shape)[0]*(*shape)[1]*(*shape)[2]*(*shape)[3]*(*shape)[4], false);
        } break;
        default: return false;
    }
    return mag_likely(*p - start == MAG_STO_TENSOR_HEADER_SIZE);
}

static bool mag_sto_write_tensor_data(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    mag_dtype_t dtype,
    const void* data,
    int64_t size
) {
    mag_sto_sanitize(size > 0, false);
    mag_sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(*p, data, size);
            *p += size;
        } break;
        default: return false;
    }
    return mag_likely(*p - start == size);
}

static bool mag_sto_read_tensor_data(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    mag_dtype_t dtype,
    void* data,
    int64_t size
) {
    mag_sto_sanitize(size > 0, false);
    mag_sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(data, *p, size); /* TODO: endianess conversion */
            *p += size;
        } break;
        default: return false;
    }
    return mag_likely(*p - start == size);
}

static size_t mag_accumulate_data_size(mag_dtype_t dtype, const int64_t (*shape)[MAG_MAX_DIMS]) {
    size_t size = mag_dtype_meta_of(dtype)->size;
    #pragma GCC unroll 6
    for (size_t i=0; i < MAG_MAX_DIMS; ++i) size *= (size_t)mag_xmax(1, (*shape)[i]);
    return size;
}

static size_t mag_sto_total_size(const mag_tensor_t** tensors, size_t n) {
    size_t total = MAG_STO_FILE_HEADER_SIZE;
    for (size_t i=0; i < n; ++i) {
        total += MAG_STO_TENSOR_HEADER_SIZE;
        total += mag_accumulate_data_size(tensors[i]->dtype, &tensors[i]->shape);
    }
    mag_assert((total & 3) == 0, "Unaligned storage size: %zu", total);
    return total;
}

static uint8_t* mag_sto_write_buffered(const mag_tensor_t** tensors, size_t n_tensors, size_t* out_size, uint32_t version) {
    if (mag_unlikely(!tensors || !n_tensors || n_tensors > UINT32_MAX || !out_size || !version || version > MAG_STORAGE_VERSION)) return NULL;  /* Check input */
    *out_size = mag_sto_total_size(tensors, n_tensors);
    uint8_t* base = (uint8_t*)(*mag_alloc)(NULL, *out_size );     /* Allocate buffer */
    uint8_t* needle = base;
    const uint8_t* end = base + *out_size ;
    if (mag_unlikely(!mag_sto_write_file_header(&needle, end, version, (uint32_t)n_tensors, 0))) goto error;     /* Write file header */
    for (size_t i=0; i < n_tensors; ++i) {   /* Write tensor headers */
        const mag_tensor_t* t = tensors[i];
        mag_assert2(t != NULL);
        if (mag_unlikely(!mag_sto_write_tensor_header(
            &needle,
            end,
            version,
            &t->name,
            t->flags,
            t->dtype,
            t->rank,
            &t->shape
        ))) goto error;
    }
    mag_assert2(needle - base == MAG_STO_FILE_HEADER_SIZE + n_tensors*MAG_STO_TENSOR_HEADER_SIZE);    /* Check written data size */
    for (size_t i=0; i < n_tensors; ++i) {  /* Write tensor data */
        const mag_tensor_t* t = tensors[i];
        mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
        if (mag_unlikely(!mag_sto_write_tensor_data(&needle, end, version, t->dtype, (const void*)t->storage.base, mag_tensor_data_size(t)))) goto error;     /* Write data */
    }
    return base;
    error: /* Error handling */
        (*mag_alloc)(base, 0);
        return NULL;
}

MAG_EXPORT mag_tensor_t** mag_sto_read_buffered(mag_ctx_t* ctx, const uint8_t* buf, size_t size, uint32_t* out_n_tensors, uint32_t* out_version) { /* Load stored tensors from buffer. Function is exported for fuzzing test. */
    if (mag_unlikely(!ctx || !buf || !out_n_tensors || !out_version || size <= MAG_STO_FILE_HEADER_SIZE + MAG_STO_TENSOR_HEADER_SIZE + 1)) return NULL;    /* Check input */
    const uint8_t* needle = buf;
    const uint8_t* end = buf + size;
    uint32_t n_tensors;
    uint32_t ud;
    if (mag_unlikely(!mag_sto_read_file_header(&needle, end, out_version, &n_tensors, &ud))) return NULL;   /* Read file header */
    if (mag_unlikely(!*out_version || *out_version > MAG_VERSION)) return NULL;
    if (mag_unlikely(!n_tensors)) return NULL;
    mag_tensor_t** tensors = (mag_tensor_t**)(*mag_alloc)(NULL, n_tensors*sizeof(*tensors));   /* Allocate return tensor array */
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor headers */
        char name[MAG_MAX_TENSOR_NAME_LEN] = {0};
        mag_tensor_flags_t flags = 0;
        mag_dtype_t dtype = 0;
        int64_t rank = 0;
        int64_t shape[MAG_MAX_DIMS] = {0};
        if (mag_unlikely(!mag_sto_read_tensor_header(&needle, end, *out_version, &name, &flags, &dtype, &rank, &shape))) goto error;   /* Read tensor header */
        mag_tensor_t* t = mag_tensor_create(ctx, dtype, shape, rank, NULL, 0);   /* Create placeholder tensor */
        mag_tensor_fmt_name(t, "%s", name);
        t->flags = flags;
        tensors[i] = t;
    }
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor data */
        mag_tensor_t* t = tensors[i];
        mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
        size_t data_size = mag_accumulate_data_size(t->dtype, &t->shape);
        mag_assert2(needle + data_size <= end && data_size == mag_tensor_data_size(t));
        if (mag_unlikely(!mag_sto_read_tensor_data(&needle, end, *out_version, t->dtype, (void*)t->storage.base, data_size))) goto error;  /* Read data into tensor's buffer */
    }
    *out_n_tensors = n_tensors;
    return tensors;
    error:
        (*mag_alloc)(tensors, 0);
        return NULL;
}

static bool mag_sto_has_mag_ext(const char* file) { /* Check if file path has magnetron extension. */
    if (mag_unlikely(!file || strlen(file) < sizeof(MAG_STORAGE_EXT))) return false;
    const char* dot = strrchr(file, '.');
    return dot && !strcmp(dot, MAG_STORAGE_EXT);
}

void mag_tensor_save(const mag_tensor_t* t, const char* file) {
    mag_assert(mag_sto_has_mag_ext(file), "Invalid file extension: %s", file);
    FILE* f = mag_fopen(file, "wb");  /* Open file */
    mag_assert(f, "Failed to open file stream: %s", file);
    uint32_t version = MAG_STORAGE_VERSION;
    size_t n_tensors = 1;
    size_t n_bytes = 0;
    uint8_t* ser = mag_sto_write_buffered(&t, n_tensors, &n_bytes, version);   /* Serialize tensor */
    mag_assert(ser && n_bytes, "Failed to serialize tensor to file: %s", file);   /* Check serialization */
    mag_assert(fwrite(ser, 1, n_bytes, f) == n_bytes, "Failed to write %zu bytes to file: %s", n_bytes, file);    /* Write to file */
    (*mag_alloc)(ser, 0);     /* Free buffer */
    fflush(f);
    fclose(f);
    double mem;
    const char* unit;
    mag_humanize_memory_size(n_bytes, &mem, &unit);
    mag_log_info("Saved %zu tensor%s to file: %s, %.03f %s written, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
}

mag_tensor_t* mag_tensor_load(mag_ctx_t* ctx, const char* file) {
    mag_assert(mag_sto_has_mag_ext(file), "Invalid file extension: %s", file);
    FILE* f = mag_fopen(file, "rb");  /* Open file */
    mag_assert(f, "Failed to open file stream: %s", file);
    mag_assert2(fseek(f, 0, SEEK_END) == 0);  /* Seek to end */
    long n_bytes = ftell(f);    /* Get file size */
    mag_assert(n_bytes > MAG_STO_FILE_HEADER_SIZE + MAG_STO_TENSOR_HEADER_SIZE + 1, "Malformed file size");   /* Check file size */
    mag_assert2(fseek(f, 0, SEEK_SET) == 0); /* Seek to start */
    uint8_t* buf = (uint8_t*)(*mag_alloc)(NULL, n_bytes);  /* Allocate buffer */
    mag_assert(fread(buf, 1, n_bytes, f) == n_bytes, "Failed to read %zu bytes from file: %s", n_bytes, file);    /* Read while file into buffer */
    fclose(f), f = NULL;    /* Close file */
    uint32_t n_tensors = 0, version = 0;
    mag_tensor_t** tensors = mag_sto_read_buffered(ctx, buf, n_bytes, &n_tensors, &version);   /* Deserialize tensors */
    mag_assert(version > 0 && version <= MAG_VERSION, "Unsupported storage version: %u", version);   /* Check version */
    mag_assert(tensors && n_tensors > 0, "Failed to load tensor from file: %s", file);
    mag_tensor_t* target = *tensors;
    (*mag_alloc)(buf, 0);     /* Free buffer */
    (*mag_alloc)(tensors, 0);     /* Free tensor array */
    double mem;
    const char* unit;
    mag_humanize_memory_size(n_bytes, &mem, &unit);
    mag_log_info("Loaded %u tensor%s from file: %s, %.03f %s read, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
    return target;
}

mag_tensor_t* mag_tensor_load_image(mag_ctx_t* ctx, const char* file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h) {
    uint8_t* (*loader)(const char*, uint32_t(*)[3], mag_color_channels_t) = ctx->image_load_fn;
    void (*load_free)(uint8_t*) = ctx->image_load_free_fn;
    mag_assert(loader && load_free, "Image loader not set");
    uint32_t whc[3] = {0};
    uint8_t* src = (*loader)(file, &whc, channels);
    mag_assert(src, "Failed to load tensor from image: '%s'", file);
    if (resize_w && resize_h) { /* Resize requested. */
        float* ori = (*mag_alloc)(NULL, whc[2]*whc[1]*whc[0]*sizeof(*ori));
        for (int64_t k=0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j=0; j < whc[1]; ++j) {
                for (int64_t i=0; i < whc[0]; ++i) {
                    ori[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        mag_tensor_t* t = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, whc[2], resize_h, resize_w);
        float* dst = mag_tensor_data_ptr(t);
        float* part = (*mag_alloc)(NULL, whc[2] * whc[1] * resize_w * sizeof(*part));
        float ws = (float)(whc[0] - 1)/(float)(resize_w - 1);
        float hs = (float)(whc[1] - 1)/(float)(resize_h - 1);
        for (uint32_t k = 0; k < whc[2]; ++k){
            for (uint32_t r = 0; r < whc[1]; ++r) {
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = 0;
                    if (c == resize_w - 1 || whc[0] == 1) {
                        val = ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (whc[0] - 1)];
                    } else {
                        float sx = (float)c*ws;
                        uint32_t ix = (uint32_t)sx;
                        float dx = sx - (float)ix;
                        val = (1-dx) * (ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + ix]) + dx*(ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (ix + 1)]);
                    }
                    part[k * resize_w * (whc[1]) + r * resize_w + c] = val;
                }
            }
        }
        for (uint32_t k = 0; k < whc[2]; ++k) {
            for (uint32_t r = 0; r < resize_h; ++r) {
                float sy = (float)r*hs;
                uint32_t iy = (uint32_t)sy;
                float dy = sy - (float)iy;
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = (1-dy)*(part[k * resize_w * whc[1] + iy * resize_w + c]);
                    dst[k * resize_w * resize_h + r * resize_w + c] = val;
                }
                if (r == resize_h - 1 || whc[1] == 1) continue;
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = dy*(part[k * resize_w * (whc[1]) + (iy + 1) * resize_w + c]);
                    dst[k * resize_w * resize_h + r * resize_w + c] += val;
                }
            }
        }
        (*mag_alloc)(ori, 0);
        (*mag_alloc)(part, 0);
        mag_assert(resize_w * resize_h * whc[2] == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", resize_w * resize_h * whc[2], (size_t)mag_tensor_numel(t));
        (*load_free)(src);
        mag_log_info("Loaded and resized tensor from image: %s, %u x %u x %u", file, resize_w, resize_h, whc[2]);
        return t;
    } else {
        mag_tensor_t* t = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, whc[2], whc[1], whc[0]);
        float* dst = mag_tensor_data_ptr(t);
        for (int64_t k = 0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j = 0; j < whc[1]; ++j) {
                for (int64_t i = 0; i < whc[0]; ++i) {
                    dst[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        mag_assert(whc[0]*whc[1]*whc[2] == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", whc[0]*whc[1]*whc[2], (size_t)mag_tensor_numel(t));
        (*load_free)(src);
        mag_log_info("Loaded tensor from image: %s, %u x %u x %u", file, whc[0], whc[1], whc[2]);
        return t;
    }
}

void mag_tensor_save_image(const mag_tensor_t* t, const char* file) {
    bool (*saver)(const char*, const uint8_t*, const uint32_t(*)[3]) = t->ctx->image_save_fn;
    mag_assert(saver, "Image saver not set");
    int64_t rank = mag_tensor_rank(t);
    mag_assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
    int64_t w = mag_tensor_image_width(t);
    int64_t h = mag_tensor_image_height(t);
    int64_t c = mag_tensor_image_channels(t);
    mag_assert(c == 1 || c == 3 || c == 4, "Invalid number of channels: %zu", (size_t)c);
    mag_assert(w*h*c == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", w*h*c, (size_t)mag_tensor_numel(t));
    uint8_t* dst = (*mag_alloc)(NULL, w*h*c); /* Allocate memory for image data */
    const float* src = mag_tensor_data_ptr(t);
    for (int64_t k = 0; k < c; ++k) /* Convert from planar to interleaved format. */
        for (int64_t i = 0; i < w*h; ++i)
            dst[i*c + k] = (uint8_t)(src[i + k*w*h]*255.0f);
    const uint32_t whc[3] = {(uint32_t)w,(uint32_t)h,(uint32_t)c};
    mag_assert((*saver)(file, dst, &whc), "Failed to save tensor to image: %s", file);
    (*mag_alloc)(dst, 0); /* Free image data */
    mag_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file, (int)w, (int)h, (int)c);
}

/* Rescale factors to push the exponent of a number towards zero. */
#define rescale_exponents(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)
#define one_e_p(X) 1e+0 ## X
#define one_e_n(X) 1e-0 ## X
static const int16_t mag_rescale_e[] = { rescale_exponents(-, +) };
static const double mag_rescale_n[] = { rescale_exponents(one_e_p, one_e_n) };
#undef one_e_n
#undef one_e_p

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t mag_four_ulp_m_e[] = {
    34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19,
    -82, -18, 35, -18, 7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16,
    45, -16, 89, -16, -78, -15, 36, -15, 72, -15, -113, -14, 29, -14, 57, -14,
    114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12, 73, -12, 15, -11, 3,
    -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15, -7,
    3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103,
    -5, 31, -5, 62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3,
    16, -2, 32, -2, 63, -2, 2, -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2,
    32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3, -51, 3, 41, 4, 82, 4, -92, 4,
    33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6, 17, 7, 34,
    7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10,
    69, 10, -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13,
    36, 13, 71, 13, -115, 14, 29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15,
    19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17
};

/* min(2^32-1, 10^e-1) for e in range 0 through 10 */
static const uint32_t mag_ndigits_dec_threshold[] = {
    0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
    9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* Compute the number of digits in the decimal representation of x. */
static size_t mag_ndigits_dec(uint32_t x) {
    size_t t = ((mag_fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
    return t + (x > mag_ndigits_dec_threshold[t]);
}

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* mag_wuint9(char* p, uint32_t u) {
    uint32_t v = u / 10000, w;
    u -= v * 10000;
    w = v / 10000;
    v -= w * 10000;
    *p++ = (char)('0'+w);
    wint_r(v, 23, 1000)
    wint_r(v, 12, 100)
    wint_r(v, 10, 10)
    *p++ = (char)('0'+v);
    wint_r(u, 23, 1000)
    wint_r(u, 12, 100)
    wint_r(u, 10, 10)
    *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* mag_wint(char* p, int32_t k) {
    uint32_t u = (uint32_t)k;
    if (k < 0) { u = ~u+1u; *p++ = '-'; }
    if (u < 10000) {
        if (u < 10) goto dig1;
        if (u < 100) goto dig2;
        if (u < 1000) goto dig3;
    } else {
        uint32_t v = u / 10000; u -= v * 10000;
        if (v < 10000) {
            if (v < 10) goto dig5;
            if (v < 100) goto dig6;
            if (v < 1000) goto dig7;
        } else {
            uint32_t w = v / 10000; v -= w * 10000;
            if (w >= 10) wint_r(w, 10, 10)
            *p++ = (char)('0'+w);
        }
        wint_r(v, 23, 1000)
        dig7: wint_r(v, 12, 100)
        dig6: wint_r(v, 10, 10)
        dig5: *p++ = (char)('0'+v);
    }
    wint_r(u, 23, 1000)
    dig3: wint_r(u, 12, 100)
    dig2: wint_r(u, 10, 10)
    dig1: *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

/* -- Extended precision arithmetic --------------------------------------- */

/*
** The "nd" format is a fixed-precision decimal representation for numbers. It
** consists of up to 64 uint32_t values, with each uint32_t storing a value
** in the range [0, 1e9). A number in "nd" format consists of three variables:
**
**  uint32_t nd[64];
**  uint32_t ndlo;
**  uint32_t ndhi;
**
** The integral part of the number is stored in nd[0 ... ndhi], the value of
** which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
** the number is zero, ndlo is zero. Otherwise, the fractional part is stored
** in nd[ndlo ... 63], the value of which is taken to be
** sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
**
** If the array part had 128 elements rather than 64, then every double would
** have an exact representation in "nd" format. With 64 elements, all integral
** doubles have an exact representation, and all non-integral doubles have
** enough digits to make both %.99e and %.99f do the right thing.
*/
#define MAG_ND_MUL2K_MAX_SHIFT 29
#define MAG_ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, mag_format_flags sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > MAG_ND_MUL2K_MAX_SHIFT*2 && MAG_FMT_FP(sf) != MAG_FMT_FP(MAG_FMT_T_FP_F)) {
        start = ndhi - (MAG_FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= MAG_ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << MAG_ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = MAG_ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= MAG_ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = MAG_ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) nd[++ndhi] = carry_in;
    }
    return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t* nd, uint32_t ndhi, uint32_t k, mag_format_flags sf) {
    uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
    /* Performance hacks. */
    if (!ndhi) {
        if (!nd[0]) {
            return 0;
        } else {
            uint32_t s = mag_ffs(nd[0]);
            if (s >= k) { nd[0] >>= k; return 0; }
            nd[0] >>= s; k -= s;
        }
    }
    if (k > 18) {
        if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_F)) {
            stop1 = 63 - (int32_t)MAG_FMT_PREC(sf) / 9;
        } else {
            int32_t floorlog2 = ndhi * 29 + mag_fls(nd[ndhi]) - k;
            int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
            stop1 = 62 + (floorlog10 - (int32_t)MAG_FMT_PREC(sf)) / 9;
            stop2 = 61 + ndhi - (int32_t)MAG_FMT_PREC(sf) / 8;
        }
    }
    /* Real logic. */
    while (k >= 9) {
        uint32_t i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> 9) + carry;
            carry = (val & 0x1ff) * 1953125;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (ndlo != stop1 && ndlo != stop2) {
            if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
            if (!nd[ndhi]) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
        } else if (!nd[ndhi]) {
            if (ndhi != ndlo) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
            else return ndlo;
        }
        k -= 9;
    }
    if (k) {
        uint32_t mask = (1U << k) - 1, mul = 1000000000 >> k, i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> k) + carry;
            carry = (val & mask) * mul;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
    }
    return ndlo;
}

/* Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9). */
static uint32_t nd_add_m10e(uint32_t* nd, uint32_t ndhi, uint8_t m, int32_t e) {
    uint32_t i, carry;
    if (e >= 0) {
        i = (uint32_t)e/9;
        carry = m * (mag_ndigits_dec_threshold[e - (int32_t)i*9] + 1);
    } else {
        int32_t f = (e-8)/9;
        i = (uint32_t)(64 + f);
        carry = m * (mag_ndigits_dec_threshold[e - f*9] + 1);
    }
    for (;;) {
        uint32_t val = nd[i] + carry;
        if (mag_unlikely(val >= 1000000000)) {
            val -= 1000000000;
            nd[i] = val;
            if (mag_unlikely(i == ndhi)) {
                ndhi = (ndhi + 1) & 0x3f;
                nd[ndhi] = 1;
                break;
            }
            carry = 1;
            i = (i + 1) & 0x3f;
        } else {
            nd[i] = val;
            break;
        }
    }
    return ndhi;
}

static bool nd_similar(uint32_t* nd, uint32_t ndhi, uint32_t* ref, size_t hilen, size_t prec) {
    char nd9[9], ref9[9];
    if (hilen <= prec) {
        if (mag_unlikely(nd[ndhi] != *ref)) return 0;
        prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
        if (prec >= 9) {
            if (mag_unlikely(nd[ndhi] != *ref)) return 0;
            prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
        }
    } else {
        prec -= hilen - 9;
    }
    mag_assert(prec < 9, "bad precision %d", prec);
    mag_wuint9(nd9, nd[ndhi]);
    mag_wuint9(ref9, *ref);
    return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

/* Format f64 according to format flags. */
static char* mag_fmt_f64(mag_format_flags sf, double n, char* p) {
    size_t width = MAG_FMT_WIDTH(sf), prec = MAG_FMT_PREC(sf), len;
    union {
        uint64_t u64;
        double n;
        struct { /* TODO: make endian aware */
            uint32_t lo, hi;
        } u32;
    } t = {.n = n};
    if (mag_unlikely((t.u32.hi << 1) >= 0xffe00000)) {
        /* Handle non-finite values uniformly for %a, %e, %f, %g. */
        int32_t prefix = 0, ch = (sf & MAG_FMT_F_UPPER) ? 0x202020 : 0;
        if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
            ch ^= ('n' << 16) | ('a' << 8) | 'n';
            if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        } else {
            ch ^= ('i' << 16) | ('n' << 8) | 'f';
            if ((t.u32.hi & 0x80000000)) prefix = '-';
            else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
            else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        }
        len = 3 + (prefix != 0);
        if (!(sf & MAG_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
        if (prefix) *p++ = prefix;
        *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
    } else if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_A)) {
        /* %a */
        const char* hexdig = (sf & MAG_FMT_F_UPPER) ? "0123456789ABCDEFPX" : "0123456789abcdefpx";
        int32_t e = (t.u32.hi >> 20) & 0x7ff;
        char prefix = 0, eprefix = '+';
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        t.u32.hi &= 0xfffff;
        if (e) {
            t.u32.hi |= 0x100000;
            e -= 1023;
        } else if (t.u32.lo | t.u32.hi) {
            /* Non-zero denormal - normalise it. */
            uint32_t shift = t.u32.hi ? 20-mag_fls(t.u32.hi) : 52-mag_fls(t.u32.lo);
            e = -1022 - shift;
            t.u64 <<= shift;
        }
        /* abs(n) == t.u64 * 2^(e - 52) */
        /* If n != 0, bit 52 of t.u64 is set, and is the highest set bit. */
        if ((int32_t)prec < 0) {
            /* Default precision: use smallest precision giving exact result. */
            prec = t.u32.lo ? 13-mag_ffs(t.u32.lo)/4 : 5-mag_ffs(t.u32.hi|0x100000)/4;
        } else if (prec < 13) {
            /* Precision is sufficiently low as to maybe require rounding. */
            t.u64 += (((uint64_t)1) << (51 - prec*4));
        }
        if (e < 0) {
            eprefix = '-';
            e = -e;
        }
        len = 5 + mag_ndigits_dec((uint32_t)e) + prec + (prefix != 0)
              + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
        if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
            while (width-- > len) *p++ = ' ';
        }
        if (prefix) *p++ = prefix;
        *p++ = '0';
        *p++ = hexdig[17]; /* x or X */
        if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
            while (width-- > len) *p++ = '0';
        }
        *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
        if ((prec | (sf & MAG_FMT_F_ALT))) {
            /* Emit fractional part. */
            char* q = p + 1 + prec;
            *p = '.';
            if (prec < 13) t.u64 >>= (52 - prec*4);
            else while (prec > 13) p[prec--] = '0';
            while (prec) { p[prec--] = hexdig[t.u64 & 15]; t.u64 >>= 4; }
            p = q;
        }
        *p++ = hexdig[16]; /* p or P */
        *p++ = eprefix; /* + or - */
        p = mag_wint(p, e);
    } else {
        /* %e or %f or %g - begin by converting n to "nd" format. */
        uint32_t nd[64];
        uint32_t ndhi = 0, ndlo, i;
        int32_t e = (int32_t)(t.u32.hi >> 20) & 0x7ff, ndebias = 0;
        char prefix = 0, *q;
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
        if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_G)) {
            /* %g - decrement precision if non-zero (to make it like %e). */
            prec--;
            prec ^= (uint32_t)((int32_t)prec >> 31);
        }
        if ((sf & MAG_FMT_T_FP_E) && prec < 14 && n != 0) {
            /* Precision is sufficiently low that rescaling will probably work. */
            if ((ndebias = mag_rescale_e[e >> 6])) {
                t.n = n * mag_rescale_n[e >> 6];
                if (mag_unlikely(!e)) t.n *= 1e10, ndebias -= 10;
                t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
                nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
                e = ((int32_t)(t.u32.hi >> 20) & 0x7ff) - 1075 - (MAG_ND_MUL2K_MAX_SHIFT < 29);
                goto load_t_lo; rescale_failed:
                t.n = n;
                e = (int32_t)(t.u32.hi >> 20) & 0x7ff;
                ndebias = 0;
                ndhi = 0;
            }
        }
        nd[0] = t.u32.hi & 0xfffff;
        if (e == 0) e++; else nd[0] |= 0x100000;
        e -= 1043;
        if (t.u32.lo) {
            e -= 32 + (MAG_ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
#if MAG_ND_MUL2K_MAX_SHIFT >= 29
            nd[0] = (nd[0]<<3) | (t.u32.lo>>29);
            ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
#elif MAG_ND_MUL2K_MAX_SHIFT >= 11
            ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo>>21, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo>>10) & 0x7ff, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo<<1) & 0x7ff, sf);
#else
#error "MAG_ND_MUL2K_MAX_SHIFT not big enough"
#endif
        }
        if (e >= 0) {
            ndhi = nd_mul2k(nd, ndhi, (uint32_t)e, 0, sf);
            ndlo = 0;
        } else {
            ndlo = nd_div2k(nd, ndhi, (uint32_t)-e, sf);
            if (ndhi && !nd[ndhi]) ndhi--;
        }
        /* |n| == nd * 10^ndebias (for slightly loose interpretation of ==) */
        if ((sf & MAG_FMT_T_FP_E)) {
            /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
            char eprefix = '+';
            int32_t nde = -1;
            size_t hilen;
            if (ndlo && !nd[ndhi]) {
                ndhi = 64; do {} while (!nd[--ndhi]);
                nde -= 64 * 9;
            }
            hilen = mag_ndigits_dec(nd[ndhi]);
            nde += (int32_t)(ndhi * 9 + hilen);
            if (ndebias) {
                /*
                ** Rescaling was performed, but this introduced some error, and might
                ** have pushed us across a rounding boundary. We check whether this
                ** error affected the result by introducing even more error (2ulp in
                ** either direction), and seeing whether a rounding boundary was
                ** crossed. Having already converted the -2ulp case, we save off its
                ** most significant digits, convert the +2ulp case, and compare them.
                */
                int32_t eidx = e + 70 + (MAG_ND_MUL2K_MAX_SHIFT < 29)
                               + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
                const int8_t *m_e = mag_four_ulp_m_e + eidx * 2;
                mag_assert(0 <= eidx && eidx < 128, "bad eidx %d", eidx);
                nd[33] = nd[ndhi];
                nd[32] = nd[(ndhi - 1) & 0x3f];
                nd[31] = nd[(ndhi - 2) & 0x3f];
                nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
                if (mag_unlikely(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
                    goto rescale_failed;
                }
            }
            if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, (int32_t)nde - prec - 1);
                nde += (hilen != mag_ndigits_dec(nd[ndhi]));
            }
            nde += ndebias;
            if ((sf & MAG_FMT_T_FP_F)) {
                /* %g */
                if ((int32_t)prec >= nde && nde >= -4) {
                    if (nde < 0) ndhi = 0;
                    prec -= nde;
                    goto g_format_like_f;
                } else if (!(sf & MAG_FMT_F_ALT) && prec && width > 5) {
                    /* Decrease precision in order to strip trailing zeroes. */
                    char tail[9];
                    uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
                    i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
                    mag_wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == ndhi) { prec = 0; break; }
                            ndlo = (ndlo + 1) & 0x3f;
                            mag_wuint9(tail, nd[ndlo]);
                            i = 9;
                        }
                    }
                }
            }
            if (nde < 0) {
                /* Make nde non-negative. */
                eprefix = '-';
                nde = -nde;
            }
            len = 3 + prec + (prefix != 0) + mag_ndigits_dec((uint32_t)nde) + (nde < 10)
                  + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
            if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            q = mag_wint(p + 1, nd[ndhi]);
            p[0] = p[1]; /* Put leading digit in the correct place. */
            if ((prec | (sf & MAG_FMT_F_ALT))) {
                /* Emit fractional part. */
                p[1] = '.'; p += 2;
                prec -= (size_t)(q - p); p = q; /* Account for digits already emitted. */
                /* Then emit chunks of 9 digits (this may emit 8 digits too many). */
                for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
                    i = (i - 1) & 0x3f;
                    p = mag_wuint9(p, nd[i]);
                }
                if ((sf & MAG_FMT_T_FP_F) && !(sf & MAG_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %e (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            } else {
                p++;
            }
            *p++ = (sf & MAG_FMT_F_UPPER) ? 'E' : 'e';
            *p++ = eprefix; /* + or - */
            if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
            p = mag_wint(p, nde);
        } else {
            /* %f (or, shortly, %g in %f style) */
            if (prec < (size_t)(0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
            }
            g_format_like_f:
            if ((sf & MAG_FMT_T_FP_E) && !(sf & MAG_FMT_F_ALT) && prec && width) {
                /* Decrease precision in order to strip trailing zeroes. */
                if (ndlo) {
                    /* nd has a fractional part; we need to look at its digits. */
                    char tail[9];
                    uint32_t maxprec = (64 - ndlo) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = 64 - (prec + 8) / 9;
                    i = prec - ((63 - ndlo) * 9);
                    mag_wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == 63) { prec = 0; break; }
                            mag_wuint9(tail, nd[++ndlo]);
                            i = 9;
                        }
                    }
                } else {
                    /* nd has no fractional part, so precision goes straight to zero. */
                    prec = 0;
                }
            }
            len = ndhi * 9 + mag_ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
                  + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
            if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            /* Emit integer part. */
            p = mag_wint(p, nd[ndhi]);
            i = ndhi;
            while (i) p = mag_wuint9(p, nd[--i]);
            if ((prec | (sf & MAG_FMT_F_ALT))) {
                /* Emit fractional part. */
                *p++ = '.';
                /* Emit chunks of 9 digits (this may emit 8 digits too many). */
                while ((int32_t)prec > 0 && i != ndlo) {
                    i = (i - 1) & 0x3f;
                    p = mag_wuint9(p, nd[i]);
                    prec -= 9;
                }
                if ((sf & MAG_FMT_T_FP_E) && !(sf & MAG_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %f (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            }
        }
    }
    if ((sf & MAG_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    return p;
}

static uint8_t* mag_default_image_load_impl(const char* file, uint32_t(*whc)[3], mag_color_channels_t channels) {
    mag_assert2(file && *file && whc);
    int w, h, c, dc;
    switch (channels) {
        default: dc = STBI_default; break;
        case MAG_COLOR_CHANNELS_GRAY: dc = STBI_grey; break;
        case MAG_COLOR_CHANNELS_GRAY_A: dc = STBI_grey_alpha; break;
        case MAG_COLOR_CHANNELS_RGB: dc = STBI_rgb; break;
        case MAG_COLOR_CHANNELS_RGBA: dc = STBI_rgb_alpha; break;
    }
    uint8_t* buf = stbi_load(file, &w, &h, &c, dc);
    if (mag_unlikely(!buf || !w || !h || !c || (c != 1 && c != 3 && c != 4))) return NULL;
    (*whc)[0] = (uint32_t)w;
    (*whc)[1] = (uint32_t)h;
    (*whc)[2] = (uint32_t)c;
    return buf;
}

static void mag_default_image_load_free_fn_impl(uint8_t* p) {
    stbi_image_free(p);
}

static bool mag_default_image_save_impl(const char* file, const uint8_t* buf, const uint32_t(*whc)[3]) {
    mag_assert2(file && *file && buf && whc);
    return stbi_write_jpg(file, (int)(*whc)[0], (int)(*whc)[1], (int)(*whc)[2], buf, 100) != 0;
}

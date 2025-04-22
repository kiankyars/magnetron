/*
** (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
**
** This file implements the magnetron runtime core:
**  - The magnetron core API which is used from Python and C as declared in magnetron.h.
**  - Context creation, destruction and all related functions.
**  - Tensor creation, destruction and utility functions, all except the compute functions.
**  - Automatic differentiation and gradient computation.
**  - Metadata of datatypes and operators and misc functions.
**  - Hardware detection and system information.
**  - File storage format loading and saving for (*.mag) files.
*/

#include <magnetron/magnetron.h>
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
#ifdef __linux__
#include <linux/prctl.h>
#include <sys/prctl.h>
#ifdef __aarch64__
#include <sys/auxv.h>
#endif
#endif
#endif

#ifdef MAGNETRON_USE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef MAG_DEBUG
#define MAG_LOG_DEFAULT_ENABLE 1
#else
#define MAG_LOG_DEFAULT_ENABLE 0
#endif
bool mag_log_enabled = MAG_LOG_DEFAULT_ENABLE; /* Read from multiple threads, allowed to be written from main thread once at start. */
#undef MAG_LOG_DEFAULT_ENABLE

void mag_set_log_mode(bool enabled) {
    mag_log_enabled = enabled;
}

#if defined(__linux__) && defined(__GLIBC__)
#include <sys/wait.h>
#include <execinfo.h>
static void mag_dump_backtrace(void) { /* Try to print backtrace using gdb or lldb. */
    char proc[64];
    snprintf(proc, sizeof(proc), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        execlp("gdb", "gdb", "--batch", "-ex", "set style enabled on", "-ex", proc, "-ex", "bt -frame-info source-and-location", "-ex", "detach", "-ex", "quit", NULL);
        execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", proc, NULL);
        exit(EXIT_FAILURE);
    }
    int stat;
    waitpid(pid, &stat, 0);
    if (WIFEXITED(stat) && WEXITSTATUS(stat) == EXIT_FAILURE) {
        void* trace[0xff];
        backtrace_symbols_fd(trace, backtrace(trace, sizeof(trace)/sizeof(*trace)), STDERR_FILENO);
    }
}
#else
static void mag_dump_backtrace(void) { }
#endif

static void MAG_COLDPROC mag_panic_dump(FILE* f, bool cc, const char* msg, va_list args) {
    if (cc) fprintf(f, "%s", MAG_CC_RED);
    vfprintf(f, msg, args);
    if (cc) fprintf(f, "%s", MAG_CC_RESET);
    fputc('\n', f);
    fflush(f);
}

MAG_NORET MAG_COLDPROC MAG_EXPORT void mag_panic(const char* msg, ...) { /* Panic and exit the program. If available print backtrace. */
    va_list args;
    va_start(args, msg);
    #if 0
        FILE* f = fopen("magnetron_panic.log", "w");
        if (f) {
            mag_panic_dump(f, false, msg, args);
            fclose(f), f = NULL;
        }
    #endif
    mag_panic_dump(stdout, true, msg, args);
    va_end(args);
    #ifdef NDEBUG
        mag_dump_backtrace();
    #endif
    abort();
}

#ifdef MAGNETRON_USE_MIMALLOC

static void* mag_alloc_stub(void* blk, size_t size) { /* Allocator stub for mimalloc. */
    if (!size) {
        mi_free(blk);
        return NULL;
    }
    if(!blk) {
        blk = mi_malloc(size);
        if (mag_unlikely(!blk))
            mag_panic("Failed to allocate %zu B memory", size);
        return blk;
    }
    void* block = mi_realloc(blk, size);
    if (mag_unlikely(!block))
        mag_panic("Failed to reallocate %zu B memory", size);
    return block;
}

void* mag_alloc_aligned(size_t size, size_t align) { /* Allocate aligned memory. Alignment must be a power of two. */
    return mi_malloc_aligned(size, align);
}

void mag_free_aligned(void* blk) {
    mi_free(blk);
}

#else

static void* mag_alloc_stub(void* blk, size_t size) { /* Allocator stub for malloc/free. */
    if (!size) {
        free(blk);
        return NULL;
    }
    if(!blk) {
        blk = malloc(size);
        if (mag_unlikely(!blk))
            mag_panic("Failed to allocate %zu B memory", size);
        return blk;
    }
    void* block = realloc(blk, size);
    if (mag_unlikely(!block))
        mag_panic("Failed to reallocate %zu B memory", size);
    return block;
}

/* Allocate aligned memory by overallocating. Alignment must be a power of two. */
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

#endif

void* (*mag_alloc)(void* blk, size_t size) = &mag_alloc_stub;

void* (*mag_get_alloc_fn(void))(void* blk, size_t size) { return mag_alloc; } /* Get global allocator. */
void mag_set_alloc_fn(void* (*alloc)(void* blk, size_t size)) { mag_assert2(alloc); mag_alloc = alloc; } /* Set global allocator. */

/* Humanize memory size. Format and convert a memory size to the appropriate unit. For example. 1024 => 1 KiB */
void mag_humanize_memory_size(size_t n, mag_e11m52_t* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (mag_e11m52_t)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<20);
        *unit = "MiB";
    } else {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<30);
        *unit = "GiB";
    }
}

static void MAG_COLDPROC mag_print_separator(FILE* f) { /* Print a separator line. */
    f = f ? f : stdout;
    char sep[100+1];
    for (size_t i=0; i < (sizeof(sep)/sizeof(*sep))-1; ++i) sep[i] = '-';
    sep[sizeof(sep)/sizeof(*sep)-1] = '\0';
    fprintf(f, "%s\n", sep);
}

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)
/* Format a dimension tuple into a Python-like string. e.g. (4, 12). */
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

/* Open file. Basically fopen but with UTF-8 support on Windows. */
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

uintptr_t mag_thread_id(void) { /* Get the current thread ID. */
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

#if defined(__x86_64__) || defined(_M_X64)
#define _(enumerator, leaf, reg, bit) #enumerator
const char* const mag_amd64_cap_names[MAG_AMD64_CAP__NUM] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#elif defined(__aarch64__)
#define _(ident) #ident
const char* const mag_arm64_cap_names[MAG_ARM64_CAP__NUM] = {
    mag_arm64_feature_def(_, MAG_SEP)
};
#endif

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
    return (uint64_t)llabs((long long)mag_hpc_clock_ns() - (long long)start);
}
static mag_e11m52_t mag_hpc_clock_elapsed_ms(uint64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (mag_e11m52_t)mag_hpc_clock_elapsed_ns(start) / 1e6;
}
#define mag_clock_cycles() ((uint64_t)clock())
#define mag_cycles_per_ms() ((uint64_t)CLOCKS_PER_SEC/1000)

/* Bitset for 32-bit integers. */
typedef uint32_t mag_bitset_t;
mag_static_assert(sizeof(mag_bitset_t) == 4);
#define mag_bitset_size(n) (((n)+((4<<3)-1))>>5)
#define mag_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define mag_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define mag_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define mag_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

/* Tensor hashset with linear probing. */
typedef struct mag_hashset_t {
    size_t len;
    mag_bitset_t* used;
    const mag_tensor_t** keys;
} mag_hashset_t;
#define MAG_HASHSET_FULL ((size_t)-1)
#define MAG_HASHSET_DUPLICATE ((size_t)-2)
#define MAG_HASHSET_MAX ((size_t)-3) /* Must be last. */
#define mag_hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

/* Find optimal hash size for lim sz. */
static size_t mag_hashset_compute_hash_size(size_t sz) {
    mag_assert2(sz > 0 && sz < MAG_HASHSET_MAX);
    static const size_t prime_lut[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    size_t l = 0;
    size_t r = sizeof(prime_lut)/sizeof(*prime_lut);
    while (l < r) { /* Binary search for the smallest prime > sz. */
        size_t mid = (l+r)>>1;
        if (prime_lut[mid] < sz) l = mid+1;
        else r = mid;
    }
    return l < sizeof(prime_lut)/sizeof(*prime_lut) ? prime_lut[l] : sz|1;
}

/* Create a new hashset. */
static mag_hashset_t mag_hashset_init(size_t size) {
    size = mag_hashset_compute_hash_size(size);
    mag_hashset_t set = {
        .len = size,
        .used = (*mag_alloc)(NULL, mag_bitset_size(size)*sizeof(*set.used)),
        .keys = (*mag_alloc)(NULL, size*sizeof(*set.keys)),
    };
    memset(set.used, 0, mag_bitset_size(size)*sizeof(*set.used));
    return set;
}

/* Lookup a key in the hashset. Returns index or MAG_HASHSET_FULL if full. */
static size_t mag_hashset_lookup(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    while (mag_bitset_get(set->used, i) && set->keys[i] != key) { /* Simple linear probe. */
        i = (i+1) % set->len;
        if (i == k) return MAG_HASHSET_FULL; /* Full */
    }
    return i;
}

/* Check if a key exists in the hashset. */
static bool mag_hashset_contains_key(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t i = mag_hashset_lookup(set, key);
    return mag_bitset_get(set->used, i) && i != MAG_HASHSET_FULL;
}

/* Insert a key into the hashset. Returns index or MAG_HASHSET_DUPLICATE if already exists. */
static size_t mag_hashset_insert(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    do { /* Simple linear probing */
        if (!mag_bitset_get(set->used, i)) { /* Insert key. */
            mag_bitset_set(set->used, i);
            set->keys[i] = key;
            return i;
        }
        if (set->keys[i] == key) return MAG_HASHSET_DUPLICATE; /* Key already exists. */
        i = (i+1) % set->len;
    } while (i != k);
    return MAG_HASHSET_FULL; /* Full */
}

/* Reset the hashset. */
static void mag_hashset_reset(mag_hashset_t* set) {
    memset(set->used, 0, mag_bitset_size(set->len)*sizeof(*set->used));
}

/* Clear the hashset. */
static void mag_hashset_free(mag_hashset_t* set) {
    (*mag_alloc)(set->used, 0);
    (*mag_alloc)(set->keys, 0);
}

/* Eval Chebyshev coeffs steps for some x. f(x) : [a, b] -> ℝ. */
static mag_e11m52_t mag_chebyshev_eval(mag_e11m52_t x, mag_e11m52_t a, mag_e11m52_t b, const mag_e11m52_t* coeffs, uint32_t steps) {
    mag_e11m52_t scale = 4.0/(b - a);
    mag_e11m52_t rls = -2.0 + (x - a)*scale;
    mag_e11m52_t k1 = 0.0, k2 = 0.0;
    for (uint32_t j = steps-1; j; --j) {
        mag_e11m52_t tmp = k1;
        k1 = rls*k1 - k2 + coeffs[j];
        k2 = tmp;
    }
    return 0.5*rls*k1 - k2 + 0.5**coeffs;
}

/* Generate Chebyshev coeffs for f(x) : [a, b] -> ℝ. */
static mag_e11m52_t* mag_chebyshev_setup(mag_e11m52_t (*f)(mag_e11m52_t), mag_e11m52_t a, mag_e11m52_t b, uint32_t steps, bool linear_l, bool linear_r) {
    mag_assert2(steps);
    mag_e11m52_t* r = (*mag_alloc)(NULL, sizeof(*r)*steps);
    memset(r, 0, sizeof(*r)*steps);
    mag_e11m52_t dsteps = (mag_e11m52_t)steps;
    for (uint32_t i=0; i < steps; ++i) {
        for (uint32_t j=0; j < steps; ++j) {
            mag_e11m52_t wav = 0.5*(1.0 + cos(M_PI*(j + 0.5)/dsteps));
            mag_e11m52_t x = a + (b - a)*wav, y = (*f)(x);
            mag_e11m52_t weight = cos(M_PI*(mag_e11m52_t)i*(j + 0.5)/dsteps);
            r[i] += 2.0*y*weight/dsteps;
        }
    }
    mag_e11m52_t xmi = 0.0, xma = 0.0;
    if (linear_l) xmi = (*f)(a) - mag_chebyshev_eval(a, a, b, r, steps);
    if (linear_r) xma = (*f)(b) - mag_chebyshev_eval(b, a, b, r, steps);
    r[0] += 2.0*(xma + xmi)*0.5;
    r[1] += (xma - xmi)*0.5;
    return r;
}

/* Performs c = ab with overflow checking. Returns true on overflow, else false. */
static bool MAG_AINLINE mag_imull64_ov(int64_t a, int64_t b, int64_t* c) {
    #ifdef _MSC_VER
    #ifdef _M_ARM64
        uint64_t high = __umulh(a, b);
        *c = a*b;
        return high != (*c>>63);
    #else
        int64_t high;
        int64_t low = _mul128(a, b, &high);
        int64_t sign = low >> 63;
        *c = low;
        return high != sign;
    #endif
    #else
    #if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
        return __builtin_smulll_overflow(a, b, (long long*)c);
    #else
        return __builtin_smull_overflow(a, b, c);
    #endif
    #endif
}

/* Initialize and seed PRNG state. */
void mag_prng_init(mag_prng_state_t* prng, mag_prng_algorithm_t algo, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull;
    switch ((prng->algo = algo)) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Mersenne Twister */
            uint32_t* state = prng->mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1]^(state[i-1]>>30))*1812433253 + i)&~0u;
            prng->mersenne.next = 0;
            prng->mersenne.remaining = 1;
        } break;
        case MAG_PRNG_PCG: { /* PCG-XSH-RR */
            prng->pcg.state = seed^0x853c49e6748fea9bull;
            prng->pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

static void mag_machine_probe(mag_ctx_t* ctx); /* Query host system information. */

/* Print host system and machine information. */
static void mag_system_host_info_dump(mag_ctx_t* ctx) {
    mag_log_info("OS/Kernel: %s", ctx->machine.os_name);
    const char* cpu_arch = "?";
    #if defined(__x86_64__) || defined(_M_X64)
        cpu_arch = "x86-64";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        cpu_arch = "aarch64";
    #else
    #error "Unknwon CPU arch"
    #endif
    mag_log_info("CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u", ctx->machine.cpu_name, ctx->machine.cpu_virtual_cores, ctx->machine.cpu_physical_cores, ctx->machine.cpu_sockets);
    #if defined(__x86_64__) || defined(_M_X64) /* Print detected CPU features for x86-64 platforms. */
        if (mag_log_enabled) {
            printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
            for (uint64_t i=0; i < MAG_AMD64_CAP__NUM; ++i) /* Print all amd64 feature flags such as AVX, AVX2, etc. */
                if (ctx->machine.amd64_cpu_caps & (1ull<<i))
                    printf("%s ", mag_amd64_cap_names[i]);
            putchar('\n');
        }
    #elif defined(__aarch64__) /* Print detected CPU features for ARM64 platforms. */
        if (mag_log_enabled) {
            printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
            for (uint32_t i=0; i < MAG_ARM64_CAP__NUM; ++i)
                if (ctx->machine.arm64_cpu_caps & (1ull<<i))
                    printf("%s ", mag_arm64_cap_names[i]);
            putchar('\n');
        }
    #endif
    /* Now print memory information. */
    mag_e11m52_t mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->machine.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->machine.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->machine.phys_mem_total-(int64_t)ctx->machine.phys_mem_free), &mem_used, &mem_unit_used);
    mag_e11m52_t mem_used_percent = fabs((mag_e11m52_t)(ctx->machine.phys_mem_total-ctx->machine.phys_mem_free))/(mag_e11m52_t)ctx->machine.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

/* Print compiler information such as name, version and build time. */
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

/* Create a magnetron context with the selected compute device. */
mag_ctx_t* mag_ctx_create(mag_compute_device_type_t device) {
    const mag_device_descriptor_t info = {device};
    return mag_ctx_create2(&info);
}

/* Create context with compute device descriptor. */
mag_ctx_t* mag_ctx_create2(const mag_device_descriptor_t* device_info) {
    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_compiler_info(); /* Dump compiler info. */

    /* Initialize context with default values or from context info. */
    mag_ctx_t* ctx = (*mag_alloc)(NULL, sizeof(*ctx)); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init memory pools */
    mag_fixed_intrusive_pool_init(&ctx->tensor_pool, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
    mag_fixed_intrusive_pool_init(&ctx->storage_pool, sizeof(mag_storage_buffer_t), __alignof(mag_storage_buffer_t), 0x1000);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */
    ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; /* Enable gradient recording by default. */
    ctx->prng_algo = MAG_PRNG_MERSENNE_TWISTER;

    /* Query and print host system information. */
    mag_machine_probe(ctx);
    mag_system_host_info_dump(ctx);

    /* Create selected compute device. */
    ctx->device_type = device_info->type;
    ctx->device = mag_init_dynamic_device(ctx, device_info);
    mag_log_info("Compute device: %s", ctx->device->name);

    /* Print context initialization time. */
    mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

void mag_ctx_destroy(mag_ctx_t* ctx) { /* Destroy magnetron context. */
    mag_assert(ctx->num_tensors == 0, "%zu tensors have not been freed", ctx->num_tensors);     /* Leak check if all tensors are freed. */
    mag_assert(ctx->num_storages == 0, "%zu storages have not been freed", ctx->num_storages);  /* Leak check if all storages are freed. */
    mag_fixed_intrusive_pool_destroy(&ctx->tensor_pool);
    mag_fixed_intrusive_pool_destroy(&ctx->storage_pool);
    mag_destroy_dynamic_device(ctx->device); ctx->device = NULL; /* Shutdown compute device. */
    memset(ctx, 255, sizeof(*ctx)); /* Poison context memory range. */
    (*mag_alloc)(ctx, 0); /* Free ctx. */
    ctx = NULL;
    mag_log_info("magnetron context destroyed.");
}

mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* ctx) {
    return ctx->prng_algo;
}

void mag_ctx_set_prng_algorithm(mag_ctx_t* ctx, mag_prng_algorithm_t algorithm, uint64_t seed) {
    mag_log_warn("NYI");
}

mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* ctx) { return ctx->device_type; }
const char* mag_ctx_get_compute_device_name(const mag_ctx_t* ctx) { return ctx->device->name; }
const char* mag_ctx_get_os_name(const mag_ctx_t* ctx) { return ctx->machine.os_name; }
const char* mag_ctx_get_cpu_name(const mag_ctx_t* ctx) { return ctx->machine.cpu_name; }
uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* ctx) { return ctx->machine.cpu_virtual_cores; }
uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* ctx) { return ctx->machine.cpu_physical_cores; }
uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* ctx) { return ctx->machine.cpu_sockets; }
uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* ctx) { return ctx->machine.phys_mem_total; }
uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* ctx) { return ctx->machine.phys_mem_free; }
bool mag_ctx_is_numa_system(const mag_ctx_t* ctx) { return false; /* TODO */ }
size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* ctx) { return 0; /* TODO */ }

/* Set scheduling priority for current thread. */
void mag_thread_set_prio(mag_thread_sched_prio_t prio) {
#ifdef _WIN32
    DWORD policy = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case MAG_THREAD_SCHED_PRIO_NORMAL: policy = THREAD_PRIORITY_NORMAL; break;
        case MAG_THREAD_SCHED_PRIO_MEDIUM: policy = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case MAG_THREAD_SCHED_PRIO_HIGH: policy = THREAD_PRIORITY_HIGHEST; break;
        case MAG_THREAD_SCHED_PRIO_REALTIME: policy = THREAD_PRIORITY_TIME_CRITICAL; break;
    }
    if (mag_unlikely(!SetThreadPriority(GetCurrentThread(), policy))) {
        mag_log_warn("Failed to set thread scheduling priority: %d", prio);
    }
#else
    int32_t policy = SCHED_OTHER;
    struct sched_param p;
    switch (prio) {
        case MAG_THREAD_SCHED_PRIO_NORMAL: p.sched_priority = 0;  policy = SCHED_OTHER; break;
        case MAG_THREAD_SCHED_PRIO_MEDIUM: p.sched_priority = 40; policy = SCHED_FIFO; break;
        case MAG_THREAD_SCHED_PRIO_HIGH: p.sched_priority = 80; policy = SCHED_FIFO; break;
        case MAG_THREAD_SCHED_PRIO_REALTIME: p.sched_priority = 90; policy = SCHED_FIFO; break;
    }
    int status = pthread_setschedparam(pthread_self(), policy, &p);
    if (mag_unlikely(status)) {
        mag_log_warn("Failed to set thread scheduling priority: %d, error: %x", prio, status);
    }
    #endif
}

/* Set thread name for current thread. */
void mag_thread_set_name(const char* name) {
    #if defined(__linux__)
        prctl(PR_SET_NAME, name);
    #elif defined(__APPLE__) && defined(__MACH__)
        pthread_setname_np(name);
    #endif
}

/* Yield current thread. */
void mag_thread_yield(void) {
    #if defined(_WIN32)
        YieldProcessor();
    #else
        sched_yield();
    #endif
}

/* Allocate a new linear chunk for a fixed pool. */
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

/* Initialize fixed intrusive pool and allocate start chunk. */
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

/* Allocate a new fixed block from the pool. Memory is uninitialized. */
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
    /* 3. Current chunk is exhausted, allocate new (slow path) */
    mag_intrusive_chunk* new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk);
    chunk->next = new_chunk;
    pool->chunk_head = new_chunk;
    new_chunk->top -= pool->block_size;
    ++pool->num_chunks;
    return new_chunk->top;
}

/* Free a fixed block back to the pool. This effectively pushes it into the freelist. */
void mag_fixed_intrusive_pool_free(mag_fixed_intrusive_pool* pool, void* blk) {
    *(void**)blk = pool->free_list;
    pool->free_list = blk;
}

/* Destroy fixed intrusive pool and free all allocated memory. */
void mag_fixed_intrusive_pool_destroy(mag_fixed_intrusive_pool* pool) {
    mag_intrusive_chunk* chunk = pool->chunks;
    while (chunk) {
        mag_intrusive_chunk* next = chunk->next;
        (*mag_alloc)(chunk, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

/* Print pool information and allocation stats. */
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
    mag_e11m52_t mem_alloced, pool_mem;
    const char* mem_unit_alloced, *mem_unit_pool;
    mag_humanize_memory_size(pool->num_chunks*pool->blocks_per_chunk*pool->block_size, &mem_alloced, &mem_unit_alloced);
    mag_humanize_memory_size(pool->num_allocs*pool->block_size, &pool_mem, &mem_unit_pool);
    mag_log_info("\t Real Mem Allocated: %.03f %s, Total Pool Mem %.03f %s", mem_alloced, mem_unit_alloced, pool_mem, mem_unit_pool);
}

/* Pack rgb8 into a 32-bit color. Alpha channel unused. */
uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b) { return ((uint32_t)r<<16)|((uint32_t)g<<8)|(uint32_t)b; }

/* Pack rgb8 into a 32-bit color and normalize. Alpha channel unused. */
uint32_t mag_pack_color_f32(mag_e8m23_t r, mag_e8m23_t g, mag_e8m23_t b) {
    return mag_pack_color_u8((uint8_t)(r*255.f), (uint8_t)(g*255.f), (uint8_t)(b*255.f));
}

void mag_ctx_grad_recorder_start(mag_ctx_t* ctx) { ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; }
void mag_ctx_grad_recorder_stop(mag_ctx_t* ctx) { ctx->flags &= ~MAG_CTX_FLAG_GRAD_RECORDER; }
bool mag_ctx_grad_recorder_is_running(const mag_ctx_t* ctx) { return ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER; }

const char* mag_device_type_get_name(mag_compute_device_type_t op) {
    static const char* const names[MAG_COMPUTE_DEVICE_TYPE__NUM] = {
        [MAG_COMPUTE_DEVICE_TYPE_CPU] = "CPU",
        [MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA] = "GPU (CUDA)",
    };
    return names[op];
}

const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type) {
    static const mag_dtype_meta_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_E8M23] = {
            sizeof(mag_e8m23_t),
            "e8m23"
        },
        [MAG_DTYPE_E5M10] = {
            sizeof(mag_e5m10_t),
            "e5m10"
        },
    };
    return &infos[type];
}

static void mag_tensor_dtor(void* self) {
    mag_tensor_t* t = self;
    mag_ctx_t* ctx = t->ctx;
    mag_assert(ctx->num_tensors > 0, "double freed tensor");
    --ctx->num_tensors;
    if (t->grad) {
        mag_tensor_decref(t->grad);
        t->grad = NULL;
    }
    for (int i=0; i < MAG_MAX_OP_INPUTS; ++i)
        if (t->op_inputs[i])
            mag_tensor_decref(t->op_inputs[i]);
    mag_rc_control_decref(&t->storage->rc_control);
    #ifndef NDEBUG
        memset(t, 0, sizeof(*t));
    #endif
    mag_fixed_intrusive_pool_free(&ctx->tensor_pool, t);
}

/* Create a new tensor. The must be created on the same thread as the context. */
static mag_tensor_t* mag_tensor_init_internal(mag_ctx_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape, mag_tensor_t* view, size_t view_offs) {
    uintptr_t tr_id = mag_thread_id();
    /* Ensure that the tensor is created on the same thread as the context. */
    mag_assert(tr_id == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", tr_id, ctx->tr_id);
    mag_assert(shape != NULL && rank >= 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS); /* Check rank */
    if (view) { /* Check if we have a view (base) tensor. */
        mag_assert(view_offs == 0, "NYI"); /* TODO: View offset is not support at the moment. */
        if (view->view_uplink) { /* Traverse view chain and accumulate offset */
            view_offs += view->view_offs;
            view = view->view_uplink;
        }
    }
    int64_t dts = mag_dtype_meta_of(type)->size;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) /* Calculate buffer size and check for overflow. */
        mag_assert2(shape[i] > 0 && !mag_imull64_ov(shape[i], numel, &numel)); /* Overflow in buffer size. Max: INT64_MAX. Reduce dimensions. */
    int64_t numbytes = numel*dts; /* Total bytes required for the data. */
    mag_assert2(!view || !numbytes || numbytes + view_offs <= mag_tensor_get_data_size(view)); /* Slice must be within viewed tensor data range. *//* Allocate memory for tensor struct on CPU RAM. */
    mag_tensor_t* hdr = mag_fixed_intrusive_pool_malloc(&ctx->tensor_pool); /* Allocate tensor header. */
    #ifndef NDEBUG
        memset(hdr, 0, sizeof(*hdr));
    #endif
    *hdr = (mag_tensor_t) { /* Initialize tensor header. */
        .ctx = ctx,
        .rc_control = mag_rc_control_init(hdr, &mag_tensor_dtor), /* Initialize reference counter. */
        .rank = rank,
        .shape = {0},
        .strides = {0},
        .dtype = type,
        .storage = NULL,
        .numel = numel,
        .flags = (view ? MAG_TFLAG_VIEW : MAG_TFLAG_OWNER) | (ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER ? MAG_TFLAG_REQUIRES_GRAD : 0), /* Set flags. */
        .op = MAG_OP_NOP,
        .op_inputs = {0},
        .op_params = {mag_op_param_none()},
        .init_op = MAG_IOP_NOP,
        .init_op_params = {mag_op_param_none()},
        .view_uplink = view,
        .view_offs = view_offs,
        .grad = NULL,
        .name = "",
        .ud = NULL
    };
    ++ctx->num_tensors; /* Increase tensor count in context. */
    /* Allocate device memory */
    mag_compute_device_t* dvc = ctx->device;
    void (*allocator)(mag_compute_device_t*, mag_storage_buffer_t**, size_t, mag_dtype_t) = dvc->alloc_storage; /* Get allocator function. */
    if (view) { /* Reference memory from view */
        hdr->storage = view->storage;
        mag_rc_control_incref(&view->storage->rc_control); /* Increase reference count of the view's storage. */
    }
    else (*allocator)(dvc, &hdr->storage, numbytes, type); /* Else allocate new device memory */
    for (int i=0; i < MAG_MAX_DIMS; ++i)  {   /* Copy dimensions and set unused to identity. */
        hdr->shape[i] = i < rank ? shape[i] : 1;
        hdr->strides[i] = 1;
    }
    /* Compute strides and check for overflow. Strides follow C's row major convention, NO Fortran ordering. */
    hdr->strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
        mag_assert(!mag_imull64_ov(hdr->strides[i+1], hdr->shape[i+1], hdr->strides+i), "Overflow while computing strides");
    }
    return hdr;
}

mag_tensor_t* mag_tensor_empty(mag_ctx_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape) {
    return mag_tensor_init_internal(ctx, type, rank, shape, NULL, 0);
}

mag_tensor_t* mag_tensor_empty_like(mag_tensor_t* isomorph) {
    return mag_tensor_init_internal(isomorph->ctx, isomorph->dtype, isomorph->rank, isomorph->shape, NULL, 0);
}

mag_tensor_t* mag_tensor_empty_scalar(mag_ctx_t* ctx, mag_dtype_t type) {
    return  mag_tensor_empty(ctx, type, 1, (int64_t[1]){1});
}

mag_tensor_t* mag_tensor_scalar(mag_ctx_t* ctx, mag_dtype_t type, float value) {
    mag_tensor_t* tensor = mag_tensor_empty_scalar(ctx, type);
    mag_tensor_fill(tensor, value);
    return tensor;
}

mag_tensor_t* mag_tensor_full(mag_ctx_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape, float value) {
    mag_tensor_t* tensor = mag_tensor_empty(ctx, type, rank, shape);
    mag_tensor_fill(tensor, value);
    return tensor;
}

mag_tensor_t* mag_tensor_full_like(mag_tensor_t* isomorph, float value) {
    mag_tensor_t* tensor = mag_tensor_empty_like(isomorph);
    mag_tensor_fill(tensor, value);
    return tensor;
}

static mag_tensor_t* mag_tensor_inplace_view(mag_tensor_t* base) {
    return mag_tensor_init_internal(base->ctx, base->dtype, base->rank, base->shape, base, 0);
}

/* Check if the input tensors are not null and valid. Return true if valid, else false. */
static bool mag_check_are_inputs_valid(mag_op_t op, mag_tensor_t** inputs, uint32_t numin) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_unlikely(meta->num_inputs != numin || numin > MAG_MAX_OP_INPUTS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            meta->mnemonic, meta->num_inputs, numin
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->num_inputs; ++i) {
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

/* Check if the op parameters exist and have valid types. Return true if valid, else false. */
static bool mag_check_are_op_params_valid(mag_op_t op, const mag_op_param_t* params, uint32_t numparams) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (!meta->num_params) return true; /* No parameters to validate. */
    if (mag_unlikely(meta->num_params != numparams || numparams > MAG_MAX_OP_PARAMS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires at most %u parameters, but %u were provided.\n"
            "    Hint: Ensure the correct number of operation parameters are provided.\n",
            meta->mnemonic, MAG_MAX_OP_PARAMS, meta->num_params
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
    for (uint32_t i=0; i < meta->num_params; ++i) {
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

static bool mag_check_are_dtypes_compatible(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(a->dtype == b->dtype)) return true;
    mag_print_separator(stderr);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor dtypes are not compatible.\n"
        "    - Tensor 1 '%s' Datatype: %s\n"
        "    - Tensor 2 '%s' Datatype: %s\n"
        "    Hint: Ensure the input tensors have the same dtype.\n",
        meta->mnemonic,
        a->name, mag_dtype_meta_of(a->dtype)->name,
        b->name, mag_dtype_meta_of(b->dtype)->name
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Checks if the shape of a and b are equal. If not a detailed error message is printed.  Return true if valid, else false. */
static bool mag_check_is_shape_eq(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_shape_eq(a, b))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Tensor shapes must be equal.\n"
        "    - Tensor 1 '%s' Shape: %s\n"
        "    - Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transpose() or permute().\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Checks if the shape of a and b are broadcastable (b -> a). If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_shape_broadcastable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_can_broadcast(b, a))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    char bc[MAG_MAX_DIMS*2+3] = "[";
    int64_t pos = 1;
    int64_t mr = mag_xmax(a->rank, b->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < a->rank ? a->shape[a->rank-1-d] : 1;
        int64_t bsz = d < b->rank ? b->shape[b->rank-1-d] : 1;
        bc[pos++] = asz == bsz || asz == 1 || bsz == 1 ? 'Y' : 'N';
        bc[pos++] = d == mr-1 ? ']' : ',';
    }
    bc[pos] = '\0';
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be broadcast‑able (NumPy rules).\n"
        "    - Tensor 1 '%s' Shape: %s\n"
        "    - Tensor 2 '%s' Shape: %s\n"
        "    Broadcast‑ability per‑dim (right‑aligned): %s\n"
        "    Hint: Use unsqueeze()/view()/permute() to match shapes.\n",
        meta->mnemonic, a->name, shape_1, b->name, shape_2, bc);

    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Check if a and b can be matrix multiplied. If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_shape_matmulable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(a->shape[1] == b->shape[0])) return true; /* Rows of a must match columns of b. */
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes are not compatible for matrix multiplication. The rows of the first tensor must match the columns of the second tensor.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transpose() or permute().\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Check if tensor is contiguous in memory. This is an required for some optimized compute algorithms. If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_contiguous(mag_op_t op, const mag_tensor_t* a) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_contiguous(a))) return true;
    mag_print_separator(stderr);
    char shape[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape, &a->shape, a->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Tensor '%s' must be contiguous. Shape: %s\n"
        "    Hint: Make tensor contiguous using clone().\n",
        meta->mnemonic,
        a->name,
        shape
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Generic function which validates the tensors for common unary operations such as abs, neg, etc. */
static bool mag_validate_op_unary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
   return mag_check_is_shape_eq(op, result, inputs[0]);
}

/* Generic function which validates the tensors for common binary operations such as add, sub, etc. */
static bool mag_validate_op_binary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    bool valid = true;
    valid = valid && mag_check_is_shape_eq(op, result, inputs[0]);
    valid = valid && mag_check_is_shape_broadcastable(op, inputs[0], inputs[1]);
    valid = valid && mag_check_is_contiguous(op, result);
    return valid;
}

/* Validation function for the transpose operation. */
static bool mag_validate_op_transpose(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return true;
}

/* Generic function which validates scalar operations such as adds, subs etc. */
static bool mag_validate_op_scalar(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_check_is_contiguous(op, inputs[0]);
}

/* Validation function for the matmul operation. */
static bool mag_validate_op_matmul(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_check_is_shape_matmulable(op, inputs[0], inputs[1]);
}

static bool mag_validate_op_repeat_rev(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_check_is_shape_broadcastable(op, inputs[0], inputs[1]);
}

static mag_tensor_t* mag_result_constructor_routine_isomorph(mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_tensor_empty_like(*inputs);
}

static mag_tensor_t* mag_result_constructor_routine_view(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* base = *inputs;
    mag_tensor_t* result = mag_tensor_inplace_view(base);
    if (*base->name)
        mag_tensor_fmt_name(result, "%s (view)", base->name);
    return result;
}

static mag_tensor_t* mag_result_constructor_routine_scalar(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* base = *inputs;
    return mag_tensor_empty_scalar(base->ctx, base->dtype);
}

static mag_tensor_t* mag_result_constructor_routine_transposed(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* transposed = mag_result_constructor_routine_view(inputs, params);
    mag_swap(int64_t, transposed->shape[0], transposed->shape[1]);
    mag_swap(int64_t, transposed->strides[0], transposed->strides[1]);
    if (*inputs[0]->name)
        mag_tensor_fmt_name(transposed, "%s (T)", inputs[0]->name);
    return transposed;
}

static mag_tensor_t* mag_result_constructor_routine_permuted(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_assert2(params != NULL);
    const mag_tensor_t* base = inputs[0];
    mag_tensor_t* permuted = mag_result_constructor_routine_view(inputs, params);
    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < base->rank; ++i)
        axes[i] = mag_op_param_unpack_u64_or_panic(params[i]);
    for (int64_t i=0; i < base->rank; ++i) {
        mag_assert2(axes[i] < base->rank);
        for (int64_t j=i+1; j < base->rank; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %zu == %zu", axes[i], axes[j]);
    }
    int64_t tmp_shape[MAG_MAX_DIMS];
    int64_t tmp_stride[MAG_MAX_DIMS];
    memcpy(tmp_shape, base->shape, sizeof(tmp_shape));
    memcpy(tmp_stride, base->strides, sizeof(tmp_stride));
    for (int64_t i=0; i < base->rank; ++i) {
        permuted->shape[i] = tmp_shape[axes[i]];
        permuted->strides[i] = tmp_stride[axes[i]];
    }
    if (*base->name)
        mag_tensor_fmt_name(permuted, "%s (perm)", base->name);
    return permuted;
}

static mag_tensor_t* mag_result_constructor_routine_matmul(mag_tensor_t** inputs,  const mag_op_param_t* params) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[MAG_MAX_DIMS] = {0};
    int64_t* rd0 = shape;
    int64_t* rd1 = shape+1;
    int64_t rank = 0;
    if (inputs[0]->rank == 1 && inputs[1]->rank == 2) { /* (ℝⁿ)(ℝⁿˣʳ) → ℝʳ */
        *rd0 = 1;
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    } else if (inputs[0]->rank == 2 && inputs[1]->rank == 1) { /* (ℝᵐˣⁿ)(ℝⁿ) → ℝᵐ */
        *rd0 = inputs[0]->shape[0];
        rank = 1;
    } else if (inputs[0]->rank == 1 && inputs[1]->rank == 1) { /* (ℝⁿ)(ℝⁿ) → ℝ */
        rank = 1;
    } else { /* (ℝᵐˣⁿ)(ℝⁿˣʳ) → ℝᵐˣʳ */
        *rd0 = inputs[0]->shape[0];
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    }
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, rank, shape, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_repeat_back(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, inputs[1]->rank, inputs[1]->shape, NULL, 0);
}

static void mag_op_backward_clone(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_view(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_transpose(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* t = mag_transpose(node->grad);
    *grads = mag_clone(t);
    mag_tensor_decref(t);
}

static void mag_op_backward_mean(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* scale = mag_tensor_full_like(x, (float)(1.0f/(mag_e11m52_t)x->numel));
    *grads = mag_mul(scale, node->grad);
    mag_tensor_decref(scale);
}

static void mag_op_backward_sum(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* ones = mag_tensor_full_like(x, 1.0f);
    *grads = mag_mul(ones, node->grad);
    mag_tensor_decref(ones);
}

static void mag_op_backward_abs(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* step = mag_step(x);
    mag_tensor_t* one = mag_tensor_scalar(x->ctx, x->dtype, 1.0f);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* step2 = mag_mul(step, two);
    mag_tensor_t* sign = mag_sub(step2, one);
    grads[0] = mag_mul(node->grad, sign);
    mag_tensor_decref(two);
    mag_tensor_decref(one);
    mag_tensor_decref(step);
    mag_tensor_decref(step2);
    mag_tensor_decref(sign);
}

static void mag_op_backward_neg(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* m1 = mag_tensor_scalar(node->grad->ctx, node->grad->dtype, -1.f);
    grads[0] = mag_mul(node->grad, m1);
    mag_tensor_decref(m1);
}

static void mag_op_backward_log(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    grads[0] = mag_div(node->grad, x);
}

static void mag_op_backward_sqr(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* two_x = mag_mul(x, two);
    grads[0] = mag_mul(node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
}

static void mag_op_backward_sqrt(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sqrt_x = mag_sqrt(x);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* denom = mag_mul(sqrt_x, two);
    grads[0] = mag_div(node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
}

static void mag_op_backward_sin(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* cos_x = mag_cos(x);
    grads[0] = mag_mul(node->grad, cos_x);
    mag_tensor_decref(cos_x);
}

static void mag_op_backward_cos(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sinx = mag_sin(x);
    mag_tensor_t* nsinx = mag_neg(sinx);
    grads[0] = mag_mul(node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
}

static void mag_op_backward_exp(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* exp_x = mag_exp(x);
    grads[0] = mag_mul(node->grad, exp_x);
    mag_tensor_decref(exp_x);
}

static void mag_op_backward_softmax(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = mag_softmax(x);
    mag_tensor_t* tmp = mag_mul(node->grad, y);
    mag_tensor_t* sum_tmp = mag_sum(tmp);
    mag_tensor_t* diff = mag_sub(node->grad, sum_tmp);
    grads[0] = mag_mul(y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
}

static void mag_op_backward_sigmoid(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_sigmoid_dv(x);
    grads[0] = mag_mul(dv, node->grad);
    mag_tensor_decref(dv);
}

static void mag_op_backward_silu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_silu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_tanh(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_tanh_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_relu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* mask = mag_step(x);
    grads[0] = mag_mul(node->grad, mask);
    mag_tensor_decref(mask);
}

static void mag_op_backward_gelu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_gelu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_add(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags&MAG_TFLAG_REQUIRES_GRAD)
        grads[0] = mag_clone(node->grad);
    if (y->flags&MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* grad = node->grad;
        if (!mag_tensor_is_shape_eq(x, y)) {
            grad = mag_repeat_back(grad, y);
        } else {
            grad = mag_clone(grad); /* Output gradients must be a new allocated tensor, so we clone. */
        }
        grads[1] = grad;
    }
}

static void mag_op_backward_sub(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags&MAG_TFLAG_REQUIRES_GRAD)
        grads[0] = mag_clone(node->grad);
    if (y->flags&MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* mg = mag_neg(node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmg = mg;
            mg = mag_repeat_back(pmg, y);
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
}

static void mag_op_backward_mul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags&MAG_TFLAG_REQUIRES_GRAD)
        grads[0] = mag_mul(node->grad, y);
    if (y->flags&MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* xg = mag_mul(x, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pxg = xg;
            xg = mag_repeat_back(pxg, y);
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
}

static void mag_op_backward_div(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags&MAG_TFLAG_REQUIRES_GRAD)
        grads[0] = mag_div(node->grad, y);
    if (y->flags&MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* gx = mag_mul(node->grad, x);
        mag_tensor_t* yy = mag_mul(y, y);
        mag_tensor_t* gxyy = mag_div(gx, yy);
        mag_tensor_t* mgxyy = mag_neg(gxyy);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmgxyy = mgxyy;
            mgxyy = mag_repeat_back(pmgxyy, y);
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
}

static void mag_op_backward_matmul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    mag_tensor_t* yt = mag_transpose(y);
    mag_tensor_t* ytc = mag_clone(yt);
    grads[0] = mag_matmul(node->grad, ytc);
    mag_tensor_t* xt = mag_transpose(x);
    mag_tensor_t* xtc = mag_clone(xt);
    grads[1] = mag_matmul(xtc, node->grad);
    mag_tensor_decref(ytc);
    mag_tensor_decref(yt);
    mag_tensor_decref(xtc);
    mag_tensor_decref(xt);
}

const mag_op_meta_t* mag_op_meta_of(mag_op_t opc) {
    static const mag_op_meta_t infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .num_inputs = 0,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = NULL,
            .validator = NULL
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_clone,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_view,
            .r_alloc = &mag_result_constructor_routine_view,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_transpose,
            .r_alloc = &mag_result_constructor_routine_transposed,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .num_inputs = 1,
            .num_params = MAG_MAX_DIMS,
            .param_types = {
                MAG_OPP_U64,
                MAG_OPP_U64,
                MAG_OPP_U64,
                MAG_OPP_U64,
                MAG_OPP_U64,
                MAG_OPP_U64,
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_permuted,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_MEAN] = {
            .mnemonic = "mean",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_mean,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MIN] = {
            .mnemonic = "min",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MAX] = {
            .mnemonic = "max",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_SUM] = {
            .mnemonic = "sum",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_sum,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_abs,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SGN] = {
            .mnemonic = "sgn",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_NEG] = {
            .mnemonic = "neg",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_neg,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_LOG] = {
            .mnemonic = "log",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_log,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQR] = {
            .mnemonic = "sqr",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqr,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQRT] = {
            .mnemonic = "sqrt",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags =  MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqrt,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIN] = {
            .mnemonic = "sin",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sin,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_COS] = {
            .mnemonic = "cos",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_cos,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_STEP] = {
            .mnemonic = "step",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_EXP] = {
            .mnemonic = "exp",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_exp,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_FLOOR] = {
            .mnemonic = "floor",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_CEIL] = {
            .mnemonic = "ceil",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ROUND] = {
            .mnemonic = "round",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX] = {
            .mnemonic = "softmax",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_softmax,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX_DV] = {
            .mnemonic = "softmax_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID] = {
            .mnemonic = "sigmoid",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = mag_op_backward_sigmoid,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID_DV] = {
            .mnemonic = "sigmoid_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_HARD_SIGMOID] = {
            .mnemonic = "hard_sigmoid",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU] = {
            .mnemonic = "silu",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_silu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU_DV] = {
            .mnemonic = "silu_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH] = {
            .mnemonic = "tanh",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_tanh,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH_DV] = {
            .mnemonic = "tanh_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU] = {
            .mnemonic = "relu",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_relu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU_DV] = {
            .mnemonic = "relu_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU] = {
            .mnemonic = "gelu",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_gelu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU_DV] = {
            .mnemonic = "gelu_dv",
            .num_inputs = 1,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ADD] = {
            .mnemonic = "add",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_add,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SUB] = {
            .mnemonic = "sub",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sub,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_MUL] = {
            .mnemonic = "mul",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_mul,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_DIV] = {
            .mnemonic = "div",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {MAG_OPP_NONE},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_div,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_matmul,
            .r_alloc = &mag_result_constructor_routine_matmul,
            .validator = &mag_validate_op_matmul,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_REPEAT_BACK] = {
            .mnemonic = "repeat_rev",
            .num_inputs = 2,
            .num_params = 0,
            .param_types = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_repeat_back,
            .validator = mag_validate_op_repeat_rev
        }
    };
    return infos+opc;
}

#undef mag_validate_inputs

int64_t mag_tensor_get_data_size(const mag_tensor_t* t) { return t->storage->size; }
int64_t mag_tensor_get_numel(const mag_tensor_t* t) { return t->numel; }

void mag_tensor_incref(mag_tensor_t* t) { /* Increase reference count of the tensor. */
    mag_rc_control_incref(&t->rc_control);
}

bool mag_tensor_decref(mag_tensor_t* t) { /* Decrease reference count of the tensor. */
    return mag_rc_control_decref(&t->rc_control);
}

mag_tensor_t* mag_tensor_detach(mag_tensor_t* t) {
    t->op = MAG_OP_NOP;
    t->init_op = MAG_IOP_NOP;
    memset(t->op_inputs, 0, sizeof(t->op_inputs));
    memset(t->op_params, 0, sizeof(t->op_params));
    memset(t->init_op_params, 0, sizeof(t->init_op_params));
    return t;
}

/*
** Hash the tensor header metadata (shape, strides, dtype, numel), TODO @mario: these values are codependent and give no new source of entropy.
** without the tensors data, opcode or parent tensors.
**
*/
uint32_t mag_tensor_weak_hash(const mag_tensor_t* _Nonnull t) {
    uint32_t h = 0;
    for (int64_t i=0; i < t->rank; ++i) {
        mag_hash_combine(&h, t->shape[i]^(t->shape[i]>>32));
        mag_hash_combine(&h, t->strides[i]^(t->strides[i]>>32));
    }
    mag_hash_combine(&h, t->dtype);
    mag_hash_combine(&h, t->numel^(t->numel>>32));
    return h;
}

static void mag_pre_validate_op(
    mag_op_t op,                /* Operator code */
    mag_tensor_t** inputs,      /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t numin,             /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_op_param_t* opps,      /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t numopps            /* Number of operation parameters. Must be same as specified in the op metadata. */
) {
    mag_assert2(op != MAG_OP_NOP);
    mag_assert(inputs && mag_check_are_inputs_valid(op, inputs, numin), "invalid input tensors for operation '%s'", mag_op_meta_of(op)->mnemonic);
    mag_assert(mag_check_are_op_params_valid(op, opps, numopps), "invalid parameters for operation '%s'", mag_op_meta_of(op)->mnemonic);
    if (numin > 1) { /* Check that dtype are compatible. */
        mag_assert(mag_check_are_dtypes_compatible(op, inputs[0], inputs[1]), "invalid dtypes for operation '%s'", mag_op_meta_of(op)->mnemonic);
    }
}

/* Execute init/normal operator on R. */
static void MAG_HOTPROC mag_op_exec(mag_tensor_t* R, mag_compute_device_t* dvc, mag_exec_stage_t stage) {
    void (*exec)(mag_compute_device_t*, mag_tensor_t*) = stage == MAG_STAGE_INIT ? dvc->eager_exec_init : dvc->eager_exec_fwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
}

/* Execute an operator on the active compute device and return result tensor. */
static mag_tensor_t* MAG_HOTPROC mag_tensor_operator(
    mag_ctx_t* ctx,             /* Context to use. All involved tensors must be allocated from this context. */
    mag_op_t op,                /* Operator code */
    bool inplace,               /* Attempt to perform inplace operation? e.g.    r <- x += y    instead of      r <- x + y */
    mag_tensor_t** inputs,      /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t num_inputs,        /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_op_param_t* params,    /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t num_params,        /* Number of operation parameters. Must be same as specified in the op metadata. */
    mag_exec_stage_t stage      /* Graph evaluation direction. */
) {
    mag_pre_validate_op(op, inputs, num_inputs, params, num_params); /* Validate that function inputs are correct. */

    /* Query validate and result constructor functions for the scheduled opcode. */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_op_param_t*) = meta->r_alloc;                         /* Get result allocator function. */
    bool (*validate_op)(mag_op_t, mag_tensor_t*, mag_tensor_t**, const mag_op_param_t*) = meta->validator;   /* Get validator function. */

    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);                                          /* Inplace operation requested and supported? */
    mag_tensor_t* result = inplace ? mag_tensor_inplace_view(*inputs) : (*r_alloc)(inputs, params);     /* If inplace, result views x (input 0), else a new result tensor is allocated. */
    result->op = op;                                                                                    /* Set opcode of result. */

    mag_assert((*validate_op)(op, result, inputs, params), "Invalid operation %s.", meta->mnemonic);    /* Now validate the full operation and panic if something doesn't work out. */
    mag_assert2(num_inputs <= MAG_MAX_OP_INPUTS);                                                       /* Assert correct input tensor count. */

    /* Apply input tensor's gradient rules and increase their lifetime. */
    for (uint32_t i=0; i < num_inputs; ++i) {
        mag_tensor_t* input = inputs[i];
        result->op_inputs[i] = input;

        /* If gradient tracking is enabled, the result tensor inherits the input's gradient rules. */
        if (ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) {
            result->flags |= input->flags & MAG_TFLAG_REQUIRES_GRAD; /* Set gradient tracking flag if set in input. */
            mag_tensor_incref(input);                                /* Keep input alive for the backward pass. */
        }
    }

    if (params) /* If available, copy operation parameters to result */
        memcpy(result->op_params, params, num_params*sizeof(*params));

    mag_op_exec(result, ctx->device, stage);  /* Now execute the operator. */

    if (!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER)) {
        result = mag_tensor_detach(result); /* If gradient are not recorded, detach the tensor (clear parent and opcode). TODO: why are we doing this? */
    }
    return result;
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_view(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_transpose(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t d4, uint32_t d5) {
    mag_op_param_t params[MAG_MAX_OP_PARAMS] = {
        mag_op_param_wrap_u64(d0),
        mag_op_param_wrap_u64(d1),
        mag_op_param_wrap_u64(d2),
        mag_op_param_wrap_u64(d3),
        mag_op_param_wrap_u64(d4),
        mag_op_param_wrap_u64(d5),
    };
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, params, sizeof(params)/sizeof(*params), MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mean(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_min(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_max(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sum(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x) {
    mag_assert(!(x->flags&MAG_TFLAG_REQUIRES_GRAD), "inplace operations are not supported for gradient-tracking tensors");
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_adds(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_adds_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_subs(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_subs_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_muls(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_muls_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_divs(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_divs_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_pows(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_POWS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_pows_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_POWS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_repeat_back(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_REPEAT_BACK, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tensor_get_arg(const mag_tensor_t* t, size_t slot) {
    mag_assert(slot < MAG_MAX_OP_INPUTS, "slot must be within [0, %d)", MAG_MAX_OP_INPUTS);
    return t->op_inputs[slot];
}

void mag_tensor_set_arg(mag_tensor_t* t, size_t slot, mag_tensor_t* arg) {
    mag_assert(slot < MAG_MAX_OP_INPUTS, "slot must be within [0, %d)", MAG_MAX_OP_INPUTS);
    mag_assert(t->op_inputs[slot] == NULL, "argument at slot #%zu already set", slot);
    t->op_inputs[slot] = arg;
}

void mag_tensor_fill_from_floats(mag_tensor_t* t, const mag_e8m23_t* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CVT_E8M23, 0, (void*)data, len*sizeof(*data));
}

void mag_tensor_fill_from_raw_bytes(mag_tensor_t* t, const void* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CPY, 0, (void*)data, len);
}

void mag_tensor_fill(mag_tensor_t* t, mag_e8m23_t x) {
    t->init_op = MAG_IOP_BROADCAST;
    t->init_op_params[0] = mag_op_param_wrap_e8m23(x);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform(mag_tensor_t* t, mag_e8m23_t min, mag_e8m23_t max) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    t->init_op = MAG_IOP_RAND_UNIFORM;
    t->init_op_params[0] = mag_op_param_wrap_e8m23(min);
    t->init_op_params[1] = mag_op_param_wrap_e8m23(max);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, mag_e8m23_t mean, mag_e8m23_t stddev) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    t->init_op = MAG_IOP_RAND_NORMAL;
    t->init_op_params[0] = mag_op_param_wrap_e8m23(mean);
    t->init_op_params[1] = mag_op_param_wrap_e8m23(stddev);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

uint64_t mag_tensor_get_refcount(const mag_tensor_t* t) { return t->rc_control.rc; }
uint64_t mag_tensor_get_storage_refcount(const mag_tensor_t* t) { return t->storage->rc_control.rc; }

size_t mag_tensor_get_memory_usage(const mag_tensor_t* t) {
    return sizeof(*t) + mag_tensor_get_data_size(t);
}

mag_static_assert(sizeof(char) == sizeof(uint8_t));
void mag_tensor_set_name(mag_tensor_t* t, const char* name) {
    snprintf((char*)t->name, MAG_MAX_TENSOR_NAME_LEN, "%s", name);
}

void mag_tensor_fmt_name(mag_tensor_t* t, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf((char*)t->name, sizeof(t->name), fmt, args);
    va_end(args);
}

const char* mag_tensor_get_name(const mag_tensor_t* t) {
    return (const char*)t->name;
}

int64_t mag_tensor_get_rank(const mag_tensor_t* t) { return t->rank; }
const int64_t* mag_tensor_get_shape(const mag_tensor_t* t) { return t->shape; }
const int64_t* mag_tensor_get_strides(const mag_tensor_t* t) { return t->strides; }
mag_dtype_t mag_tensor_get_dtype(const mag_tensor_t* t) { return t->dtype; }
void* mag_tensor_get_data_ptr(const mag_tensor_t* t) { return (void*)t->storage->base; }

void* _Nonnull mag_tensor_get_raw_data_as_bytes(mag_tensor_t* _Nonnull t) {
    size_t size = t->storage->size;
    mag_assert2(size);
    void* dst = (*mag_alloc)(NULL, size); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, MAG_TRANSFER_OP_CPY, 0, dst, size);
    return dst;
}

void mag_tensor_get_raw_data_as_bytes_free(void* _Nonnull ret_val) {
    (*mag_alloc)(ret_val, 0);
}


mag_e8m23_t* mag_tensor_get_data_as_floats(mag_tensor_t* t) {
    size_t size = t->numel*sizeof(mag_e8m23_t);
    mag_assert2(size);
    mag_e8m23_t* dst = (*mag_alloc)(NULL, size); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, MAG_TRANSFER_OP_CVT_E8M23, 0, dst, size);
    return dst;
}

void mag_tensor_get_data_as_floats_free(mag_e8m23_t* ret_val) {
    (*mag_alloc)(ret_val, 0);
}

bool mag_tensor_is_shape_eq(const mag_tensor_t* x, const mag_tensor_t* y) {
    return memcmp(x->shape, y->shape, sizeof(x->shape)) == 0;
}

bool mag_tensor_are_strides_eq(const mag_tensor_t* x, const mag_tensor_t* y) {
    return memcmp(x->strides, y->strides, sizeof(x->strides)) == 0;
}

bool mag_tensor_can_broadcast(const mag_tensor_t* small, const mag_tensor_t* big) {
    int64_t mr = mag_xmax(small->rank, big->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < small->rank ? small->shape[small->rank-1-d] : 1;
        int64_t bsz = d < big->rank ? big->shape[big->rank-1-d] : 1;
        if (asz != bsz && asz != 1 && bsz != 1)
            return false;
    }
    return true;
}

bool mag_tensor_is_transposed(const mag_tensor_t* t) { return t->strides[0] > t->strides[1]; }

bool mag_tensor_is_permuted(const mag_tensor_t* t) {
    for (int i=0; i < MAG_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool mag_tensor_is_contiguous(const mag_tensor_t* t) {
    int64_t str = 1;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t size_d = t->shape[d];
        if (size_d == 1) continue;
        if (t->strides[d] != str) return false;
        str *= size_d;
    }
    return true;
}

mag_tensor_t* mag_tensor_get_grad(const mag_tensor_t* t) {
    mag_assert2(t->flags & MAG_TFLAG_REQUIRES_GRAD);
    if (t->grad) mag_tensor_incref(t->grad);
    return t->grad;
}

bool mag_tensor_requires_grad(const mag_tensor_t* t) {
    return t->flags & MAG_TFLAG_REQUIRES_GRAD;
}

void mag_tensor_set_requires_grad(mag_tensor_t* t, bool requires_grad) {
    if (requires_grad && t->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER)
        t->flags |= MAG_TFLAG_REQUIRES_GRAD;
    else t->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
}

typedef struct mag_topo_stack_record_t {
    mag_tensor_t* tensor;
    uint32_t next_child_idx;
} mag_topo_stack_record_t;

typedef struct mag_tensor_array_t {
    mag_tensor_t** data;
    size_t size;
    size_t capacity;
} mag_tensor_array_t;

static void mag_tensor_array_init(mag_tensor_array_t* arr) {
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

static void mag_tensor_array_free(mag_tensor_array_t* arr) {
    (*mag_alloc)(arr->data, 0);
    arr->size = 0;
    arr->capacity = 0;
}

static void mag_tensor_array_push(mag_tensor_array_t* arr, mag_tensor_t* t) {
    if (arr->size == arr->capacity) {
        size_t cap = !arr->capacity ? 16 : arr->capacity<<1;
        arr->data = (*mag_alloc)(arr->data, cap*sizeof(*arr->data));
        arr->capacity = cap;
    }
    arr->data[arr->size++] = t;
}

static void mag_collect_topo_iterative(mag_tensor_t* root, mag_tensor_array_t* out_array) {
    size_t sta_len = 0, sta_cap = 0;
    mag_topo_stack_record_t* stack = NULL;

    #define mag_sta_push(_t) do { \
        if (sta_len == sta_cap) { \
        size_t old_cap = sta_cap; \
        size_t nc = (old_cap == 0) ? 16 : (old_cap * 2); \
        stack = (*mag_alloc)(stack, nc*sizeof(*stack)); \
        sta_cap = nc; \
        } \
        stack[sta_len].tensor = (_t); \
        stack[sta_len].next_child_idx = 0; \
        sta_len++; \
    } while(0)
    #define mag_sta_pop() (stack[--sta_len])

    if (!(root->flags & MAG_TFLAG_REQUIRES_GRAD)) return;
    mag_hashset_t visited = mag_hashset_init(8192); // todo dynamic
    mag_sta_push(root);
    while (sta_len) { /* Iterative DFS */
        mag_topo_stack_record_t* top = &stack[sta_len - 1];
        mag_tensor_t* cur_tensor = top->tensor;
        if (top->next_child_idx < mag_op_meta_of(cur_tensor->op)->num_inputs) {
            mag_tensor_t* child = cur_tensor->op_inputs[top->next_child_idx++];
            if (child && (child->flags & MAG_TFLAG_REQUIRES_GRAD)) {
                if (!mag_hashset_contains_key(&visited, child)) {
                    mag_hashset_insert(&visited, child);
                    mag_sta_push(child);
                }
            }
        } else {
            (void)mag_sta_pop();
            mag_tensor_array_push(out_array, cur_tensor);
        }
    }

    #undef mag_sta_push
    #undef mag_sta_pop

    (*mag_alloc)(stack, 0);
    mag_hashset_free(&visited);
}

static void mag_tensor_patch_grad(mag_tensor_t* dst, mag_tensor_t* grad) {
    mag_tensor_fmt_name(grad, "%s (grad)", dst->name);
    grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
    dst->grad = grad;
}

void mag_tensor_backward(mag_tensor_t* root) {
    mag_assert(root->flags & MAG_TFLAG_REQUIRES_GRAD, "Tensor must require grad to back-propagate");
    mag_assert(root->rank == 1 && root->numel == 1, "Tensor must be a scalar to back-propagate");
    mag_ctx_grad_recorder_stop(root->ctx);
    mag_tensor_array_t post_order;
    mag_tensor_array_init(&post_order);
    mag_collect_topo_iterative(root, &post_order);
    fflush(stdout);
    if (mag_unlikely(!post_order.size)) goto end;
    for (size_t i=0, j = post_order.size-1; i < j; ++i, --j)
        mag_swap(mag_tensor_t*, post_order.data[i], post_order.data[j]);
    for (size_t id=0; id < post_order.size; ++id) {
        mag_tensor_t* child = post_order.data[id];
        mag_assert2(child);
        const mag_op_meta_t* meta = mag_op_meta_of(child->op);
        if (!child->grad) {
            mag_tensor_t* grad = mag_tensor_empty_like(child);
            mag_tensor_fill(grad, 1.0f);
            mag_tensor_patch_grad(child, grad);
        }
        if (mag_unlikely(child->op == MAG_OP_NOP)) continue;
        mag_tensor_t* grads[MAG_MAX_OP_INPUTS] = {0};
        void (*op_bwd)(mag_tensor_t*, mag_tensor_t**) = meta->backward;
        mag_assert2(op_bwd);
        (*op_bwd)(child, grads);
        uint32_t numin = meta->num_inputs;
        mag_assert2(numin <= MAG_MAX_OP_INPUTS);
        for (uint32_t i=0; i < numin; ++i) {
            mag_tensor_t* input = child->op_inputs[i];
            mag_assert2(input);
            if (!(input->flags & MAG_TFLAG_REQUIRES_GRAD)) continue;
            mag_tensor_t* gri = grads[i];
            mag_assert(gri, "Gradient for op %s, input #%d is not computed", meta->mnemonic, i);
            if (!input->grad) {
                mag_tensor_patch_grad(input, gri);
            } else {
                mag_tensor_t* acc = mag_add(gri, input->grad);
                mag_tensor_decref(input->grad);
                mag_tensor_patch_grad(input, acc);
                mag_tensor_decref(gri);
            }
        }
    }
    mag_tensor_array_free(&post_order);
    end:
    mag_ctx_grad_recorder_start(root->ctx);
}

void mag_tensor_zero_grad(mag_tensor_t* t) {
    if (t->grad && t->flags & MAG_TFLAG_REQUIRES_GRAD)
        mag_tensor_fill(t->grad, 0.0f);
}

mag_e8m23_t mag_tensor_subscript_get_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_storage_buffer_t* sto = t->storage;
    mag_e8m23_t val;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, MAG_TRANSFER_OP_CVT_E8M23, sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val));
    return val;
}

void mag_tensor_subscript_set_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, mag_e8m23_t val) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CVT_E8M23, sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val));
}

static MAG_AINLINE void mag_tensor_unravel_index(const mag_tensor_t* t, int64_t v_idx, int64_t(*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

mag_e8m23_t mag_tensor_subscript_get_flattened(mag_tensor_t* t, int64_t idx) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        return mag_tensor_subscript_get_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    mag_storage_buffer_t* sto = t->storage;
    mag_e8m23_t val;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, MAG_TRANSFER_OP_CVT_E8M23, sto->granularity*idx, &val, sizeof(val));
    return val;
}

void mag_tensor_subscript_set_flattened(mag_tensor_t* t, int64_t idx, mag_e8m23_t val) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        mag_tensor_subscript_set_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], val);
        return;
    }
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CVT_E8M23, sto->granularity*idx, &val, sizeof(val));
}

void mag_tensor_img_draw_box(mag_tensor_t* t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb) {
    mag_assert(t->rank == 3, "Tensor must be 3D image tensor");
    mag_assert2(x2 > x1 && y2 > y1 && x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0);
    mag_e8m23_t* buf = mag_tensor_get_data_ptr(t);
    int32_t w = (int32_t)mag_tensor_get_width(t);
    int32_t h = (int32_t)mag_tensor_get_height(t);
    int32_t c = (int32_t)mag_tensor_get_channels(t);
    mag_assert2(w && h && c == 3);
    mag_e8m23_t r = (mag_e8m23_t)((rgb>>16)&0xff) / 255.0f;
    mag_e8m23_t g = (mag_e8m23_t)((rgb>>8)&0xff) / 255.0f;
    mag_e8m23_t b = (mag_e8m23_t)(rgb&0xff) / 255.0f;
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
            mag_e8m23_t* r1 = buf + j + yy1*w + 0*w*h;
            mag_e8m23_t* r2 = buf + j + yy2*w + 0*w*h;
            mag_e8m23_t* g1 = buf + j + yy1*w + 1*w*h;
            mag_e8m23_t* g2 = buf + j + yy2*w + 1*w*h;
            mag_e8m23_t* b1 = buf + j + yy1*w + 2*w*h;
            mag_e8m23_t* b2 = buf + j + yy2*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_get_data_size(t));
            *r1 = *r2 = r;
            *g1 = *g2 = g;
            *b1 = *b2 = b;
        }
        for (int32_t j = yy1; j <= yy2; ++j) {
            mag_e8m23_t* r1 = buf + xx1 + j*w + 0*w*h;
            mag_e8m23_t* r2 = buf + xx2 + j*w + 0*w*h;
            mag_e8m23_t* g1 = buf + xx1 + j*w + 1*w*h;
            mag_e8m23_t* g2 = buf + xx2 + j*w + 1*w*h;
            mag_e8m23_t* b1 = buf + xx1 + j*w + 2*w*h;
            mag_e8m23_t* b2 = buf + xx2 + j*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_get_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_get_data_size(t));
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
    mag_e8m23_t* buf = (mag_e8m23_t*)t->storage->base;
    int32_t w = (int32_t)mag_tensor_get_width(t);
    int32_t h = (int32_t)mag_tensor_get_height(t);
    int32_t c = (int32_t)mag_tensor_get_channels(t);
    mag_assert2(w && h && c == 3);
    mag_e8m23_t* pr = buf;
    mag_e8m23_t* pg = buf + w*h;
    mag_e8m23_t* pb = buf + w*h*2;
    mag_e8m23_t r = (mag_e8m23_t)((rgb>>16)&0xff) / 255.0f;
    mag_e8m23_t g = (mag_e8m23_t)((rgb>>8)&0xff) / 255.0f;
    mag_e8m23_t b = (mag_e8m23_t)(rgb&0xff) / 255.0f;
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

char* mag_tensor_to_string(const mag_tensor_t* t, bool with_header, size_t from_start_count, size_t from_end_count) {
    return malloc(3); /* TODO */
}

void mag_tensor_to_string_free_data(char* ret_val) {
    (*mag_alloc)(ret_val, 0); /* TODO: use scratch buffer */
}

mag_ctx_t* mag_tensor_get_ctx(const mag_tensor_t* t) { return t->ctx; }
void* mag_tensor_get_user_data(const mag_tensor_t* t) { return t->ud; }
void mag_tensor_set_user_data(mag_tensor_t* t, void* ud) { t->ud = ud; }
int64_t mag_tensor_get_width(const mag_tensor_t* t) { return t->shape[2]; }
int64_t mag_tensor_get_height(const mag_tensor_t* t) { return t->shape[1]; }
int64_t mag_tensor_get_channels(const mag_tensor_t* t) { return t->shape[0]; }

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

static void MAG_COLDPROC mag_machine_probe_os_name(char (*out_os_name)[128]) { /* Get OS name */
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

static void MAG_COLDPROC mag_machine_probe_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
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

static void MAG_COLDPROC mag_machine_probe_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
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

static void MAG_COLDPROC mag_machine_probe_memory(uint64_t* out_phys_mem_total, uint64_t* out_phys_mem_free) { /* Get physical memory */
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
    static void MAG_COLDPROC mag_system_info_query_amd64_cpu_caps(uint64_t* caps, bool* is_amd) {
        *caps = 0;
        uint32_t regs[8][4] = {0};

        #define H0 0
        #define H1 1
        #define H2 2
        #define H7 3
        #define H80000001 4
        #define H80000007 5
        #define H16 6
        #define H7_1H 7
        #define EAX 0
        #define EBX 1
        #define ECX 2
        #define EDX 3

        #define mag_cpy_regs(id) \
        regs[id][EAX] = eax; \
        regs[id][EBX] = ebx; \
        regs[id][ECX] = ecx; \
        regs[id][EDX] = edx

        #define _(enumerator, leaf, reg, shift) (0xff&leaf)
            static const uint8_t feature_leaves[MAG_AMD64_CAP__NUM] = {
                mag_x86_64_feature_def(_, MAG_SEP)
            };
        #undef _
        #define _(enumerator, leaf, reg, shift) (0xff&reg)
            static const uint8_t feature_regs[MAG_AMD64_CAP__NUM] = {
                mag_x86_64_feature_def(_, MAG_SEP)
            };
        #undef _
        #define _(enumerator, leaf, reg, shift) (1u<<(shift))
            static const uint32_t feature_masks[MAG_AMD64_CAP__NUM] = {
                mag_x86_64_feature_def(_, MAG_SEP)
            };
        #undef _
        #undef mag_x86_64_feature_def
        #undef _
        /* Detect features. */
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        uint32_t max_basic_leaf, max_extended_leaf;
        mag_cpuid(0, -1, &eax, &ebx, &ecx, &edx);
        mag_cpy_regs(H0);
        max_basic_leaf = eax;
        mag_cpuid(0x80000000u, -1, &eax, &ebx, &ecx, &edx);
        max_extended_leaf = eax;
        if (max_basic_leaf >= 1u) {
            mag_cpuid(1, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H1);
        }
        if (max_basic_leaf >= 2u) {
            mag_cpuid(2u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H2);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 0, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H7);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H7_1H);
        }
        if (max_basic_leaf >= 0x16u) {
            mag_cpuid(0x16u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H16);
        }
        if (max_extended_leaf >= 0x80000001u) {
            mag_cpuid(0x80000001u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H80000001);
        }
        if (max_extended_leaf >= 0x80000007u) {
            mag_cpuid(0x80000007u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(H80000007);
        }
        bool cpu_avx_support = !!(regs[H1][ECX] & 0x10000000u);
        bool cpu_osxsave_support = !!(regs[H1][ECX] & 0x8000000u);
        if (cpu_avx_support && cpu_osxsave_support) {
            uint64_t xcr0 = mag_xgetbv();
            if ((xcr0 & 0x6) != 0x6u) {
                regs[H1][ECX] &= ~0x10000000u; /* Clear AVX */
                regs[H7][EBX] &= ~0x20u; /* Clear AVX2 */
            }
            if ((xcr0 & 0xe0) != 0xe0u) { /* OS does not support AVX-512, clear AVX512 */
                regs[H7][EBX] &= ~0xdc230000u;
                regs[H7][ECX] &= ~0x5842u;
                regs[H7][EDX] &= ~0x10cu;
                regs[H7_1H][EAX] &= ~0x20u;
            }
        } else {
            regs[H1][ECX] &= ~0x10000000u;  /* Clear AVX */
            regs[H7][EBX] &= ~0x20u;        /* Clear AVX2 */
            regs[H7][EBX] &= ~0xdc230000u;  /* Clear AVX512 */
            regs[H7][ECX] &= ~0x5842u;      /* Clear AVX512 */
            regs[H7][EDX] &= ~0x10cu;       /* Clear AVX512 */
            regs[H7_1H][EAX] &= ~0x20u;     /* Clear AVX512 */
        }

        for (uint64_t i=1; i < MAG_AMD64_CAP__NUM; ++i) /* Create bitset of features */
            if (regs[feature_leaves[i]][feature_regs[i]] & feature_masks[i])
                *caps |= 1ull<<i;

        /* Check if AMD CPU using brand string. */
        char vendor[12+1];
        mag_cpuid(0, -1, &eax, &ebx, &ecx, &edx);
        ((uint32_t*)vendor)[0] = ebx;
        ((uint32_t*)vendor)[1] = edx;
        ((uint32_t*)vendor)[2] = ecx;
        vendor[sizeof(vendor)-1] = '\0';
        *is_amd = !strncmp(vendor, "AuthenticAMD", sizeof(vendor));

        #undef H0
        #undef H1
        #undef H2
        #undef H7
        #undef H80000001
        #undef H80000007
        #undef H16
        #undef H7_1H
        #undef EAX
        #undef EBX
        #undef ECX
        #undef EDX
        #undef mag_cpy_regs
    }

#elif defined(__aarch64__) || defined(_M_ARM64)
static void MAG_COLDPROC mag_system_info_query_arm64_cpu_caps(uint64_t* caps, int64_t* sve_width) {
    *caps = MAG_ARM64_CAP_NONE;
    #ifdef __linux__
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        if (hwcap & HWCAP_ASIMD) *caps |= 1ull<<MAG_ARM64_CAP_NEON;
        if (hwcap & HWCAP_ASIMDDP) *caps |= 1ull<<MAG_ARM64_CAP_DOTPROD;
        if (hwcap2 & HWCAP2_I8MM) *caps |= 1ull<<MAG_ARM64_CAP_I8MM;
        if (hwcap & HWCAP_FPHP) *caps |= 1ull<<MAG_ARM64_CAP_F16SCA;
        if (hwcap & HWCAP_ASIMDHP) *caps |= 1ull<<MAG_ARM64_CAP_F16VEC;
        if (hwcap2 & HWCAP2_BF16) *caps |= 1ull<<MAG_ARM64_CAP_BF16;
        if (hwcap & HWCAP_SVE) *caps |= 1ull<<MAG_ARM64_CAP_SVE;
        if (hwcap2 & HWCAP2_SVE2) *caps |= 1ull<<MAG_ARM64_CAP_SVE2;
        *sve_width = 0; /* NYI */
    #elif defined(_WIN32)
        if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE))
            *caps |= 1ull << MAG_ARM64_CAP_NEON;
        if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE))
            *caps |= 1ull << MAG_ARM64_CAP_DOTPROD;
        /* Other features not supported by IsProcessorFeaturePresent*/
        *sve_width = 0; /* NYI */
    #elif defined(__APPLE__)
        int sx = 0;
        size_t size = sizeof(sx);
        if (sysctlbyname("hw.optional.AdvSIMD", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_NEON;
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_DOTPROD;
        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_I8MM;
        if (sysctlbyname("hw.optional.arm.FEAT_FP16", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_F16SCA;
        if (sysctlbyname("hw.optional.AdvSIMD_HPFPCvt", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_F16VEC;
        if (sysctlbyname("hw.optional.arm.FEAT_BF16", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_BF16;
        if (sysctlbyname("hw.optional.arm.FEAT_SVE", &sx, &size, NULL, 0) != 0) sx = 0;
        if (sx) *caps |= 1ull<<MAG_ARM64_CAP_SVE;
        *sve_width = 0; /* NYI */
    #endif
}
#endif

static void MAG_COLDPROC mag_machine_probe(mag_ctx_t* ctx) {
    mag_machine_probe_os_name(&ctx->machine.os_name);
    mag_machine_probe_cpu_name(&ctx->machine.cpu_name);
    mag_machine_probe_cpu_cores(&ctx->machine.cpu_virtual_cores, &ctx->machine.cpu_physical_cores, &ctx->machine.cpu_sockets);
    mag_machine_probe_memory(&ctx->machine.phys_mem_total, &ctx->machine.phys_mem_free);
    #if defined(__x86_64__) || defined(_M_X64)
        mag_system_info_query_amd64_cpu_caps(&ctx->machine.amd64_cpu_caps, &ctx->machine.is_amd);
    #elif defined(__aarch64__)
        mag_system_info_query_arm64_cpu_caps(&ctx->machine.arm64_cpu_caps, &ctx->machine.arm64_cpu_sve_width);
    #endif
    if (mag_unlikely(!*ctx->machine.os_name)) snprintf(ctx->machine.os_name, sizeof(ctx->machine.os_name), "Unknown");
    if (mag_unlikely(!*ctx->machine.cpu_name)) snprintf(ctx->machine.cpu_name, sizeof(ctx->machine.cpu_name), "Unknown");
}

static MAG_COLDPROC void mag_graphviz_dump(const mag_tensor_t* node, FILE *fp, mag_hashset_t* visited) {
    if (mag_hashset_contains_key(visited, node)) return;
    mag_hashset_insert(visited, node);
    bool is_input = true;
    for (unsigned i = 0; i < MAG_MAX_OP_INPUTS; ++i) {
        if (node->op_inputs[i] != NULL) {
            is_input = false;
            break;
        }
    }
    const char* fillcolor = is_input ? "palegreen" : "skyblue2";
    char dim_buf[150];
    mag_fmt_dims(&dim_buf, &node->shape, node->rank);
    bool gra = node->flags & MAG_TFLAG_REQUIRES_GRAD;
    fprintf(
        fp,
        "  \"%p\" [label=\"⊕ %s|∇ %s|%s|0x%x\", shape=record, style=\"rounded,filled\", fillcolor=%s];\n",
        (void*)node,
        mag_op_meta_of(node->op)->mnemonic,\
        gra ? "✓" : "🗙",
        dim_buf,
        node->flags,
        fillcolor
    );
    for (unsigned i=0; i < MAG_MAX_OP_INPUTS; ++i) {
        mag_tensor_t* input = node->op_inputs[i];
        if (!input) continue;
        char name[128];
        if (*input->name) snprintf(name, sizeof(name), " in %u (%s)", i, input->name);
        else snprintf(name, sizeof(name), " in %u", i);
        fprintf(fp, "  \"%p\" -> \"%p\" [label=\"%s\"];\n", (void*)input, (void*)node, name);
        mag_graphviz_dump(input, fp, visited);
    }
}

MAG_COLDPROC void mag_tensor_export_forward_graph_graphviz(mag_tensor_t* t, const char* file) {
    mag_assert2(t && file && *file);
    FILE* f = mag_fopen(file, "w");
    fprintf(f, "digraph computation_graph {\n");
    fprintf(f, "  rankdir=TD;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", shape=box];\n");
    fprintf(f, "  edge [fontname=\"Helvetica\"];\n");
    mag_hashset_t visited = mag_hashset_init(0xffff);
    mag_graphviz_dump(t, f, &visited);
    mag_hashset_free(&visited);
    fprintf(f, "}\n");
    fclose(f);
}

MAG_COLDPROC void mag_tensor_export_backward_graph_graphviz(mag_tensor_t* t, const char* file) {
    mag_tensor_array_t post_order;
    mag_tensor_array_init(&post_order);
    mag_collect_topo_iterative(t, &post_order);
    for (size_t i=0, j=post_order.size - 1; i < j; ++i, --j) {
        mag_swap(mag_tensor_t*, post_order.data[i], post_order.data[j]);
    }
    FILE* fp = mag_fopen(file, "wt");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing the graphviz output.\n");
        return;
    }
    fprintf(fp, "digraph backward_graph {\n");
    fprintf(fp, "    rankdir=TD;\n");
    fprintf(fp, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t* node = post_order.data[i];
        const mag_op_meta_t* meta = mag_op_meta_of(node->op);
        fprintf(fp, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
        for (int r = 0; r < node->rank; ++r) {
            fprintf(fp, "%zu", (size_t)node->shape[r]);
            if (r < node->rank - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, ")\\nGrad: %s\"];\n", node->grad ? "set" : "none");
    }
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t* node = post_order.data[i];
        const mag_op_meta_t* meta = mag_op_meta_of(node->op);
        for (uint32_t j = 0; j < meta->num_inputs; ++j) {
            mag_tensor_t* input = node->op_inputs[j];
            if (input) {
                fprintf(fp, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
            }
        }
    }
    fprintf(fp, "}\n");
    fclose(fp);
    mag_tensor_array_free(&post_order);
}

static inline uint32_t mag_murmur_32_scramble(uint32_t k) {
    k *= 0xcc9e2d51;
    k = (k<<15) | (k>>17);
    k *= 0x1b873593;
    return k;
}

uint64_t mag_hash(const void* key, size_t len, uint32_t seed) {
    #define	mag_rol32(x, r) (((x)<<(r))|((x)>>(32-(r))))
    #define mag_mix32(h) h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16;
    const uint8_t* p = key;
    int64_t nblocks = (int64_t)len>>4;
    uint32_t h1 = seed;
    uint32_t h2 = seed;
    uint32_t h3 = seed;
    uint32_t h4 = seed;
    uint32_t c1 = 0x239b961b;
    uint32_t c2 = 0xab0e9789;
    uint32_t c3 = 0x38b34ae5;
    uint32_t c4 = 0xa1e38b93;
    const uint32_t * blocks = (const uint32_t *)(p + nblocks*16);
    for (int64_t i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i*4+0];
        uint32_t k2 = blocks[i*4+1];
        uint32_t k3 = blocks[i*4+2];
        uint32_t k4 = blocks[i*4+3];
        k1 *= c1; k1  = mag_rol32(k1,15); k1 *= c2; h1 ^= k1;
        h1 = mag_rol32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;
        k2 *= c2; k2  = mag_rol32(k2,16); k2 *= c3; h2 ^= k2;
        h2 = mag_rol32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;
        k3 *= c3; k3  = mag_rol32(k3,17); k3 *= c4; h3 ^= k3;
        h3 = mag_rol32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;
        k4 *= c4; k4  = mag_rol32(k4,18); k4 *= c1; h4 ^= k4;
        h4 = mag_rol32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
    }
    const uint8_t * tail = (const uint8_t*)(p + nblocks*16);
    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;
    switch(len&15) {
        case 15: k4 ^= tail[14] << 16;
        case 14: k4 ^= tail[13] << 8;
        case 13: k4 ^= tail[12] << 0;
            k4 *= c4;
            k4 = mag_rol32(k4,18);
            k4 *= c1;
            h4 ^= k4;
        case 12: k3 ^= tail[11] << 24;
        case 11: k3 ^= tail[10] << 16;
        case 10: k3 ^= tail[9] << 8;
        case 9: k3 ^= tail[8] << 0;
            k3 *= c3;
            k3 = mag_rol32(k3,17);
            k3 *= c4;
            h3 ^= k3;
        case 8: k2 ^= tail[7] << 24;
        case 7: k2 ^= tail[6] << 16;
        case 6: k2 ^= tail[5] << 8;
        case 5: k2 ^= tail[4] << 0;
            k2 *= c2;
            k2 = mag_rol32(k2,16);
            k2 *= c3;
            h2 ^= k2;
        case 4: k1 ^= tail[3] << 24;
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0] << 0;
            k1 *= c1;
            k1 = mag_rol32(k1,15);
            k1 *= c2;
            h1 ^= k1;
    };
    h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;
    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
    mag_mix32(h1); mag_mix32(h2); mag_mix32(h3); mag_mix32(h4);
    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
    return (((uint64_t)h2)<<32)|h1;
    #undef mag_rol32
    #undef mag_mix32
}

uint32_t mag_crc32c(const void* buffer, size_t size) {
    if (mag_unlikely(!buffer || !size)) return 0;
    const uint8_t* buf = buffer;
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
        crc = (crc>>8) ^ crc_lut[buf[i] ^ (crc&0xff)];
    return ~crc;
}

typedef struct mag_bucket_t {
    uint64_t hash : 48;
    uint64_t dib : 16;
} mag_bucket_t;
mag_static_assert(sizeof(mag_bucket_t) == 8);

struct mag_hashmap_t {
    size_t elsize;
    size_t cap;
    uint32_t seed;
    uint64_t (*hash)(const void* item, uint32_t seed);
    bool (*cmp)(const void* a, const void* b, void* ud);
    void (*elfree)(void* el);
    void* ud;
    size_t bucketsz;
    size_t nbuckets;
    size_t count;
    size_t mask;
    size_t growat;
    size_t shrinkat;
    uint8_t loadfactor;
    uint8_t growpower;
    bool oom;
    void* buckets;
    void* spare;
    void* edata;
    double grow_fac;
    double shrink_fac;
    double load_fac;
};

void mag_hashmap_set_grow_by_power(mag_hashmap_t* map, size_t pow) {
    map->growpower = pow < 1 ? 1 : pow > 16 ? 16 : pow;
}
static double mag_hashmap_clamp_load_factor(double fac, double def) {
    return isnan(fac) ? def : fac < 0.50 ? 0.50 : fac > 0.95 ? 0.95 : fac;
}
void mag_hashmap_set_load_factor(mag_hashmap_t* map, double factor) {
    factor = mag_hashmap_clamp_load_factor(factor, map->loadfactor/100.0);
    map->loadfactor = (uint8_t)(factor*100.0);
    map->growat = (size_t)((double)map->nbuckets*(map->loadfactor/100.0));
}
static mag_bucket_t* mag_hashmap_bucket_at0(void* buckets, size_t bucketsz, size_t i) { return (mag_bucket_t*)((char*)buckets + bucketsz*i); }
static mag_bucket_t* mag_hashmap_bucket_at(mag_hashmap_t* map, size_t index) { return mag_hashmap_bucket_at0(map->buckets, map->bucketsz, index); }
static void* mag_hashmap_bucket_item(mag_bucket_t* entry) { return (char*)entry+sizeof(mag_bucket_t); }
static uint64_t mag_hashmap_clip_hash(uint64_t hash) {return hash & 0xffffffffffff; }
static uint64_t mag_hashmap_get_hash(mag_hashmap_t* map, const void* key) { return mag_hashmap_clip_hash((*map->hash)(key, map->seed)); }

mag_hashmap_t* mag_hashmap_create(
    size_t elsize,
    size_t cap,
    uint32_t seed,
    uint64_t (*hash)(const void* item, uint32_t seed),
    bool (*cmp)(const void* a, const void* b, void *ud),
    void (*elfree)(void* el),
    void* ud,
    double grow_fac,
    double shrink_fac,
    double load_fac
) {
    grow_fac = grow_fac ? grow_fac : MAG_DEF_MAP_GROW_FACTOR;
    shrink_fac = shrink_fac ? shrink_fac : MAG_DEF_MAP_SHRINK_FACTOR;
    load_fac = load_fac ? load_fac : grow_fac;
    size_t ncap = 16;
    if (cap < ncap) cap = ncap;
    else {
        while (ncap < cap) ncap <<= 1;
        cap = ncap;
    }
    size_t bsz = sizeof(mag_bucket_t)+elsize;
    for (; bsz&7; ++bsz);
    size_t sz = sizeof(mag_hashmap_t)+(bsz<<1);
    mag_hashmap_t* map = (*mag_alloc)(NULL, sz);
    memset(map, 0, sizeof(mag_hashmap_t));
    map->elsize = elsize;
    map->bucketsz = bsz;
    map->seed = seed;
    map->hash = hash;
    map->cmp = cmp;
    map->elfree = elfree;
    map->ud = ud;
    map->spare = (char*)map+sizeof(mag_hashmap_t);
    map->edata = (char*)map->spare+bsz;
    map->cap = cap;
    map->nbuckets = cap;
    map->mask = map->nbuckets-1;
    map->grow_fac = grow_fac;
    map->shrink_fac = shrink_fac;
    map->load_fac = load_fac;
    map->buckets = (*mag_alloc)(NULL, map->bucketsz*map->nbuckets);
    memset(map->buckets, 0, map->bucketsz*map->nbuckets);
    map->growpower = 1;
    map->loadfactor = (uint8_t)(mag_hashmap_clamp_load_factor(map->load_fac, map->grow_fac) * 100.0);
    map->growat = (size_t)((double)map->nbuckets*(map->loadfactor / 100.0));
    map->shrinkat = (size_t)((double)map->nbuckets*map->shrink_fac);
    return map;
}

static void mag_hashmap_free_elems(mag_hashmap_t* map) {
    if (map->elfree) {
        for (size_t i=0; i < map->nbuckets; i++) {
            mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
            if (bucket->dib) map->elfree(mag_hashmap_bucket_item(bucket));
        }
    }
}

void mag_hashmap_clear(mag_hashmap_t* map, bool update_cap) {
    map->count = 0;
    mag_hashmap_free_elems(map);
    if (update_cap) {
        map->cap = map->nbuckets;
    } else if (map->nbuckets != map->cap) {
        void* nb = (*mag_alloc)(NULL, map->bucketsz*map->cap);
        if (nb) {
            (*mag_alloc)(map->buckets, 0);
            map->buckets = nb;
        }
        map->nbuckets = map->cap;
    }
    memset(map->buckets, 0, map->bucketsz*map->nbuckets);
    map->mask = map->nbuckets-1;
    map->growat = (size_t)((double)map->nbuckets*(map->loadfactor / 100.0));
    map->shrinkat = (size_t)((double)map->nbuckets*map->shrink_fac);
}

static bool mag_hashmap_resize0(mag_hashmap_t* map, size_t new_cap) {
    mag_hashmap_t* map2 = mag_hashmap_create(
        map->elsize,
        new_cap,
        map->seed,
        map->hash,
        map->cmp,
        map->elfree,
        map->ud,
        map->grow_fac,
        map->shrink_fac,
        map->load_fac
    );
    for (size_t i=0; i < map->nbuckets; i++) {
        mag_bucket_t* entry = mag_hashmap_bucket_at(map, i);
        if (!entry->dib) {
            continue;
        }
        entry->dib = 1;
        size_t j = entry->hash & map2->mask;
        while(1) {
            mag_bucket_t* bucket = mag_hashmap_bucket_at(map2, j);
            if (bucket->dib == 0) {
                memcpy(bucket, entry, map->bucketsz);
                break;
            }
            if (bucket->dib < entry->dib) {
                memcpy(map2->spare, bucket, map->bucketsz);
                memcpy(bucket, entry, map->bucketsz);
                memcpy(entry, map2->spare, map->bucketsz);
            }
            j = (j+1) & map2->mask;
            ++entry->dib;
        }
    }
    (*mag_alloc)(map->buckets, 0);
    map->buckets = map2->buckets;
    map->nbuckets = map2->nbuckets;
    map->mask = map2->mask;
    map->growat = map2->growat;
    map->shrinkat = map2->shrinkat;
    (*mag_alloc)(map2, 0);
    return true;
}

static bool mag_hashmap_resize(mag_hashmap_t* map, size_t new_cap) {
    return mag_hashmap_resize0(map, new_cap);
}

const void* mag_hashmap_set_with_hash(mag_hashmap_t* map, const void* item, uint64_t hash) {
    hash = mag_hashmap_clip_hash(hash);
    map->oom = false;
    if (map->count >= map->growat) {
        if (!mag_hashmap_resize(map, map->nbuckets*(1<<map->growpower))) {
            map->oom = true;
            return NULL;
        }
    }
    mag_bucket_t* entry = map->edata;
    entry->hash = hash;
    entry->dib = 1;
    void* eitem = mag_hashmap_bucket_item(entry);
    memcpy(eitem, item, map->elsize);
    void* bitem;
    size_t i = entry->hash&map->mask;
    for (;;) {
        mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
        if (bucket->dib == 0) {
            memcpy(bucket, entry, map->bucketsz);
            ++map->count;
            return NULL;
        }
        bitem = mag_hashmap_bucket_item(bucket);
        if (entry->hash == bucket->hash && (!map->cmp || (*map->cmp)(eitem, bitem, map->ud))) {
            memcpy(map->spare, bitem, map->elsize);
            memcpy(bitem, eitem, map->elsize);
            return map->spare;
        }
        if (bucket->dib < entry->dib) {
            memcpy(map->spare, bucket, map->bucketsz);
            memcpy(bucket, entry, map->bucketsz);
            memcpy(entry, map->spare, map->bucketsz);
            eitem = mag_hashmap_bucket_item(entry);
        }
        i = (i+1) & map->mask;
        ++entry->dib;
    }
}

const void* mag_hashmap_insert(mag_hashmap_t* map, const void* item) {
    return mag_hashmap_set_with_hash(map, item, mag_hashmap_get_hash(map, item));
}

const void *mag_hashmap_get_with_hash(mag_hashmap_t* map, const void* key, uint64_t hash) {
    hash = mag_hashmap_clip_hash(hash);
    size_t i = hash&map->mask;
    while(1) {
        mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
        if (!bucket->dib) return NULL;
        if (bucket->hash == hash) {
            void* bitem = mag_hashmap_bucket_item(bucket);
            if (!map->cmp || (*map->cmp)(key, bitem, map->ud)) {
                return bitem;
            }
        }
        i = (i+1) & map->mask;
    }
}

const void* mag_hashmap_lookup(mag_hashmap_t* map, const void* key) { return mag_hashmap_get_with_hash(map, key, mag_hashmap_get_hash(map, key)); }
const void* mag_hashmap_probe(mag_hashmap_t* map, uint64_t position) {
    size_t i = position & map->mask;
    mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
    if (!bucket->dib) {
        return NULL;
    }
    return mag_hashmap_bucket_item(bucket);
}

const void* mag_hashmap_delete_with_hash(mag_hashmap_t* map, const void* key, uint64_t hash) {
    hash = mag_hashmap_clip_hash(hash);
    map->oom = false;
    size_t i = hash&map->mask;
    while(1) {
        mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
        if (!bucket->dib) {
            return NULL;
        }
        void* bitem = mag_hashmap_bucket_item(bucket);
        if (bucket->hash == hash && (!map->cmp || (*map->cmp)(key, bitem, map->ud))) {
            memcpy(map->spare, bitem, map->elsize);
            bucket->dib = 0;
            while(1) {
                mag_bucket_t* prev = bucket;
                i = (i+1) & map->mask;
                bucket = mag_hashmap_bucket_at(map, i);
                if (bucket->dib <= 1) {
                    prev->dib = 0;
                    break;
                }
                memcpy(prev, bucket, map->bucketsz);
                prev->dib--;
            }
            map->count--;
            if (map->nbuckets > map->cap && map->count <= map->shrinkat) {
                mag_hashmap_resize(map, map->nbuckets>>1);
            }
            return map->spare;
        }
        i = (i+1) & map->mask;
    }
}

const void *mag_hashmap_delete(mag_hashmap_t* map, const void* key) {
    return mag_hashmap_delete_with_hash(map, key, mag_hashmap_get_hash(map, key));
}

size_t mag_hashmap_count(mag_hashmap_t* map) { return map->count; }

void mag_hashmap_destroy(mag_hashmap_t* map) {
    if (!map) return;
    mag_hashmap_free_elems(map);
    (*mag_alloc)(map->buckets, 0);
    (*mag_alloc)(map, 0);
}

bool mag_hashmap_is_oom(mag_hashmap_t* map) { return map->oom; }

bool mag_hashmap_scan(mag_hashmap_t* map, bool (*iter)(const void* item, void* ud), void* ud) {
    for (size_t i=0; i < map->nbuckets; i++) {
        mag_bucket_t* bucket = mag_hashmap_bucket_at(map, i);
        if (bucket->dib && !iter(mag_hashmap_bucket_item(bucket), ud)) {
            return false;
        }
    }
    return true;
}

bool mag_hashmap_iter(mag_hashmap_t* map, size_t* i, void** item) {
    mag_bucket_t* bucket;
    do {
        if (*i >= map->nbuckets) return false;
        bucket = mag_hashmap_bucket_at(map, *i);
        (*i)++;
    } while (!bucket->dib);
    *item = mag_hashmap_bucket_item(bucket);
    return true;
}

static const uint8_t* mag_utf8_validate(const uint8_t* restrict s, const uint8_t* restrict end) {
    const uint8_t* p = s;
    while (p < end) {
        uint8_t c = *p;
        if (c < 0x80) { ++p; continue; }
        if ((c & 0xe0) == 0xc0) {
            if (p+1 >= end) return p;
            uint8_t b1 = p[1];
            if ((b1 & 0xc0) != 0x80 || (c & 0xfe) == 0xc0) return p;
            p += 2;
        } else if ((c & 0xf0) == 0xe0) {
            if (p+2 >= end) return p;
            uint8_t b1 = p[1], b2 = p[2];
            if ((b1 & 0xc0) != 0x80 || (b2 & 0xc0) != 0x80) return p;
            if (c == 0xe0 && (b1 & 0xe0) == 0x80) return p;
            if (c == 0xeD && (b1 & 0xe0) == 0xa0) return p;
            p += 3;
        } else if ((c & 0xf8) == 0xf0) {
            if (p+3 >= end) return p;
            uint8_t b1 = p[1], b2 = p[2], b3 = p[3];
            if ((b1 & 0xc0) != 0x80 || (b2 & 0xc0) != 0x80 || (b3 & 0xc0) != 0x80) return p;
            if (c == 0xf0 && (b1 & 0xf0) == 0x80) return p;
            if ((c == 0xf4 && b1 > 0x8f) || c > 0xf4) return p;
            p += 4;
            continue;
        }
        return p;
    }
    return NULL;
}

static size_t mag_utf8_strlen(const uint8_t* str) {
    const uint8_t* s;
    for (s = str; *s; ++s);
    return s-str;
}

static uint8_t* mag_utf8_strclone(const char* s, size_t* out_len) {
    size_t n = strlen(s);
    uint8_t* u8 = (*mag_alloc)(NULL, n+1);
    memcpy(u8, s, n);
    u8[n] = '\0';
    if (out_len) *out_len = n;
    return u8;
}

#define mag_make_magic(a, b, c, d) ((((d)&0xff)<<24) + (((c)&0xff)<<16) + (((b)&0xff)<<8) + ((a)&0xff))
#define MAG_STO_FILE_MAGIC mag_make_magic('M', 'A', 'G', '&')
#define MAG_STO_TENSOR_HDR_SECTION mag_make_magic('T', 'H', 'D', 'R')
#define MAG_STO_TENSOR_DAT_SECTION mag_make_magic('T', 'D', 'A', 'T')
#define MAG_STO_TENSOR_KV_SECTION mag_make_magic('K', 'V', 'D', 'T')
mag_static_assert(sizeof(MAG_STO_FILE_MAGIC) == 4);
mag_static_assert(sizeof(MAG_STO_TENSOR_HDR_SECTION) == 4);
mag_static_assert(sizeof(MAG_STO_TENSOR_DAT_SECTION) == 4);
mag_static_assert(sizeof(MAG_STO_TENSOR_KV_SECTION) == 4);
#define MAG_STO_MAX_KEY_LEN 1024
#define MAG_STO_FILE_HDR_SIZE (4*6)
#define MAG_STO_TENSOR_HDR_SIZE (4 + 8*MAG_MAX_DIMS + 4 + MAG_MAX_TENSOR_NAME_LEN)
mag_static_assert(MAG_MAX_TENSOR_NAME_LEN % 8 == 0); /* Because we write it as multiple u64s. */
#define MAG_STO_TENSOR_KEY_HASH_SEED 0xa38876a9
#define mag_sto_sanitize(exp, msg, ret_stmt) do { if (mag_unlikely(!(exp))) { mag_log_error("magnetron storage sanitize error: " #exp " <- " msg); ret_stmt; } } while (0)

static bool mag_sto_write_u32_le(FILE* f, uint32_t v) {
    v = mag_bswap32(v);
    return fwrite(&v, sizeof(v), 1, f) == 1;
}
static bool mag_sto_write_u64_le(FILE* f, uint64_t v) {
    v = mag_bswap64(v);
    return fwrite(&v, sizeof(v), 1, f) == 1;
}
static bool mag_sto_read_u32_le(FILE* f, uint32_t* v) {
    if (mag_unlikely(fread(v, sizeof(*v), 1, f) != 1)) return false;
    *v = mag_bswap32(*v);
    return true;
}
static bool mag_sto_read_u64_le(FILE* f, uint64_t* v) {
    if (mag_unlikely(fread(v, sizeof(*v), 1, f) != 1)) return false;
    *v = mag_bswap64(*v);
    return true;
}

static bool mag_pack_aux32(uint32_t* out, uint8_t unused0, uint8_t unused1, int64_t rank, mag_dtype_t dtype) {
    mag_sto_sanitize(rank >= 1 && rank <= MAG_MAX_DIMS, "invalid rank", return false);
    mag_sto_sanitize(dtype >= 0 && dtype < MAG_DTYPE__NUM, "invalid dtype", return false);
    *out = ((uint32_t)(unused0&0xff)<<24) + ((uint32_t)(unused1&0xff)<<16) + ((uint32_t)(rank&0xff)<<8) + (uint32_t)(dtype&0xff);
    return true;
}
static bool mag_unpack_aux(uint32_t aux, uint8_t* unused0, uint8_t* unused1, int64_t* rank, mag_dtype_t* dtype) {
    *unused0 = (aux>>24)&0xff;
    *unused1 = (aux>>16)&0xff;
    *rank = (aux>>8)&0xff;
    *dtype = aux&0xff;
    mag_sto_sanitize(*rank >= 1 && *rank <= MAG_MAX_DIMS, "invalid rank", return false);
    mag_sto_sanitize(*dtype >= 0 && *dtype < MAG_DTYPE__NUM, "invalid dtype", return false);
    return true;
}

static void mag_bswap_block_le(void* dst, size_t numel, size_t granularity) {
    switch (granularity) {
        case 1: return; /* No need to swap bytes. */
        case 2: {
            uint16_t* d = (uint16_t*)dst;
            for (size_t i=0; i < numel; ++i)
                d[i] = mag_bswap16(d[i]);
        } return;
        case 4: {
            uint32_t* d = (uint32_t*)dst;
            for (size_t i=0; i < numel; ++i)
                d[i] = mag_bswap32(d[i]);
        } return;
        case 8: {
            uint64_t* d = (uint64_t*)dst;
            for (size_t i=0; i < numel; ++i)
                d[i] = mag_bswap64(d[i]);
        } return;
        default: mag_panic("invalid granularity"); return;
    }
}

static bool mag_sto_write_file_hdr(FILE* f, uint32_t num_tensors, uint32_t num_kv) {
    long start = ftell(f);
    mag_sto_sanitize(num_tensors < UINT32_MAX, "invalid num tensors", return false);
    mag_sto_sanitize(num_kv < UINT32_MAX, "invalid num kv", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, MAG_STO_FILE_MAGIC), "failed to write file magic", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, MAG_STORAGE_VERSION), "failed to write version", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, 0), "failed to write checksum", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, num_tensors), "failed to write num tensors", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, num_kv), "failed to write num kv", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, 0), "failed to write aux", return false);
    long end = ftell(f);
    mag_sto_sanitize(end - start == MAG_STO_FILE_HDR_SIZE, "invalid file header size", return false);
    return true;
}
static bool mag_sto_read_file_hdr(FILE* f, uint32_t* num_tensors, uint32_t* num_kv) {
    long start = ftell(f);
    uint32_t magic, version, checksum, aux;
    mag_sto_sanitize(mag_sto_read_u32_le(f, &magic), "failed to read file magic", return false);
    mag_sto_sanitize(magic == MAG_STO_FILE_MAGIC, "invalid file magic", return false);
    mag_sto_sanitize(mag_sto_read_u32_le(f, &version), "failed to read version", return false);
    mag_sto_sanitize(version != 0, "invalid version", return false);
    mag_sto_sanitize(mag_sto_read_u32_le(f, &checksum), "failed to read checksum", return false);
    mag_sto_sanitize(checksum == 0, "invalid checksum", return false);
    mag_sto_sanitize(mag_sto_read_u32_le(f, num_tensors), "failed to read num tensors", return false);
    mag_sto_sanitize(mag_sto_read_u32_le(f, num_kv), "failed to read num kv", return false);
    mag_sto_sanitize(mag_sto_read_u32_le(f, &aux), "failed to read aux", return false);
    mag_sto_sanitize(aux == 0, "invalid aux", return false);
    long end = ftell(f);
    mag_sto_sanitize(end - start == MAG_STO_FILE_HDR_SIZE, "invalid file header size", return false);
    return true;
}

static bool mag_sto_write_tensor_hdr(
    FILE* f,
    const uint8_t* key,
    size_t key_len,
    const int64_t(*shape)[MAG_MAX_DIMS],
    const uint8_t (*name)[MAG_MAX_TENSOR_NAME_LEN],
    int64_t rank,
    mag_dtype_t dtype
) {
    long start = ftell(f);
    mag_sto_sanitize(!mag_utf8_validate(key, key+key_len), "invalid utf8-8 in tensor key", return false);
    mag_sto_sanitize(key_len > 0 && key_len <= MAG_STO_MAX_KEY_LEN, "invalid key length", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, key_len), "failed to write key length", return false); /* Write key length */
    for (int i=0; i < MAG_MAX_DIMS; ++i) /* Write shape */
        mag_sto_sanitize(mag_sto_write_u64_le(f, (*shape)[i]), "failed to write shape", return false);
    uint32_t aux;
    mag_sto_sanitize(mag_pack_aux32(&aux, 0, 0, rank, dtype), "failed to pack aux", return false);
    mag_sto_sanitize(mag_sto_write_u32_le(f, aux), "failed to write aux", return false); /* Write aux */
    mag_sto_sanitize(!mag_utf8_validate(*name, *name+sizeof(name)), "invalid utf8-8 in tensor name", return false);
    mag_sto_sanitize(fwrite(name, MAG_MAX_TENSOR_NAME_LEN, 1, f) == 1, "failed to write name", return false); /* Write name */
    mag_sto_sanitize(fwrite(key, key_len, 1, f) == 1, "failed to write key", return false); /* Write key */
    long end = ftell(f);
    mag_sto_sanitize(end - start == MAG_STO_TENSOR_HDR_SIZE+key_len, "invalid tensor header size", return false);
    return true;
}

static bool mag_sto_read_tensor_hdr(
    FILE* f,
    uint8_t** key,
    int64_t(*shape)[MAG_MAX_DIMS],
    uint8_t (*name)[MAG_MAX_TENSOR_NAME_LEN],
    int64_t* rank,
    mag_dtype_t* dtype
) {
    long start = ftell(f);
    uint32_t key_len;
    mag_sto_sanitize(mag_sto_read_u32_le(f, &key_len), "failed to read key length", return false); /* Read key length */
    mag_sto_sanitize(key_len > 0 && key_len <= MAG_STO_MAX_KEY_LEN, "invalid key length", return false);
    for (int i=0; i < MAG_MAX_DIMS; ++i) { /* Read shape */
        uint64_t udim;
        mag_sto_sanitize(mag_sto_read_u64_le(f, &udim), "failed to read shape", return false);
        int64_t dim = (int64_t)udim;
        mag_sto_sanitize(dim > 0 && dim < INT64_MAX, "invalid shape dim", return false);
        (*shape)[i] = dim;
    }
    int64_t numel_total = (*shape)[0];
    for (int i=1; i < MAG_MAX_DIMS; ++i) {
        mag_sto_sanitize(!mag_imull64_ov(numel_total, (*shape)[i], &numel_total), "overflowing shape", return false);
    }
    mag_sto_sanitize(numel_total > 0 && numel_total < INT64_MAX, "invalid shape total", return false);
    uint32_t aux;
    mag_sto_sanitize(mag_sto_read_u32_le(f, &aux), "failed to read aux", return false);
    uint8_t unused0, unused1;
    mag_sto_sanitize(mag_unpack_aux(aux, &unused0, &unused1, rank, dtype), "failed to unpack aux", return false);
    mag_sto_sanitize(*rank >= 1 && *rank <= MAG_MAX_DIMS, "invalid rank", return false);
    mag_sto_sanitize(*dtype >= 0 && *dtype < MAG_DTYPE__NUM, "invalid dtype", return false);
    for (int i=0; i < MAG_MAX_TENSOR_NAME_LEN>>3; ++i) { /* todo read individual chars */
        uint64_t packed;
        mag_sto_sanitize(mag_sto_read_u64_le(f, &packed), "failed to read name", return false);
        memcpy(*name+(i<<3), &packed, sizeof(packed));
    }
    (*name)[MAG_MAX_TENSOR_NAME_LEN-1] = 0; /* Ensure null termination */
    mag_sto_sanitize(!mag_utf8_validate(*name, *name+sizeof(name)), "invalid utf8-8 in tensor name", return false);
    *key = (*mag_alloc)(NULL, key_len+1); /* Allocate key */
    mag_sto_sanitize(fread(*key, key_len, 1, f) == 1, "failed to read key", return false); /* Read key */
    (*key)[key_len] = '\0'; /* Ensure null termination */
    mag_sto_sanitize(*key, "empty key", false);
    mag_sto_sanitize(!mag_utf8_validate(*key, *key+key_len), "invalid utf8-8 in tensor key", return false);
    long end = ftell(f);
    mag_sto_sanitize(end - start == MAG_STO_TENSOR_HDR_SIZE+key_len, "invalid tensor header size", return false);
    return true;
}

static bool mag_sto_write_tensor_data(FILE* f, mag_tensor_t* t) {
    long start = ftell(f);
    size_t size = mag_tensor_get_data_size(t);
    if (mag_likely(size)) {
        void* data = mag_tensor_get_raw_data_as_bytes(t);
        #if MAG_BE /* Byteswap data if system is big-endian */
            mag_bswap_block_le(data, t->numel, mag_dtype_meta_of(t->dtype)->size);
        #endif
        mag_sto_sanitize(fwrite(data, size, 1, f) == 1, "failed to write tensor data", return false);
        mag_tensor_get_raw_data_as_bytes_free(data);
    }
    long end = ftell(f);
    mag_sto_sanitize(end - start == size, "invalid tensor data size", return false);
    return true;
}

static bool mag_sto_read_tensor_data(FILE* f, mag_tensor_t* t) {
    long start = ftell(f);
    size_t size = mag_tensor_get_data_size(t);
    if (mag_likely(size)) {
        void* data = (*mag_alloc)(NULL, size);
        mag_sto_sanitize(fread(data, size, 1, f) == 1, "failed to read tensor data", return false);
        #if MAG_BE /* Byteswap data if system is big-endian */
            mag_bswap_block_le(data, t->numel, mag_dtype_meta_of(t->dtype)->size);
        #endif
        mag_storage_buffer_t* sto = t->storage;
        (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CPY, 0, data, size);
        (*mag_alloc)(data, 0);
    }
    long end = ftell(f);
    mag_sto_sanitize(end - start == size, "invalid tensor data size", return false);
    return true;
}

mag_static_assert(sizeof(char) == sizeof(uint8_t));

typedef struct mag_sto_tensor_kv_t {
    const uint8_t* const key;
    const size_t key_len;
    mag_tensor_t*  tensor;
} mag_sto_tensor_kv_t;

static uint64_t mag_sto_tensor_kv_hash(const void* el, uint32_t seed) {
    mag_sto_tensor_kv_t* kv = (mag_sto_tensor_kv_t*)el;
    return mag_hash(kv->key, kv->key_len, seed);
}

static bool mag_sto_tensor_kv_cmp(const void* a, const void *b, void* ud) {
    (void)ud;
    const mag_sto_tensor_kv_t* kva = (const mag_sto_tensor_kv_t*)a;
    const mag_sto_tensor_kv_t* kvb = (const mag_sto_tensor_kv_t*)b;
    return kva->key_len == kvb->key_len && memcmp(kva->key, kvb->key, kva->key_len) == 0;
}

static void mag_sto_tensor_kv_free(void* el) {
    mag_sto_tensor_kv_t* kv = (mag_sto_tensor_kv_t*)el;
    (*mag_alloc)((void*)kv->key, 0);
    mag_tensor_decref(kv->tensor);
}

struct mag_storage_stream_t {
    mag_ctx_t* ctx;
    mag_hashmap_t* tensors;
};

mag_storage_stream_t* mag_storage_stream_new(mag_ctx_t* ctx) {
    mag_assert2(ctx);
    mag_storage_stream_t* stream = (*mag_alloc)(NULL, sizeof(*stream));
    stream->ctx = ctx;
    stream->tensors = mag_hashmap_create(
        sizeof(mag_sto_tensor_kv_t),
        32,
        MAG_STO_TENSOR_KEY_HASH_SEED,
        &mag_sto_tensor_kv_hash,
        &mag_sto_tensor_kv_cmp,
        &mag_sto_tensor_kv_free,
        NULL,
        MAG_DEF_MAP_GROW_FACTOR,
        MAG_DEF_MAP_SHRINK_FACTOR,
        MAG_DEF_MAP_LOAD_FACTOR
    );
    return stream;
}

void mag_storage_stream_close(mag_storage_stream_t* st) {
    mag_hashmap_destroy(st->tensors);
    (*mag_alloc)(st, 0);
}

bool mag_storage_stream_serialize(mag_storage_stream_t* st, const char* path) {
    FILE* f = mag_fopen(path, "wb"); /* TODO: is not closed by santinize */
    mag_tensor_t** ord = NULL;
    mag_sto_sanitize(f, "failed to open file for writing", return false);
    mag_sto_sanitize(mag_hashmap_count(st->tensors) <= UINT32_MAX, "invalud num tensors", goto cleanup); /* We should never have more than 4B tensors (haha) */
    mag_sto_sanitize(mag_sto_write_file_hdr(f, (uint32_t)mag_hashmap_count(st->tensors), 0), "failed to write file header", goto cleanup);
    mag_sto_sanitize(mag_sto_write_u32_le(f, MAG_STO_TENSOR_KV_SECTION), "failed to write kv section marker", goto cleanup);
    /* TODO: kv */
    mag_sto_sanitize(mag_sto_write_u32_le(f, MAG_STO_TENSOR_HDR_SECTION), "failed to write tensor header section marker", goto cleanup);
    size_t ord_num=0, ord_cap=32;
    ord = (*mag_alloc)(NULL, ord_cap*sizeof(*ord)); /* Ordered tensor list to preserve insertion order */
    size_t i=0;
    void* el;
    while (mag_hashmap_iter(st->tensors, &i, &el)) { /* Write tensor headers */
        const mag_sto_tensor_kv_t* bucket = (const mag_sto_tensor_kv_t*)el;
        mag_tensor_t* tensor = bucket->tensor;
        mag_sto_sanitize(
            mag_sto_write_tensor_hdr(f, bucket->key, bucket->key_len, &tensor->shape, &tensor->name, tensor->rank, tensor->dtype),
            "failed to write tensor header",
            goto cleanup
        );
        if (ord_num == ord_cap)
            ord = (*mag_alloc)(ord, (ord_cap<<=1)*sizeof(*ord));
        ord[ord_num++] = tensor;
    }
    /* Write tensor data section */
    mag_sto_sanitize(mag_sto_write_u32_le(f, MAG_STO_TENSOR_DAT_SECTION), "failed to write tensor data section marker", goto cleanup);
    for (i=0; i < ord_num; ++i) {
        mag_sto_sanitize(mag_sto_write_tensor_data(f, ord[i]), "failed to write tensor data", goto cleanup);
    }
    (*mag_alloc)(ord, 0);
    fclose(f);
    return true;
    cleanup:
        if (ord) (*mag_alloc)(ord, 0);
        fclose(f);
        return false;
}

mag_storage_stream_t* mag_storage_stream_deserialize(mag_ctx_t* ctx, const char* file) {
    FILE* f = mag_fopen(file, "rb");
    mag_tensor_t** ord = NULL;
    uint8_t* key = NULL;
    mag_sto_sanitize(f, "failed to open file for reading", return NULL);
    mag_storage_stream_t* stream = mag_storage_stream_new(ctx);
    uint32_t num_tensors, num_kv;
    mag_sto_sanitize(mag_sto_read_file_hdr(f, &num_tensors, &num_kv), "failed to read file header", goto cleanup);
    mag_sto_sanitize(num_tensors < UINT32_MAX, "invalid number of tensors", goto cleanup);
    mag_sto_sanitize(num_kv < UINT32_MAX, "invalid number of kv", goto cleanup);
    uint32_t tmp_marker; /* Section marker */
    mag_sto_sanitize(mag_sto_read_u32_le(f, &tmp_marker), "failed to read kv section marker", goto cleanup);
    mag_sto_sanitize(tmp_marker == MAG_STO_TENSOR_KV_SECTION, "invalid kv section marker", goto cleanup);
    /* TODO: kv */
    mag_sto_sanitize(mag_sto_read_u32_le(f, &tmp_marker), "failed to read tensor header section marker", goto cleanup);
    mag_sto_sanitize(tmp_marker == MAG_STO_TENSOR_HDR_SECTION, "invalid tensor header section marker", goto cleanup);
    ord = (*mag_alloc)(NULL, num_tensors*sizeof(*ord)); /* Ordered tensor list to preserve insertion order */
    for (uint32_t i=0; i < num_tensors; ++i) {
        int64_t shape[MAG_MAX_DIMS] = {};
        uint8_t name[MAG_MAX_TENSOR_NAME_LEN] = {};
        int64_t rank = 0;
        mag_dtype_t dtype = MAG_DTYPE__NUM;
        mag_sto_sanitize(mag_sto_read_tensor_hdr(f, &key, &shape, &name, &rank, &dtype), "failed to read tensor header", goto cleanup);
        mag_sto_sanitize(rank >= 1 && rank <= MAG_MAX_DIMS, "invalid tensor rank", goto cleanup);
        mag_tensor_t* tensor = mag_tensor_empty(ctx, dtype, rank, shape);
        mag_sto_sanitize(tensor, "failed to create tensor", goto cleanup);
        ord[i] = tensor;
        mag_tensor_set_name(tensor, (const char*)name);
        mag_sto_sanitize(mag_storage_stream_put_tensor(stream, (const char*)key, tensor), "failed to put tensor", goto cleanup);
        mag_tensor_decref(tensor); /* The function above increments the refcount and we also retain wich get_tensor, so we decref by one. */
        (*mag_alloc)(key, 0); /* Free key */
        key = NULL;
    }
    mag_sto_sanitize(mag_sto_read_u32_le(f, &tmp_marker), "failed to read tensor data section marker", goto cleanup);
    mag_sto_sanitize(tmp_marker == MAG_STO_TENSOR_DAT_SECTION, "invalid tensor data section marker", goto cleanup);
    for (uint32_t i=0; i < num_tensors; ++i)
        mag_sto_sanitize(mag_sto_read_tensor_data(f, ord[i]), "failed to read tensor data", goto cleanup);
    fclose(f);
    (*mag_alloc)(ord, 0);
    return stream;
    cleanup:
        if (ord) (*mag_alloc)(ord, 0);
        if (key) (*mag_alloc)(key,0);
        if (stream) mag_storage_stream_close(stream);
        fclose(f);
        return NULL;
}

bool mag_storage_stream_put_tensor(mag_storage_stream_t* st, const char* key, mag_tensor_t* t) {
    if (mag_unlikely(!key || !*key || !t)) return false;
    mag_tensor_incref(t);
    size_t len;
    uint8_t* u8key = mag_utf8_strclone(key, &len);
    const void* prev = mag_hashmap_insert(st->tensors, &(mag_sto_tensor_kv_t) {
        .key = u8key,
        .key_len = len,
        .tensor = t
    });
    if (mag_unlikely(prev)) {
        mag_log_error("Tensor with key '%s' already exists", key);
        return false;
    }
    return true;
}

mag_tensor_t* mag_storage_stream_get_tensor(mag_storage_stream_t* st, const char* key) {
    if (mag_unlikely(!key || !*key)) return NULL;
    const void* el = mag_hashmap_lookup(st->tensors, &(mag_sto_tensor_kv_t) {
        .key = (const uint8_t*)key,
        .key_len = strlen(key),
    });
    if (el) {
        mag_sto_tensor_kv_t* kv = (mag_sto_tensor_kv_t*)el;
        mag_tensor_incref(kv->tensor);
        return kv->tensor;
    }
    return NULL;
}

const char** mag_storage_stream_get_all_tensor_keys(mag_storage_stream_t* st, size_t* count) {
    void* el;
    *count = mag_hashmap_count(st->tensors);
    char** keys = (*mag_alloc)(NULL, (*count+1)*sizeof(*keys));
    size_t j=0;
    for (size_t i=0; mag_hashmap_iter(st->tensors, &i, &el); ++j) {
        mag_sto_tensor_kv_t* kv = (mag_sto_tensor_kv_t*)el;
        char* clone = (*mag_alloc)(NULL, kv->key_len+1);
        memcpy(clone, kv->key, kv->key_len);
        clone[kv->key_len] = '\0';
        keys[j] = clone;
    }
    keys[*count] = NULL; /* Null-terminate the array */
    return (const char**)keys;
}

void mag_storage_stream_get_all_tensor_keys_free_data(const char** ret_val) {
    for (const char** key = ret_val; *key; ++key)
        (*mag_alloc)((void*)*key, 0);
    (*mag_alloc)(ret_val, 0);
}

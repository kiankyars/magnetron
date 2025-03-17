/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef MAGNETRON_H
#define MAGNETRON_H

#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_DEFAULT_CHUNK_SIZE (1ull<<30)  /* Default size of memory chunk in bytes. 1 GiB */
#define MAG_DEFAULT_CHUNK_CAP 128          /* Default capacity of memory chunk */
#define MAG_MAX_DIMS 6                     /* Maximum number of dimensions for a tensor */
#define MAG_MAX_TENSOR_NAME_LEN 64         /* Maximum length for tensor name */
#define MAG_MAX_INPUT_TENSORS 2            /* Maximum number of input tensors for an operation */
#define MAG_MAX_OP_PARAMS 6                /* Maximum number of parameters for an operation */

#ifndef MAG_EXPORT
#ifdef MAG_SHARED
#ifdef _MSC_VER
#define MAG_EXPORT __declspec(dllexport)
#else
#define MAG_EXPORT __attribute__((visibility("default")))
#endif
#else
#define MAG_EXPORT
#endif
#endif
#if defined(_MSC_VER) || !__has_feature(nullability)
#ifndef _Nonnull
#define _Nonnull
#endif
#ifndef _Nullable
#define _Nullable
#endif
#endif

#define mag_version_pack(maj, mi) ((uint32_t)((((maj)&255)<<8)+((mi)&255)))
#define mag_version_major(ver) (((ver)>>8)&255)
#define mag_version_minor(ver) ((ver)&255)
#define MAG_VERSION mag_version_pack(0, 3) /* magnetron library version. */
#define MAG_STORAGE_VERSION 1 /* magnetron storage format version. */

typedef enum mag_compute_device_type_t {
    MAG_COMPUTE_DEVICE_TYPE_CPU = 0, /* CPU compute device */
    MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA = 1,  /* CUDA GPU compute device */

    MAG_COMPUTE_DEVICE_TYPE__NUM
} mag_compute_device_type_t;
extern MAG_EXPORT const char* _Nonnull mag_device_type_get_name(mag_compute_device_type_t op);

typedef enum mag_exec_mode_t {
    MAG_EXEC_MODE_EAGER = 0,        /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
    MAG_EXEC_MODE_DEFERRED = 1,     /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */

    MAG_EXEC_MODE__NUM
} mag_exec_mode_t;

typedef enum mag_prng_algorithm_t {
    MAG_PRNG_MERSENNE_TWISTER = 0,  /* Mersenne Twister PRNG */
    MAG_PRNG_PCG = 1,               /* Permuted Congruential Generator PRNG */

    MAG_PRNG__NUM
} mag_prng_algorithm_t;

typedef enum mag_thread_sched_prio_t {  /* Thread scheduling priority for CPU compute */
    MAG_THREAD_SCHED_PRIO_NORMAL = 0,   /* Normal thread priority */
    MAG_THREAD_SCHED_PRIO_MEDIUM = 1,   /* Medium thread priority */
    MAG_THREAD_SCHED_PRIO_HIGH = 2,     /* High thread priority */
    MAG_THREAD_SCHED_PRIO_REALTIME = 3, /* Real-time thread priority */
} mag_thread_sched_prio_t;

typedef enum mag_color_channels_t {
    MAG_COLOR_CHANNELS_AUTO,    /* Automatically detect number of color channels */
    MAG_COLOR_CHANNELS_GRAY,    /* Grayscale F32 */
    MAG_COLOR_CHANNELS_GRAY_A,  /* Grayscale F32 + Alpha F32 */
    MAG_COLOR_CHANNELS_RGB,     /* R32G32B32 */
    MAG_COLOR_CHANNELS_RGBA,    /* R32G32B32A32 */

    MAG_COLOR_CHANNELS__NUM
} mag_color_channels_t;

extern MAG_EXPORT void* _Nonnull (*_Nonnull mag_get_alloc_fn(void))(void* _Nullable blk, size_t size); /* Get global allocator. */
extern MAG_EXPORT void mag_set_alloc_fn(void* _Nonnull (*_Nonnull alloc)(void* _Nullable blk, size_t size)); /* Set global allocator. */
extern MAG_EXPORT void mag_set_log_mode(bool enabled); /* Enable/disable logging. */

typedef struct mag_ctx_t mag_ctx_t; /* Opaque context type for managing memory pools */

typedef struct mag_device_descriptor_t {
    mag_compute_device_type_t type;         /* Device type */
    uint32_t thread_count;                  /* Number of threads if type == MAG_COMPUTE_DEVICE_TYPE_CPU. If set to 0, hardware concurrency of host CPU is detected. */
    uint32_t cuda_device_id;                /* CUDA device ID if type == MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA. Default: 0 (first GPU). */
} mag_device_descriptor_t;

extern MAG_EXPORT mag_ctx_t* _Nonnull mag_ctx_create(mag_compute_device_type_t device);                                     /* Create context with default config, and only specify device type. */
extern MAG_EXPORT mag_ctx_t* _Nonnull mag_ctx_create2(const mag_device_descriptor_t* _Nonnull device_info);                 /* Create context with customized device config, and only specify device type. */
extern MAG_EXPORT mag_exec_mode_t mag_ctx_get_exec_mode(const mag_ctx_t* _Nonnull ctx);                                     /* Get execution mode */
extern MAG_EXPORT void mag_ctx_set_exec_mode(mag_ctx_t* _Nonnull ctx, mag_exec_mode_t mode);                                /* Set execution mode */
extern MAG_EXPORT mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* _Nonnull ctx);                           /* Get PRNG algorithm */
extern MAG_EXPORT void mag_ctx_set_prng_algorithm(mag_ctx_t* _Nonnull ctx, mag_prng_algorithm_t algorithm, uint64_t seed);  /* Set PRNG algorithm */
extern MAG_EXPORT mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* _Nonnull ctx);                 /* Get compute device type */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_compute_device_name(const mag_ctx_t* _Nonnull ctx);                      /* Get the name of the compute device */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_os_name(const mag_ctx_t* _Nonnull ctx);                                  /* Get the name of the operating system */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_cpu_name(const mag_ctx_t* _Nonnull ctx);                                 /* Get the name of the CPU */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* _Nonnull ctx);                                    /* Get the number of virtual cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* _Nonnull ctx);                                   /* Get the number of physical cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* _Nonnull ctx);                                          /* Get the number of CPU sockets */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* _Nonnull ctx);                                /* Get the total physical memory in bytes */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* _Nonnull ctx);                                 /* Get the free physical memory in bytes */
extern MAG_EXPORT bool mag_ctx_is_numa_system(const mag_ctx_t* _Nonnull ctx);                                               /* Check if the system is NUMA */
extern MAG_EXPORT size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* _Nonnull ctx);                                  /* Get total tensors created. (Including views) */
extern MAG_EXPORT void mag_ctx_profiler_start(mag_ctx_t* _Nonnull ctx);                                                     /* Start profiling */
extern MAG_EXPORT void mag_ctx_profiler_end(mag_ctx_t* _Nonnull ctx, const char* _Nullable export_csv_file);                /* Reset profiling data */
extern MAG_EXPORT bool mag_ctx_profiler_is_running(const mag_ctx_t* _Nonnull ctx);                                          /* Check if profiler is running */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_ctx_t* _Nonnull ctx);                                                /* Start gradient recording */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_ctx_t* _Nonnull ctx);                                                 /* Stop gradient recording */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_ctx_t* _Nonnull ctx);                                     /* Check if gradient recording is running */
extern MAG_EXPORT void mag_ctx_destroy(mag_ctx_t* _Nonnull ctx);                                                            /* Destroy context and free memory */

/**
 * @brief Multidimensional tensor of arbitrary rank and data type.
 *      The tensor is reference counted and can be shared between multiple tensors.
 *      Rule of Thumb for Reference Counting:
 *          - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *          - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 *      The rank is > 0 and <= MAG_MAX_DIMS. The shape of the tensor is an array of dimensions of size MAG_MAX_DIMS.
 *      Is a node in a static or dynamic computation graph, depending on the context execution mode.
 */
typedef struct mag_tensor_t mag_tensor_t;

typedef enum mag_dtype_t {
    MAG_DTYPE_E8M23,        /* IEEE-754 32-bit floating point number */
    MAG_DTYPE_E5M10,        /* IEEE-754 16-bit floating point number */
    MAG_DTYPE__NUM          /* Total number of data types */
} mag_dtype_t;

typedef struct mag_dtype_meta_t {
    int64_t size;                   /* Size of the data type in bytes */
    const char* _Nonnull name;      /* Name of the data type */
} mag_dtype_meta_t;
extern MAG_EXPORT const mag_dtype_meta_t* _Nonnull mag_dtype_meta_of(mag_dtype_t type);

extern MAG_EXPORT uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b);
extern MAG_EXPORT uint32_t mag_pack_color_f32(float r, float g, float b);

typedef enum mag_graph_eval_order_t {
    MAG_GRAPH_EVAL_ORDER_FORWARD = 0,   /* Evaluate graph from left to right */
    MAG_GRAPH_EVAL_ORDER_REVERSE = 1    /* Evaluate graph from right to left */
} mag_graph_eval_order_t;

/**
 * @brief Create a new 1-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_1d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1);

/**
 * @brief Create a new 2-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_2d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1, int64_t d2);

/**
 * @brief Create a new 3-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_3d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3);

/**
 * @brief Create a new 4-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_4d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);

/**
 * @brief Create a new 5-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @param d5 Size of the fifth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_5d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5);

/**
 * @brief Create a new 6-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @param d5 Size of the fifth dimension. Must be > 0 and < INT64_MAX.
 * @param d6 Size of the sixth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_create_6d(mag_ctx_t* _Nonnull ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6);

extern MAG_EXPORT mag_tensor_t* _Nonnull mag_clone(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_view(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_transpose(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_permute(mag_tensor_t* _Nonnull x, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t d4, uint32_t d5);

extern MAG_EXPORT mag_tensor_t* _Nonnull mag_mean(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_min (mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_max (mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sum (mag_tensor_t* _Nonnull x);

extern MAG_EXPORT mag_tensor_t* _Nonnull mag_abs(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_abs_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_neg(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_neg_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_log(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_log_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sqr(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sqr_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sqrt(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sqrt_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sin(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sin_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_cos(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_cos_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_step(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_step_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_exp(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_exp_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_softmax(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_softmax_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_softmax_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_softmax_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sigmoid(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sigmoid_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sigmoid_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sigmoid_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_hard_sigmoid(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_hard_sigmoid_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_silu(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_silu_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_silu_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_silu_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tanh(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tanh_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tanh_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tanh_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_relu(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_relu_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_relu_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_relu_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_gelu(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_gelu_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_gelu_dv(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_gelu_dv_(mag_tensor_t* _Nonnull x);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_add(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_add_(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sub( mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_sub_(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_mul( mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_mul_(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_div(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_div_(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_adds(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_adds_(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_subs( mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_subs_(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_muls(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_muls_(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_divs(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_divs_(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_pows(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_pows_(mag_tensor_t* _Nonnull x, float xi);
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_matmul(mag_tensor_t* _Nonnull x, mag_tensor_t* _Nonnull y);

/**
 * @brief Increment reference count of tensor.
 *      Increment the strong reference count of the tensor. The tensor is not destroyed until the strong reference count reaches zero.
 *      Rule of Thumb for Reference Counting:
 *      - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *      - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 * @param t Tensor. Must not be NULL.
 */
extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t* _Nonnull t);

/**
 * @brief Decrement reference count of tensor.
 *      Decrement the strong reference count of the tensor. The tensor is destroyed when the strong reference count reaches zero.
 *      Rule of Thumb for Reference Counting:
 *      - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *      - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 * @param t Tensor. Must not be NULL.
 * @returns True if the tensor was destroyed, false if the tensor is still alive.
 */
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t* _Nonnull t);

extern MAG_EXPORT void mag_tensor_copy_buffer_from(mag_tensor_t* _Nonnull t, const void* _Nonnull data, size_t size);   /* Copy data into tensor buffer */
extern MAG_EXPORT void mag_tensor_fill(mag_tensor_t* _Nonnull t, float x);                                              /* Set all tensor elements to a specific value */
extern MAG_EXPORT void mag_tensor_fill_random_uniform(mag_tensor_t* _Nonnull t, float min, float max);                  /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_normal(mag_tensor_t* _Nonnull t, float mean, float stddev);               /* Fill tensor with random values from the normal distribution. */

extern MAG_EXPORT uint64_t mag_tensor_get_packed_refcounts(const mag_tensor_t* _Nonnull t);                             /* Return strong refcount is loword, weak refcount is hiword. */
extern MAG_EXPORT void mag_tensor_retain(mag_tensor_t* _Nonnull t);                                                     /* Increment refcount */
extern MAG_EXPORT size_t mag_tensor_get_memory_usage(const mag_tensor_t* _Nonnull t);                                   /* Return memory used by this tensor in bytes. */
extern MAG_EXPORT void mag_tensor_print(const mag_tensor_t* _Nonnull t, bool with_header, bool with_data);              /* Print tensor info (with or without data) */
extern MAG_EXPORT void mag_tensor_set_name(mag_tensor_t* _Nonnull t, const char* _Nonnull name);                        /* Set the name of the tensor */
extern MAG_EXPORT void mag_tensor_fmt_name(mag_tensor_t* _Nonnull t, const char* _Nonnull fmt, ...);                    /* Format the name of the tensor */
extern MAG_EXPORT const char* _Nonnull mag_tensor_get_name(const mag_tensor_t* _Nonnull t);                             /* Get the name of the tensor */
extern MAG_EXPORT int64_t mag_tensor_rank(const mag_tensor_t* _Nonnull t);                                              /* Get the rank (number of dimensions) of the tensor */
extern MAG_EXPORT const int64_t* _Nonnull mag_tensor_shape(const mag_tensor_t* _Nonnull t);                             /* Get the dimensions of the tensor */
extern MAG_EXPORT const int64_t* _Nonnull mag_tensor_strides(const mag_tensor_t* _Nonnull t);                           /* Get the strides of the tensor */
extern MAG_EXPORT mag_dtype_t mag_tensor_dtype(const mag_tensor_t* _Nonnull t);                                         /* Get the data type of the tensor */
extern MAG_EXPORT void* _Nonnull mag_tensor_data_ptr(const mag_tensor_t* _Nonnull t);                                   /* Get the tensor raw buffer pointer. Might pointer to GPU or any other device memory. */
extern MAG_EXPORT int64_t mag_tensor_data_size(const mag_tensor_t* _Nonnull );                                          /* Get the size of the tensor buffer in bytes. */
extern MAG_EXPORT int64_t mag_tensor_numel(const mag_tensor_t* _Nonnull t);                                             /* Get the total amount of elements in the tensor. */
extern MAG_EXPORT int64_t mag_tensor_num_rows(const mag_tensor_t* _Nonnull t);                                          /* Get the number of rows (for 2D tensors) */
extern MAG_EXPORT int64_t mag_tensor_num_cols(const mag_tensor_t* _Nonnull t);                                          /* Get the number of columns (for 2D tensors) */
extern MAG_EXPORT bool mag_tensor_is_scalar(const mag_tensor_t* _Nonnull t);                                            /* Check if the tensor is a scalar */
extern MAG_EXPORT bool mag_tensor_is_vector(const mag_tensor_t* _Nonnull t);                                            /* Check if the tensor is a vector */
extern MAG_EXPORT bool mag_tensor_is_matrix(const mag_tensor_t* _Nonnull t);                                            /* Check if the tensor is a matrix */
extern MAG_EXPORT bool mag_tensor_is_volume(const mag_tensor_t* _Nonnull t);                                            /* Check if the tensor is higher-order (3D or more) */
extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_tensor_t* _Nonnull x, const mag_tensor_t* _Nonnull y);          /* Checks if a and b have the same shape. */
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_tensor_t* _Nonnull x, const mag_tensor_t* _Nonnull y);       /* Checks if a and b have the same strides. */
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_tensor_t* _Nonnull x, const mag_tensor_t* _Nonnull y);        /* Checks if b can be broadcasted into a. */
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_tensor_t* _Nonnull t);                                        /* Check if the tensor is transposed */
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_tensor_t* _Nonnull t);                                          /* Check if the tensor is permuted */
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_tensor_t* _Nonnull t);                                        /* Check if the tensor memory is contiguous */

extern MAG_EXPORT mag_tensor_t* _Nullable mag_tensor_grad(const mag_tensor_t* _Nonnull t);                              /* Get the gradient tensor of the tensor */
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_tensor_t* _Nonnull t);                                        /* Check if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_set_requires_grad(mag_tensor_t* _Nonnull t, bool requires_grad);                      /* Set if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_backward(mag_tensor_t* _Nonnull t);                                                   /* Compute the gradient of the tensor */
extern MAG_EXPORT void mag_tensor_zero_grad(mag_tensor_t* _Nonnull t);                                                  /* Zero the gradient of the tensor */

extern MAG_EXPORT float mag_tensor_get_scalar_physical_index(mag_tensor_t* _Nonnull t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5);                             /* Get scalar value at physical index */
extern MAG_EXPORT void mag_tensor_set_scalar_physical_index(mag_tensor_t* _Nonnull t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x);                     /* Set scalar value at physical index */
extern MAG_EXPORT float mag_tensor_get_scalar_virtual_index(mag_tensor_t* _Nonnull t, int64_t v_idx);                                                                                       /* Get scalar value at virtual index */
extern MAG_EXPORT void mag_tensor_set_scalar_virtual_index(mag_tensor_t* _Nonnull t, int64_t v_idx, float x);                                                                               /* Set scalar value at virtual indexs */
extern MAG_EXPORT bool mag_tensor_eq(const mag_tensor_t* _Nonnull x, const mag_tensor_t* _Nonnull y);                                                                                       /* Check if two tensors are equal without epsilon. */
extern MAG_EXPORT bool mag_tensor_is_close(const mag_tensor_t* _Nonnull x, const mag_tensor_t* _Nonnull y, float eps, double* _Nullable percent_eq);                                        /* Check if two tensors are equal with epsilon and percentage in equality. Set eps to < 0 to use machine epsilon. */
extern MAG_EXPORT void mag_tensor_img_draw_box(mag_tensor_t* _Nonnull t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb);
extern MAG_EXPORT void mag_tensor_img_draw_text(mag_tensor_t* _Nonnull t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* _Nonnull txt);                                      /* Draw text on image tensor */
extern MAG_EXPORT mag_ctx_t* _Nonnull mag_tensor_get_ctx(const mag_tensor_t* _Nonnull t);                                                                                                   /* Get the context of the tensor */
extern MAG_EXPORT void* _Nullable mag_tensor_get_user_data(const mag_tensor_t* _Nonnull t);                                                                                                 /* Get the user data of the tensor */
extern MAG_EXPORT void mag_tensor_set_user_data(mag_tensor_t* _Nonnull t, void* _Nullable ud);                                                                                              /* Set the user data of the tensor */
extern MAG_EXPORT void mag_tensor_save(const mag_tensor_t* _Nonnull t, const char* _Nonnull file);                                                                                          /* Save tensor to magnetron binary file. */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_load(mag_ctx_t* _Nonnull ctx, const char* _Nonnull file);                                                                               /* Load tensor from magnetron binary file. */
extern MAG_EXPORT mag_tensor_t* _Nonnull mag_tensor_load_image(mag_ctx_t* _Nonnull ctx, const char* _Nonnull file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h);    /* Create a tensor from an image file. */
extern MAG_EXPORT void mag_tensor_save_image(const mag_tensor_t* _Nonnull t, const char* _Nonnull file);                                                                                    /* Save tensor data as an image */
extern MAG_EXPORT void mag_tensor_export_graphviz(const mag_tensor_t* _Nonnull t, const char* _Nonnull file);                                                                               /* Export tensor computation graph as Graphviz DOT file */
#define mag_tensor_width(tensor) (mag_tensor_shape(tensor)[2])                                                                                                                              /* Get image width from tensor */
#define mag_tensor_height(tensor) (mag_tensor_shape(tensor)[1])                                                                                                                             /* Get image height from tensor */
#define mag_tensor_channels(tensor) (mag_tensor_shape(tensor)[0])                                                                                                                           /* Get image channels from tensor */

#ifdef __cplusplus
}
#endif
#endif

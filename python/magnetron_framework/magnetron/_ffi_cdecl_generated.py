# Autogenered by /Users/mariosieg/Documents/projects/magnetron/python/magnetron_framework/bing_gen.py 2025-01-20 11:24:33.732169, do NOT edit!

__MAG_CDECLS: str = '''
typedef enum mag_compute_device_type_t {
MAG_COMPUTE_DEVICE_TYPE_CPU = 0,
MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA = 1,
MAG_COMPUTE_DEVICE_TYPE__NUM
} mag_compute_device_type_t;
extern   const char* mag_device_type_get_name(mag_compute_device_type_t op);
typedef enum mag_exec_mode_t {
MAG_EXEC_MODE_EAGER = 0,
MAG_EXEC_MODE_DEFERRED = 1,
MAG_EXEC_MODE__NUM
} mag_exec_mode_t;
typedef enum mag_prng_algorithm_t {
MAG_PRNG_MERSENNE_TWISTER = 0,
MAG_PRNG_PCG = 1,
MAG_PRNG__NUM
} mag_prng_algorithm_t;
typedef enum mag_thread_sched_prio_t {
MAG_THREAD_SCHED_PRIO_NORMAL = 0,
MAG_THREAD_SCHED_PRIO_MEDIUM = 1,
MAG_THREAD_SCHED_PRIO_HIGH = 2,
MAG_THREAD_SCHED_PRIO_REALTIME = 3,
} mag_thread_sched_prio_t;
typedef enum mag_color_channels_t {
MAG_COLOR_CHANNELS_AUTO,
MAG_COLOR_CHANNELS_GRAY,
MAG_COLOR_CHANNELS_GRAY_A,
MAG_COLOR_CHANNELS_RGB,
MAG_COLOR_CHANNELS_RGBA,
MAG_COLOR_CHANNELS__NUM
} mag_color_channels_t;
extern   void* (*mag_get_alloc_fn(void))(void* blk, size_t size);
extern   void mag_set_alloc_fn(void* (*alloc)(void* blk, size_t size));
extern   void mag_set_log_mode(bool enabled);
typedef uint32_t mag_char32_t;
typedef struct mag_ctx_t mag_ctx_t;
typedef struct mag_device_descriptor_t {
mag_compute_device_type_t type;
uint32_t thread_count;
uint32_t cuda_device_id;
} mag_device_descriptor_t;
extern   mag_ctx_t* mag_ctx_create(mag_compute_device_type_t device);
extern   mag_ctx_t* mag_ctx_create2(const mag_device_descriptor_t* device_info);
extern   mag_exec_mode_t mag_ctx_get_exec_mode(const mag_ctx_t* _ptr);
extern   void mag_ctx_set_exec_mode(mag_ctx_t* _ptr, mag_exec_mode_t mode);
extern   mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* _ptr);
extern   void mag_ctx_set_prng_algorithm(mag_ctx_t* _ptr, mag_prng_algorithm_t algorithm, uint64_t seed);
extern   mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* _ptr);
extern   const char* mag_ctx_get_compute_device_name(const mag_ctx_t* _ptr);
extern   const char* mag_ctx_get_os_name(const mag_ctx_t* _ptr);
extern   const char* mag_ctx_get_cpu_name(const mag_ctx_t* _ptr);
extern   uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* _ptr);
extern   uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* _ptr);
extern   uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* _ptr);
extern   uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* _ptr);
extern   uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* _ptr);
extern   bool mag_ctx_is_numa_system(const mag_ctx_t* _ptr);
extern   size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* _ptr);
extern   void mag_ctx_profile_start_recording(mag_ctx_t* _ptr);
extern   void mag_ctx_profile_stop_recording(mag_ctx_t* _ptr, const char* export_csv_file);
extern   void mag_ctx_destroy(mag_ctx_t* _ptr);
typedef struct mag_tensor_t mag_tensor_t;
typedef enum mag_dtype_t {
MAG_DTYPE_F32,
MAG_DTYPE__NUM
} mag_dtype_t;
typedef struct mag_dtype_meta_t {
int64_t size;
const char* name;
} mag_dtype_meta_t;
extern   const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type);
extern   uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b);
extern   uint32_t mag_pack_color_f32(float r, float g, float b);
typedef enum mag_graph_eval_order_t {
MAG_GRAPH_EVAL_ORDER_FORWARD = 0,
MAG_GRAPH_EVAL_ORDER_REVERSE = 1
} mag_graph_eval_order_t;
extern   mag_tensor_t* mag_tensor_create_1d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1);
extern   mag_tensor_t* mag_tensor_create_2d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1, int64_t d2);
extern   mag_tensor_t* mag_tensor_create_3d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3);
extern   mag_tensor_t* mag_tensor_create_4d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
extern   mag_tensor_t* mag_tensor_create_5d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5);
extern   mag_tensor_t* mag_tensor_create_6d(mag_ctx_t* _ptr, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6);
extern   mag_tensor_t* mag_clone(mag_tensor_t* x);
extern   mag_tensor_t* mag_view(mag_tensor_t* x);
extern   mag_tensor_t* mag_transpose(mag_tensor_t* x);
extern   mag_tensor_t* mag_permute(mag_tensor_t* x, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t d4, uint32_t d5);
extern   mag_tensor_t* mag_mean(mag_tensor_t* x);
extern   mag_tensor_t* mag_min(mag_tensor_t* x);
extern   mag_tensor_t* mag_max(mag_tensor_t* x);
extern   mag_tensor_t* mag_sum(mag_tensor_t* x);
extern   mag_tensor_t* mag_abs(mag_tensor_t* x);
extern   mag_tensor_t* mag_abs_(mag_tensor_t* x);
extern   mag_tensor_t* mag_neg(mag_tensor_t* x);
extern   mag_tensor_t* mag_neg_(mag_tensor_t* x);
extern   mag_tensor_t* mag_log(mag_tensor_t* x);
extern   mag_tensor_t* mag_log_(mag_tensor_t* x);
extern   mag_tensor_t* mag_sqr(mag_tensor_t* x);
extern   mag_tensor_t* mag_sqr_(mag_tensor_t* x);
extern   mag_tensor_t* mag_sqrt(mag_tensor_t* x);
extern   mag_tensor_t* mag_sqrt_(mag_tensor_t* x);
extern   mag_tensor_t* mag_sin(mag_tensor_t* x);
extern   mag_tensor_t* mag_sin_(mag_tensor_t* x);
extern   mag_tensor_t* mag_cos(mag_tensor_t* x);
extern   mag_tensor_t* mag_cos_(mag_tensor_t* x);
extern   mag_tensor_t* mag_step(mag_tensor_t* x);
extern   mag_tensor_t* mag_step_(mag_tensor_t* x);
extern   mag_tensor_t* mag_softmax(mag_tensor_t* x);
extern   mag_tensor_t* mag_softmax_(mag_tensor_t* x);
extern   mag_tensor_t* mag_softmax_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_sigmoid(mag_tensor_t* x);
extern   mag_tensor_t* mag_sigmoid_(mag_tensor_t* x);
extern   mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x);
extern   mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x);
extern   mag_tensor_t* mag_silu(mag_tensor_t* x);
extern   mag_tensor_t* mag_silu_(mag_tensor_t* x);
extern   mag_tensor_t* mag_silu_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_silu_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_tanh(mag_tensor_t* x);
extern   mag_tensor_t* mag_tanh_(mag_tensor_t* x);
extern   mag_tensor_t* mag_tanh_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_relu(mag_tensor_t* x);
extern   mag_tensor_t* mag_relu_(mag_tensor_t* x);
extern   mag_tensor_t* mag_relu_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_relu_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_gelu(mag_tensor_t* x);
extern   mag_tensor_t* mag_gelu_(mag_tensor_t* x);
extern   mag_tensor_t* mag_gelu_dv(mag_tensor_t* x);
extern   mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x);
extern   mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_sub( mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_mul( mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y);
extern   mag_tensor_t* mag_adds(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_adds_(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_subs( mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_subs_(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_muls(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_muls_(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_divs(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_divs_(mag_tensor_t* x, float xi);
extern   mag_tensor_t* mag_matmul(mag_tensor_t* a, mag_tensor_t* b);
extern   void mag_tensor_incref(mag_tensor_t* t);
extern   bool mag_tensor_decref(mag_tensor_t* t);
extern   void mag_tensor_copy_buffer_from(mag_tensor_t* t, const void* data, size_t size);
extern   void mag_tensor_fill(mag_tensor_t* t, float x);
extern   void mag_tensor_fill_random_uniform(mag_tensor_t* t, float min, float max);
extern   void mag_tensor_fill_random_normal(mag_tensor_t* t, float mean, float stddev);
extern   uint64_t mag_tensor_get_packed_refcounts(const mag_tensor_t* t);
extern   void mag_tensor_retain(mag_tensor_t* t);
extern   size_t mag_tensor_get_memory_usage(const mag_tensor_t* t);
extern   void mag_tensor_print(const mag_tensor_t* t, bool with_header, bool with_data);
extern   void mag_tensor_set_name(mag_tensor_t* t, const char* name);
extern   void mag_tensor_fmt_name(mag_tensor_t* t, const char* fmt, ...);
extern   const char* mag_tensor_get_name(const mag_tensor_t* t);
extern   int64_t mag_tensor_rank(const mag_tensor_t* t);
extern   const int64_t* mag_tensor_shape(const mag_tensor_t* t);
extern   const int64_t* mag_tensor_strides(const mag_tensor_t* t);
extern   mag_dtype_t mag_tensor_dtype(const mag_tensor_t* t);
extern   void* mag_tensor_data_ptr(const mag_tensor_t* t);
extern   int64_t mag_tensor_data_size(const mag_tensor_t* t);
extern   int64_t mag_tensor_numel(const mag_tensor_t* t);
extern   int64_t mag_tensor_num_rows(const mag_tensor_t* t);
extern   int64_t mag_tensor_num_cols(const mag_tensor_t* t);
extern   bool mag_tensor_is_scalar(const mag_tensor_t* t);
extern   bool mag_tensor_is_vector(const mag_tensor_t* t);
extern   bool mag_tensor_is_matrix(const mag_tensor_t* t);
extern   bool mag_tensor_is_volume(const mag_tensor_t* t);
extern   bool mag_tensor_is_shape_eq(const mag_tensor_t* a, const mag_tensor_t* b);
extern   bool mag_tensor_are_strides_eq(const mag_tensor_t* a, const mag_tensor_t* b);
extern   bool mag_tensor_can_broadcast(const mag_tensor_t* a, const mag_tensor_t* b);
extern   bool mag_tensor_is_transposed(const mag_tensor_t* t);
extern   bool mag_tensor_is_permuted(const mag_tensor_t* t);
extern   bool mag_tensor_is_contiguous(const mag_tensor_t* t);
extern   float mag_tensor_get_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5);
extern   void mag_tensor_set_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x);
extern   float mag_tensor_get_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx);
extern   void mag_tensor_set_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx, float x);
extern   bool mag_tensor_eq(const mag_tensor_t* a, const mag_tensor_t* b);
extern   bool mag_tensor_is_close(const mag_tensor_t* a, const mag_tensor_t* b, float eps, double* percent_eq);
extern   void mag_tensor_img_draw_box(mag_tensor_t* t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb);
extern   void mag_tensor_img_draw_text(mag_tensor_t* t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* txt);
extern   mag_ctx_t* mag_tensor_get_ctx(const mag_tensor_t* t);
extern   void* mag_tensor_get_user_data(const mag_tensor_t* t);
extern   void mag_tensor_set_user_data(mag_tensor_t* t, void* ud);
extern   void mag_tensor_save(const mag_tensor_t* t, const char* file);
extern   mag_tensor_t* mag_tensor_load(mag_ctx_t* _ptr, const char* file);
extern   mag_tensor_t* mag_tensor_load_image(mag_ctx_t* _ptr, const char* file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h);
extern   void mag_tensor_save_image(const mag_tensor_t* t, const char* file);
'''

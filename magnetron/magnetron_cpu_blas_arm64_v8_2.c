/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) \
    || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    || !defined(__ARM_FEATURE_DOTPROD)
#error "BLAS specialization requires matching compile flags"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_arm64_v_8_2
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_arm64_v_8_2_features

#include "magnetron_cpu_blas.inl"

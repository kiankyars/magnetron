/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#if !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_SVE2)
#error "BLAS specialization requires matching compile flags"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_arm64_v_9
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_arm64_v_9_features

#include "magnetron_cpu_blas.inl"

/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#if !defined(__AVX512F__) || !defined(__AVX512VL__)  || !defined(__AVX512VNNI__)
    || !defined(__AVX512BF16__)  || !defined(__AVX512BW__)  || !defined(__AVX512DQ__)
#error "BLAS specialization requires matching compile flags"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_znver4
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_znver4_features

#include "magnetron_cpu_blas.inl"

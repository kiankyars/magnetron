/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef __AVX2__
#error "BLAS specialization requires matching compile flags"
#endif
#ifdef __AVX512f__
#error "BLAS specialization feature too high"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_avx2
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_avx2_features

#include "magnetron_cpu_blas.inl"

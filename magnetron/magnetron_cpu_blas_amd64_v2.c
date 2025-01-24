/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */


#ifndef _MSC_VER
#if !defined(__SSE__) \
    || !defined(__SSE2__) \
    || !defined(__SSE3__) \
    || !defined(__SSSE3__) \
    || !defined(__SSE4_1__) \
    || !defined(__SSE4_2__)
#error "BLAS specialization requires matching compile flags"
#endif
#endif
#ifdef __AVX__
#error "BLAS specialization feature too high"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_v2
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_v2_features

#include "magnetron_cpu_blas.inl"


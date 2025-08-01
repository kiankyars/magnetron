/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#ifndef _MSC_VER
#if !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_SVE2)
#pragma message ("Current compiler lacks modern optimization flags - upgrade GCC/Clang to enable better optimizations!")
#endif
#else
#pragma message("MSVC does not allow to fine tune CPU architecture level, usine clang-cl or mingw-w64 for best performance!")
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_arm64_v_9
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_arm64_v_9_features

#include "magnetron_cpu_blas.inl"

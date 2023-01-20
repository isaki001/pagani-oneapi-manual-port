#ifndef ONE_API_RULE_PARAMS_H
#define ONE_API_RULE_PARAMS_H

#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/util/cuhreResult.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/util/MemoryUtil.h"

template <size_t ndim>
constexpr size_t
CuhreFuncEvalsPerRegion()
{
  return (1 + 2 * ndim + 2 * ndim + 2 * ndim + 2 * ndim +
          2 * ndim * (ndim - 1) + 4 * ndim * (ndim - 1) +
          4 * ndim * (ndim - 1) * (ndim - 2) / 3 + (1 << ndim));
}

template <size_t ndim>
constexpr size_t
get_permutation_size()
{
  return (1 + 1 * 1 + 2 * ndim * 1 + 2 * ndim * 1 + 2 * ndim * 1 +
          2 * ndim * 1 + 2 * ndim * (ndim - 1) * 2 + 4 * ndim * (ndim - 1) * 2 +
          4 * ndim * (ndim - 1) * (ndim - 2) * 3 / 3 + ndim * (1 << ndim));
}

template <size_t ndim>
constexpr size_t
get_feval()
{
  return (1 + 2 * ndim + 2 * ndim + 2 * ndim + 2 * ndim +
          2 * ndim * (ndim - 1) + 4 * ndim * (ndim - 1) +
          4 * ndim * (ndim - 1) * (ndim - 2) / 3 + (1 << ndim));
}

template <typename T>
struct Structures {

  Structures() = default;

  T* _gpuG = nullptr;
  T* _cRuleWt = nullptr;
  T* _GPUScale = nullptr;
  T* _GPUNorm = nullptr;
  int* _gpuGenPos = nullptr;
  int* _gpuGenPermGIndex = nullptr;
  int* _gpuGenPermVarCount = nullptr;
  int* _gpuGenPermVarStart = nullptr;
  size_t* _cGeneratorCount = nullptr;
};

template <size_t ndim>
struct Rule_Params {
  Rule_Params() = default;
  void init(queue& q, quad::Rule<double>& rule);
  void destroy(queue& q);

  double* _gpuG;
  double* _cRuleWt;
  double* _GPUScale;
  double* _GPUNorm;
  double* _generators;

  int* _gpuGenPos;
  int* _gpuGenPermGIndex;
  int* _gpuGenPermVarCount;
  int* _gpuGenPermVarStart;
};

template <size_t ndim>
void
Rule_Params<ndim>::init(queue& q, quad::Rule<double>& rule)
{
  constexpr size_t feval = get_feval<ndim>();
  constexpr size_t permutations_size = get_permutation_size<ndim>();
  constexpr size_t nsets = 9;
  constexpr size_t nrules = 5;

  // variables needed to compute the generators

  _gpuGenPermVarStart = sycl::malloc_device<int>(feval + 1, q);
  _gpuGenPermGIndex = sycl::malloc_device<int>(feval, q);
  _gpuGenPos = sycl::malloc_device<int>(permutations_size, q);
  //_gpuGenPermVarStart = quad::copy_to_shared<int>(q, rule.cpuGenPermVarStart,
  // feval + 1); _gpuGenPermGIndex = quad::copy_to_shared<int>(q,
  // rule.cpuGenPermGIndex, feval); _gpuGenPos = quad::copy_to_shared<int>(q,
  // rule.genPtr, permutations_size);

  quad::copy_to_device<int>(
    _gpuGenPermVarStart, rule.cpuGenPermVarStart, feval + 1);
  quad::copy_to_device<int>(_gpuGenPermGIndex, rule.cpuGenPermGIndex, feval);
  quad::copy_to_device<int>(_gpuGenPos, rule.genPtr, permutations_size);

  _gpuG = sycl::malloc_device<double>(ndim * nsets, q);
  _cRuleWt = sycl::malloc_device<double>(nrules * nsets, q);
  _GPUScale = sycl::malloc_device<double>(nrules * nsets, q);
  _GPUNorm = sycl::malloc_device<double>(nrules * nsets, q);
  _gpuGenPermVarCount = sycl::malloc_device<int>(feval, q);

  quad::copy_to_device<int>(
    _gpuGenPermVarCount, rule.cpuGenPermVarCount, feval);
  quad::copy_to_device<double>(_GPUNorm, rule.CPUNorm, nrules * nsets);
  quad::copy_to_device<double>(_GPUScale, rule.CPUScale, nrules * nsets);
  quad::copy_to_device<double>(_cRuleWt, rule.CPURuleWt, nrules * nsets);
  quad::copy_to_device<double>(_gpuG, rule.cpuG, ndim * nsets);

  //_gpuG = quad::copy_to_shared<double>(q, rule.cpuG, ndim * nsets);
  // variables needed to compute estimates
  //_cRuleWt = quad::copy_to_shared<double> (q, rule.CPURuleWt, nrules * nsets);
  //_GPUScale = quad::copy_to_shared<double> (q, rule.CPUScale, nrules * nsets);
  //_GPUNorm = quad::copy_to_shared<double> (q, rule.CPUNorm, nrules * nsets);
  //_gpuGenPermVarCount = quad::copy_to_shared<int> (q, rule.cpuGenPermVarCount,
  // feval);

  _generators = sycl::malloc_device<double>(feval * ndim, q);
}

template <size_t ndim>
void
Rule_Params<ndim>::destroy(queue& q)
{
  free(_gpuG, q);
  free(_cRuleWt, q);
  free(_GPUScale, q);
  free(_GPUNorm, q);
  free(_generators, q);
  free(_gpuGenPos, q);
  free(_gpuGenPermGIndex, q);
  free(_gpuGenPermVarCount, q);
  free(_gpuGenPermVarStart, q);
}

#endif

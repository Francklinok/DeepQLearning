#pragma once

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#endif

#if defined(__APPLE__) && defined(__arm64__)
#define HAS_NPU 1
#include <Accelerate/Accelerate.h>
#elif defined(_WIN32) && defined(ENABLE_DIRECTML)
#define HAS_NPU 1
#include <directml.h>
#endif

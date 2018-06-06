#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

// compute low + q*high (mod p)
__global__ void ModCastKernel(const int size, const float* in_low, const float* in_high, const double q, const double p, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
	double tmp = in_low[i] + q * in_high[i];
    out[i] = (float) remainder(tmp, p);
  }
}


void ModCastFunctor(int size, const float* in_low, const float* in_high, const double q, const double p, float* out) {
  ModCastKernel<<<32, 256>>>(size, in_low, in_high, q, p, out);
}


// compute fmod(inp, p)
template <typename T>
__global__ void FmodKernel(const int size, const T* inp, const T p, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    out[i] = fmod(inp[i], p);
  }
}

template <typename T>
void FmodFunctor(int size, const T* inp, const T p, T* out) {
  FmodKernel<T><<<32, 256>>>(size, inp, p, out);
}

template void FmodFunctor<float>(int size, const float* inp, const float p, float* out);
template void FmodFunctor<double>(int size, const double* inp, const double p, double* out);

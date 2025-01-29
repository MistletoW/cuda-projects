#ifndef _1D_CONVOLUTION_CUH_
#define _1D_CONVOLUTION_CUH_

// Includes
#include <cuda_runtime.h>

// Function declarations
__global__ void convolve1D(const float* input, const float* mask, float* output, int inputSize, int maskSize);

#endif // _1D_CONVOLUTION_CUH_
#ifndef CONVOLVE_1D_H
#define CONVOLVE_1D_H

#include "tensor.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <iostream>
#include <functional>

using namespace std;

// Define tile size for shared memory optimization
#define TILE_SIZE 64

// CUDA Kernel for Strategy 1 (Thread-Based on Input Tile)
__global__ void convolve1DKernel_Strategy1(const float* input, const float* filter, float* output, int inputWidth, int filterWidth);

// Host Function to Call Kernel
Tensor convolve1D_Strategy1(Tensor input, Tensor filter);

#endif // CONVOLVE_1D_H

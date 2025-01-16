#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <vector>
#include <string>
#include <cuda_runtime.h>

// Define Tensor as a 3D vector of floats
using Tensor = std::vector<std::vector<std::vector<float>>>;

// Prototypes for tensor operations
void generateTensor(Tensor& tensor, int depth, int rows, int cols, float min, float max);
bool verifyTensors(const Tensor& A, const Tensor& B);
void printTensor(const Tensor& tensor, const std::string& label);
std::vector<float> flattenTensor(const Tensor& tensor);
Tensor unflattenTensor(const std::vector<float>& flatTensor, int depth, int rows, int cols);

__global__ void generateTensorCUDAKernel(float* tensor, int depth, int rows, int cols, float min, float max, int seed);
void generateTensorCUDA(Tensor& tensor, int depth, int rows, int cols, float min, float max);

#endif

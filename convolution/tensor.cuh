#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <iostream>

class Tensor {
private:
    float* data;         // Host-side data
    float* deviceData;   // Device-side data for CUDA
    int dims[3];         // Dimensions: [width, height, depth]
    enum Layout { ROW_MAJOR, COLUMN_MAJOR, DEPTH_MAJOR } layout;

    // Index calculation based on layout
    size_t index(int x, int y, int z) const;

public:
    // Constructor
    Tensor(int width, int height, int depth, Layout tensorLayout);

    // Destructor
    ~Tensor();

    // Set element
    void set(int x, int y, int z, float value);

    // Get element
    float get(int x, int y, int z) const;

    // Allocate device memory
    void allocateDeviceMemory();

    // Copy data to device
    void copyToDevice();

    // Copy data back to host
    void copyToHost();

    // Print tensor
    void print() const;

    // Access dimensions
    int getWidth() const;
    int getHeight() const;
    int getDepth() const;

    // Tensor verification (compares two tensors)
    static bool verifyTensor(const Tensor& A, const Tensor& B, float tolerance = 1e-5f);

    // CPU Random Tensor Generator
    static Tensor generateTensor(int width, int height, int depth, Layout layout);

    // CUDA Random Tensor Generator (Launches Kernel)
    static Tensor generateTensorCUDA(int width, int height, int depth, Layout layout);

    // CUDA Kernel for Random Number Generation
    __global__ static void generateTensorCUDAKernel(float* deviceData, int width, int height, int depth, unsigned long long seed);
};

#endif // TENSOR_CUH

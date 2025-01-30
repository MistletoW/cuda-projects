#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdexcept>

using namespace std;

// CUDA error checking macro
#define cuda_error_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "GPU Assert: %s at %s: %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}

// Layout type for tensor storage
enum Layout { ROW_MAJOR, COLUMN_MAJOR, DEPTH_MAJOR };

// Tensor class definition
class Tensor {
public:
    float* data;         // Host data
    float* deviceData;   // Device data (CUDA)
    int dims[3];         // Stores width, height, depth
    Layout layout;       // Storage layout (Row-Major, Column-Major, etc.)

    // Constructor & Destructor
    Tensor(int width, int height, int depth, Layout tensorLayout = ROW_MAJOR);
    ~Tensor();

    // Index calculation based on tensor layout
    size_t index(int x, int y, int z) const;

    // Get & Set functions
    void set(int x, int y, int z, float value);
    float get(int x, int y, int z) const;
    void print() const;

    int getWidth() const { return dims[0]; }
    int getHeight() const { return dims[1]; }
    int getDepth() const { return dims[2]; }
    float* getData() const { return data; }


    // Device memory management
    void allocateDeviceMemory();
    void copyToDevice();
    void copyToHost();

    // Tensor comparison for validation
    static bool verifyTensor(const Tensor& A, const Tensor& B, float tolerance = 1e-5f);

    // Random tensor generation (CPU)
    static Tensor generateTensor(int width, int height, int depth, Layout layout = ROW_MAJOR);

    // Random tensor generation (CUDA)
    void generateTensorCUDA(int width, int height, int depth, Layout layout = ROW_MAJOR);
};

// CUDA Kernel for tensor randomization
__global__ void generateTensorCUDAKernel(float* deviceData, int width, int height, int depth, unsigned long long seed);

#endif // TENSOR_H

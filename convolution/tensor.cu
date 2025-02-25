#include "tensor.cuh"
#include <curand_kernel.h>
#include <cstdlib>
#include <cmath>
#include <ctime>

// Constructor
Tensor::Tensor(int width, int height, int depth, Layout tensorLayout)
    : data(nullptr), deviceData(nullptr), layout(tensorLayout) {
    dims[0] = width;
    dims[1] = height;
    dims[2] = depth;

    size_t totalSize = width * height * depth;
    data = new float[totalSize](); // Initialize to zero
}

// Destructor
Tensor::~Tensor() {
    delete[] data;
    if (deviceData) {
        cuda_error_chk(cudaFree(deviceData));
    }
}

// Index calculation
size_t Tensor::index(int x, int y, int z) const {
    switch (layout) {
        case ROW_MAJOR:
            return z * (dims[0] * dims[1]) + y * dims[0] + x;
        case COLUMN_MAJOR:
            return x * (dims[1] * dims[2]) + y * dims[2] + z;
        case DEPTH_MAJOR:
            return y * (dims[0] * dims[2]) + x * dims[2] + z;
    }
    throw std::runtime_error("Invalid layout.");
}

// Set element
void Tensor::set(int x, int y, int z, float value) {
    data[index(x, y, z)] = value;
}

// Get element
float Tensor::get(int x, int y, int z) const {
    return data[index(x, y, z)];
}

void Tensor::print() const {
    for (int z = 0; z < dims[2]; ++z) {
        for (int y = 0; y < dims[1]; ++y) {
            for (int x = 0; x < dims[0]; ++x) {
                std::cout << get(x, y, z) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "------\n";
    }
}

// Allocate device memory
void Tensor::allocateDeviceMemory() {
    if (!deviceData) {
        size_t totalSize = dims[0] * dims[1] * dims[2] * sizeof(float);
        cuda_error_chk(cudaMalloc(&deviceData, totalSize));
    }
}

// Copy data to device
void Tensor::copyToDevice() {
    allocateDeviceMemory();
    size_t totalSize = dims[0] * dims[1] * dims[2] * sizeof(float);
    cuda_error_chk(cudaMemcpy(deviceData, data, totalSize, cudaMemcpyHostToDevice));
}

// Copy data back to host
void Tensor::copyToHost() {
    if (deviceData) {
        size_t totalSize = dims[0] * dims[1] * dims[2] * sizeof(float);
        cuda_error_chk(cudaMemcpy(data, deviceData, totalSize, cudaMemcpyDeviceToHost));
    }
}

// Verify if two tensors are identical
bool Tensor::verifyTensor(const Tensor& A, const Tensor& B, float tolerance) {
    if (A.dims[0] != B.dims[0] || A.dims[1] != B.dims[1] || A.dims[2] != B.dims[2]) {
        return false;
    }
    
    for (int z = 0; z < A.dims[2]; ++z) {
        for (int y = 0; y < A.dims[1]; ++y) {
            for (int x = 0; x < A.dims[0]; ++x) {
                float diff = std::abs(A.get(x, y, z) - B.get(x, y, z));
                if (diff > tolerance) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Generate a tensor with random numbers (CPU)
Tensor Tensor::generateTensor(int width, int height, int depth, Layout layout) {
    Tensor tensor(width, height, depth, layout);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                tensor.set(x, y, z, static_cast<float>(rand()) / RAND_MAX);
            }
        }
    }
    return tensor;
}

// CUDA Kernel for random number generation
__global__ void generateTensorCUDAKernel(float* deviceData, int width, int height, int depth, unsigned long long seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < width && y < height && z < depth) {
        int index = z * (width * height) + y * width + x;
        
        curandState state;
        curand_init(seed + index, index, 0, &state);
        deviceData[index] = curand_uniform(&state);
    }
}

// Generate a tensor with random numbers using CUDA
void Tensor::generateTensorCUDA(int width, int height, int depth, Layout layout) {
    allocateDeviceMemory();  // Ensure device memory is allocated

    // Compute grid and block sizes
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);

    // Generate seed from host
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    // Launch CUDA kernel
    generateTensorCUDAKernel<<<numBlocks, threadsPerBlock>>>(deviceData, width, height, depth, seed);
    
    cuda_error_chk(cudaGetLastError());  // Check for kernel launch errors
    cuda_error_chk(cudaDeviceSynchronize());  // Ensure all kernel executions complete

    copyToHost();  // Copy data back to host
}

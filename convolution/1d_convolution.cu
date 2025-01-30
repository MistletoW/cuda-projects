#include "1d_convolution.cuh"
#include "tensor.cuh"

using namespace std;

int main(){
    Tensor A = Tensor(4, 1, 1, ROW_MAJOR);
    A.generateTensorCUDA(4, 1, 1, ROW_MAJOR);

    Tensor filter = Tensor(3, 1, 1, ROW_MAJOR);
    filter.generateTensorCUDA(3, 1, 1, ROW_MAJOR);

    Tensor result = convolve1D_Strategy1(A, filter);

    A.print();
    filter.print();
    result.print();

    return 0;
}

// int64_t Benchmark(const function<Tensor()>& test, Tensor& result)
// {
//     auto start = high_resolution_clock::now();
//     result = test();
//     auto end = high_resolution_clock::now();
//     return duration_cast<microseconds>(end - start).count();
// }

Tensor convolve1D(Tensor& input, Tensor& kernel){
    if (input.getWidth() <= kernel.getWidth()) {
        throw runtime_error("Input tensor width must be greater than kernel width.");
    }
    

    if(input.getHeight() != 1 || kernel.getHeight() != 1){
        throw runtime_error("Height of both input and kernel tensors must be 1.");
    }

    if(input.getDepth() != 1 || kernel.getDepth() != 1){
        throw runtime_error("Depth of input and kernel tensors must be 1.");
    }

    int outputWidth = input.getWidth() + kernel.getWidth() - 1;

    Tensor output(outputWidth, 1, 1, ROW_MAJOR);

    for(int i = 0; i < outputWidth; i++){
        float sum = 0.0f;
        for(int j = 0; j < kernel.getWidth(); j++){
            int inputIndex = i - j;
            if(inputIndex >= 0 && inputIndex < input.getWidth()){
                sum += input.get(inputIndex, 0, 0) * kernel.get(j, 0, 0);
            }
        }
        output.set(i, 0, 0, sum);
    }

    return output;
}

__global__ void convolve1DKernel_Strategy1(const float* input, const float* filter, float* output, int inputWidth, int filterWidth) {
    extern __shared__ float sharedMemory[];

    float* sharedInput = sharedMemory;
    float* sharedFilter = &sharedMemory[TILE_SIZE + filterWidth - 1];


    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tileStart = blockIdx.x * TILE_SIZE;
    int outputIndex = tileStart + tid;

    // Load input data into shared memory
    for (int i = tid; i < TILE_SIZE + filterWidth - 1; i += blockDim.x) {
        int memoryIndex = tileStart + i;
        if (memoryIndex < inputWidth) {
            sharedInput[i] = input[memoryIndex];  // Load input data
        } else {
            sharedInput[i] = 0.0f;  // Zero padding
        }
    }

    // Load kernel into shared memory (only first filterWidth threads)
    if (tid < filterWidth) {
        sharedFilter[tid] = filter[tid];
    }

    __syncthreads(); // Ensure shared memory is loaded before computation

    // Compute output using shared memory
    if (outputIndex < inputWidth + filterWidth - 1 && outputIndex < inputWidth) {
        float sum = 0.0f;
        int localIndex = outputIndex - tileStart;
        for (int i = 0; i < filterWidth; i++) {
            if ((localIndex + i) < (TILE_SIZE + filterWidth - 1) && (localIndex + i) >= 0) {
                sum += sharedInput[localIndex + i] * sharedFilter[i];
            }
        }
        output[outputIndex] = sum;
    }
}

Tensor convolve1D_Strategy1(Tensor input, Tensor filter) {
    // Check input conditions
    if (input.getWidth() < filter.getWidth()) {
        throw runtime_error("Input tensor width must be greater than kernel width.");
    }
    if (input.getHeight() != 1 || filter.getHeight() != 1) {
        throw runtime_error("Height of both input and kernel tensors must be 1.");
    }
    if (input.getDepth() != 1 || filter.getDepth() != 1) {
        throw runtime_error("Depth of input and kernel tensors must be 1.");
    }

    int outputWidth = input.getWidth() + filter.getWidth() - 1;
    Tensor output(outputWidth, 1, 1, ROW_MAJOR);

    // Allocate device memory
    float* d_input;
    float* d_filter;
    float* d_output;

    cuda_error_chk(cudaMalloc(&d_input, input.getWidth() * sizeof(float)));
    cuda_error_chk(cudaMalloc(&d_filter, filter.getWidth() * sizeof(float)));
    cuda_error_chk(cudaMalloc(&d_output, outputWidth * sizeof(float)));

    // Copy data to device
    cuda_error_chk(cudaMemcpy(d_input, input.getData(), input.getWidth() * sizeof(float), cudaMemcpyHostToDevice));
    cuda_error_chk(cudaMemcpy(d_filter, filter.getData(), filter.getWidth() * sizeof(float), cudaMemcpyHostToDevice));

    // Define CUDA grid/block configuration
    int blockSize = min(TILE_SIZE, 1024);  // Limit block size to max CUDA threads per block
    int numBlocks = (outputWidth + blockSize - 1) / blockSize;

    // Launch kernel
    convolve1DKernel_Strategy1<<<numBlocks, blockSize>>>(d_input, d_filter, d_output, input.getWidth(), filter.getWidth());
    cuda_error_chk(cudaPeekAtLastError());  // Check for kernel launch errors
    cuda_error_chk(cudaDeviceSynchronize());  // Ensure kernel execution is completed

    // Copy result back to host
    cuda_error_chk(cudaMemcpy(output.getData(), d_output, outputWidth * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cuda_error_chk(cudaFree(d_input));
    cuda_error_chk(cudaFree(d_filter));
    cuda_error_chk(cudaFree(d_output));

    return output;
}

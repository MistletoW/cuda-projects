// Skeleton for CUDA Vector Addition Project

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <random>

using namespace std;

// Function prototypes
void readVectors(const string &filename, vector<float> &vec1, vector<float> &vec2);
int verifySize(const vector<float> &vec1, const vector<float> &vec2);
vector<float> addVectorsCPU(const vector<float> &vec1, const vector<float> &vec2, int vectorSize);
__global__ void addVectorsKernel(const float *vec1, const float *vec2, float *result, int size);
vector<float> addVectorsCUDA(const vector<float> &vec1, const vector<float> &vec2, int vectorSize);
void verifyResults(const vector<float> &cpuResult, const vector<float> &gpuResult);
void generateRandomVectors(vector<float> &vec1, vector<float> &vec2, int size, float min, float max);

int main() {
    // Input vectors
    vector<float> vec1, vec2;

    // Read vectors from file
    readVectors("vectors.txt", vec1, vec2);

    int vectorSize = verifySize(vec1 , vec2); //maybe make this const

    // CPU Implementation
    vector<float> resultCPU = addVectorsCPU(vec1, vec2, vectorSize);

    // CUDA Implementation
    vector<float> resultCuda = addVectorsCUDA(vec1, vec2, vectorSize);

    // Verify results
    verifyResults(resultCPU, resultCuda);

    return 0;
}

// Function to read vectors from a file
void readVectors(const string &filename, vector<float> &vec1, vector<float> &vec2) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    float value;
    while (file >> value) {
        vec1.push_back(value);
        if (file >> value) {
            vec2.push_back(value);
        }
    }

    file.close();
}

// CPU implementation of vector addition
// Skeleton for: addVectorsCPU
vector<float> addVectorsCPU(const vector<float> &vec1, const vector<float> &vec2, int vectorSize){
    vector<float> result(vectorSize);

    for (int i = 0; i < vectorSize; i++) {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

// CUDA kernel for vector addition
// Skeleton for: addVectorsKernel
//Kernel: The specificed computation each thread will perform
__global__ void addVectorsKernel(const float *vec1, const float *vec2, float *result, int size) {
    // Calculate the global thread index
    //blockId.x = index of block within grid
    //blockDim.x = num of threads in block
    //threadIDx.x = index of threat in block
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (id < size) {
        result[id] = vec1[id] + vec2[id];
    }
}


// CUDA implementation of vector addition
// Skeleton for: addVectorsCUDA
vector<float> addVectorsCUDA(const vector<float> &vec1, const vector<float> &vec2, int vectorSize) {
    size_t bytes = vectorSize * sizeof(float); //size of vectors we will be adding

    // give our vectrors on GPU the size of a vector we will add
    float *d_vec1, *d_vec2, *d_result;
    cudaMalloc(&d_vec1, bytes);
    cudaMalloc(&d_vec2, bytes);
    cudaMalloc(&d_result, bytes);

    // give our vectors on the gpu our data from vectors on the cpu
    cudaMemcpy(d_vec1, vec1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (vectorSize + threads - 1) / threads; //need to calculate the min number of blocks needed to process all elements, threads+1 to round up our blocks
    //grid with blocks and threads in each block
    //passes addresses of vectors allocated on the gpu
    //thread then processes global id and id determines which float in vector each thread processes
    addVectorsKernel<<<blocks, threads>>>(d_vec1, d_vec2, d_result, vectorSize); 

    // Copy result back to host
    vector<float> result(vectorSize);
    cudaMemcpy(result.data(), d_result, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);

    return result;
}


// Function to verify results
int verifySize(const vector<float> &vec1, const vector<float> &vec2) {
    if (vec1.size() != vec2.size()) {
        cerr << "Error: Vectors are not the same size." << endl
             << "Vector 1 size: " << vec1.size() << endl
             << "Vector 2 size: " << vec2.size() << endl;
        exit(EXIT_FAILURE);
    }

    return vec1.size();
}
// Skeleton for: verifyResults
void verifyResults(const vector<float> &resultCPU, const vector<float> &resultCuda){
    if(resultCPU.size() != resultCuda.size()){
        cerr << "Error: Results vector sizes do not match." << endl;
        exit(EXIT_FAILURE);
    }

    for(size_t i = 0; i < resultCPU.size(); ++i){
        if(abs(resultCPU[i] - resultCuda[i]) > 1e-5){
            cerr << "Error: Mismatch at index " << i << ": CPU=" << resultCPU[i] << ", GPU=" << resultCuda[i] << endl;
            exit(EXIT_FAILURE);
        }
    }

    cout << "Match!"<< endl;
}

void generateRandomVectors(vector<float> &vec1, vector<float> &vec2, int size, float min, float max){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min, max);

    vec1.resize(size);
    vec2.resize(size);

    for (int i = 0; i < size; ++i) {
        vec1[i] = dis(gen);
        vec2[i] = dis(gen);
    }
}
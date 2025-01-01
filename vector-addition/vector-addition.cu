// Skeleton for CUDA Vector Addition Project

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// Function prototypes
void readVectors(const string &filename, vector<float> &vec1, vector<float> &vec2);
vector<float> addVectorsCPU(const vector<float> &vec1, const vector<float> &vec2);
__global__ void addVectorsKernel(const float *vec1, const float *vec2, float *result, int size);
vector<float> addVectorsCUDA(const vector<float> &vec1, const vector<float> &vec2);
void verifyResults(const vector<float> &cpuResult, const vector<float> &gpuResult);

int main() {
    // Input vectors
    vector<float> vec1, vec2;

    // Read vectors from file
    readVectors("vectors.txt", vec1, vec2);

    // CPU Implementation
    // Function: addVectorsCPU

    // CUDA Implementation
    // Function: addVectorsCUDA

    // Verify results
    // Function: verifyResults

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

// CUDA kernel for vector addition
// Skeleton for: addVectorsKernel

// CUDA implementation of vector addition
// Skeleton for: addVectorsCUDA

// Function to verify results
void verifyVectorsSameSize(const vector<float> &vec1, const vector<float> &vec2) {
    if (vec1.size() != vec2.size()) {
        cerr << "Error: Vectors are not the same size.
"
             << "Vector 1 size: " << vec1.size() << "
"
             << "Vector 2 size: " << vec2.size() << endl;
        exit(EXIT_FAILURE);
    }
}
// Skeleton for: verifyResults
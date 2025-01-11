#include "matrix_multiplication.cuh"
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;

// Main Function
int main() {
    testSuite(3, 0, 100, 1, 20000);
}

// Function to generate a random matrix
void generateMatrix(Matrix& matrix, int rows, int cols, float min, float max) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min, max);

    matrix.resize(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

// CPU implementation of matrix multiplication
Matrix matrixMultiplyCPU(const Matrix& A, const Matrix& B) {
     // Check dimensions
     if (A[0].size() != B.size()) {
        cerr << "Error: Incompatible matrix dimensions for multiplication." << endl;
        return Matrix(); // Return an empty matrix
    }

    // Initialize result matrix C with appropriate dimensions
    int rowsA = A.size(); //m
    int colsA = A[0].size(); //n also rowsB
    int colsB = B[0].size(); //p

    Matrix C(rowsA, vector<float>(colsB, 0.0f));

    // Perform matrix multiplication
    //We make a matrix of size M x P, so we explore every cell in that Matrix C and then we get every item in that row for A, and column for B.
    //This shared value is N or k in our for loop
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// CUDA global memory kernel for matrix multiplication
__global__ void matrixMultiplyGlobalKernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    // Compute the row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within bounds
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;

        // Compute the dot product for C[row][col]
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }

        // Write the result to the output matrix
        C[row * colsB + col] = sum;
    }
}

// CUDA global memory implementation of matrix multiplication
Matrix matrixMultiplyCUDA_Global(const Matrix& A, const Matrix& B) {
    // Check dimensions
    if (A[0].size() != B.size()) {
        cerr << "Error: Incompatible matrix dimensions for multiplication." << endl;
        return Matrix(); // Return an empty matrix
    }

    float *d_A, *d_B, *d_C;

    int rowsA = A.size(); //m
    int colsA = A[0].size(); //n also rowsB
    int colsB = B[0].size(); //p

    //To calculate bytes for each matrix use
    //bytes = rows * cols * sizeof(float);
    size_t bytesA = rowsA * colsA * sizeof(float);
    size_t bytesB = colsA * colsB * sizeof(float);
    size_t bytesC = rowsA * colsB * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    //flatten our matrix to a 1d vector
    vector<float> flatA = flattenMatrix(A);
    vector<float> flatB = flattenMatrix(B);

    //give our matrix data to the gpu
    cudaMemcpy(d_A, flatA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), bytesB, cudaMemcpyHostToDevice);

    //launch kernel
    dim3 threadsPerBlock(16, 16); //256 threads
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    matrixMultiplyGlobalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    //create a flatten C
    vector<float> flatC(rowsA * colsB);

    // Copy the result back to the host
    cudaMemcpy(flatC.data(), d_C, bytesC, cudaMemcpyDeviceToHost);

    Matrix C = unflattenMatrix(flatC, rowsA, colsB);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// CUDA shared memory kernel for matrix multiplication
__global__ void matrixMultiplySharedKernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) 
{
    const int BLOCK_SIZE = 16;
     // Shared memory for tiles of A and B
     __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
     __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
 
     // Row and column index of the element in C this thread computes
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
 
     // Accumulate the result for C[row][col]
     float sum = 0.0f;
 
     // Loop over tiles
     for (int t = 0; t < (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) 
     {
         // Load tiles into shared memory
         // each thread loads in the tile cell where our result tile cell will go
         // so thread will find C_Tile[i][j] will load in A_Tile[i][j] and B_Tile[i][j]
         if (row < rowsA && t * BLOCK_SIZE + threadIdx.x < colsA) {
             tileA[threadIdx.y][threadIdx.x] = A[row * colsA + t * BLOCK_SIZE + threadIdx.x]; //cell in tileA equals cell in Mat A
         } else {
             tileA[threadIdx.y][threadIdx.x] = 0.0f;
         }
 
         if (t * BLOCK_SIZE + threadIdx.y < colsA && col < colsB) {
             tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * colsB + col]; //cell in tileB equals cell in Mat B
         } else {
             tileB[threadIdx.y][threadIdx.x] = 0.0f;
         }
 
         // Synchronize threads to ensure all tiles are loaded
         __syncthreads();
 
         // Perform computation for the tile
         for (int k = 0; k < BLOCK_SIZE; ++k) {
             sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
         }
 
         // Synchronize threads before loading the next tile
         __syncthreads();
     }
 
     // Write the computed value to C
     if (row < rowsA && col < colsB) {
         C[row * colsB + col] = sum;
     }
}

// CUDA shared memory implementation of matrix multiplication
Matrix matrixMultiplyCUDA_Shared(const Matrix &A, const Matrix &B) {
    // Check dimensions
    if (A[0].size() != B.size()) {
        cerr << "Error: Incompatible matrix dimensions for multiplication." << endl;
        return Matrix(); // Return an empty matrix
    }

    float *d_A, *d_B, *d_C;

    int rowsA = A.size(); //m
    int colsA = A[0].size(); //n also rowsB
    int colsB = B[0].size(); //p

    //To calculate bytes for each matrix use
    //bytes = rows * cols * sizeof(float);
    size_t bytesA = rowsA * colsA * sizeof(float);
    size_t bytesB = colsA * colsB * sizeof(float);
    size_t bytesC = rowsA * colsB * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    //flatten our matrix to a 1d vector
    vector<float> flatA = flattenMatrix(A);
    vector<float> flatB = flattenMatrix(B);

    //give our matrix data to the gpu
    cudaMemcpy(d_A, flatA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), bytesB, cudaMemcpyHostToDevice);

    //launch kernel
    const int BLOCK_SIZE = 16; //NEEDS TO MATCH KERNEL VALUE, GENERATES ERROR IF PASSED IN
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    //we use this code because of potential fractional cases
    //instead of dim3 blocksPerGrid((colsB / BLOCK_SIZE) + 1, (rowsA / BLOCK_SIZE) + 1);
    dim3 blocksPerGrid((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMultiplySharedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Allocate memory for the result matrix on the host
    vector<float> flatC(rowsA * colsB);

    // Copy the result back to the host
    cudaMemcpy(flatC.data(), d_C, bytesC, cudaMemcpyDeviceToHost);

    Matrix C = unflattenMatrix(flatC, rowsA, colsB);


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return C;
}

// Function to verify that two matrices are the same
bool verifyMatrices(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        cerr << "Error: Matrices have different dimensions." << endl;
        return false;
    }

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            if (abs(A[i][j] - B[i][j]) > 1e-1) {
                cerr << "Error: Mismatch at element (" << i << ", " << j << "): "
                     << "A=" << A[i][j] << ", B=" << B[i][j] << endl;
                return false;
            }
        }
    }

    return true;
}

// Function to print a matrix
void printMatrix(const Matrix& matrix, const string& label) {
    cout << label << ":\n";
    for (const auto &row : matrix) {
        for (const auto &val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void testSuite(int size, float min, float max, int matMin, int matMax) {
    Matrix A, B, C, D, E;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> matSizeDist(matMin, matMax);

    for (size_t i = 0; i < size; ++i) {
        // Generate random dimensions for matrices A and B
        int rowsA = matSizeDist(gen);
        int colsA = matSizeDist(gen);
        int colsB = matSizeDist(gen);

        cout << "Matrix Dimensions - A: " << rowsA << "x" << colsA
             << ", B: " << colsA << "x" << colsB << endl;

        // Generate random matrices A and B
        generateMatrixCUDA(A, rowsA, colsA, min, max);
        generateMatrixCUDA(B, colsA, colsB, min, max);

        // Measure CPU multiplication time
        auto start = std::chrono::high_resolution_clock::now();
        // C = matrixMultiplyCPU(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> cpuDuration = end - start;
        // cout << "CPU Matrix Multiplication Time: " << cpuDuration.count() << " seconds\n";

        // Measure CUDA Global memory multiplication time
        start = std::chrono::high_resolution_clock::now();
        D = matrixMultiplyCUDA_Global(A, B);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cudaGlobalDuration = end - start;
        cout << "CUDA Global Matrix Multiplication Time: " << cudaGlobalDuration.count() << " seconds\n";

        // Measure CUDA Shared memory multiplication time
        start = std::chrono::high_resolution_clock::now();
        E = matrixMultiplyCUDA_Shared(A, B);
        end = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> cudaSharedDuration = end - start;
        cout << "CUDA Shared Matrix Multiplication Time: " << cudaSharedDuration.count() << " seconds\n";

        // // Verify the results
        // if (verifyMatrices(C, D)) {
        //     cout << "C & D Match!" << endl;
        // } else {
        //     cout << "C & D Do Not Match!" << endl;
        // }

        // if (verifyMatrices(C, E)) {
        //     cout << "C & E Match!" << endl;
        // } else {
        //     cout << "C & E Do Not Match!" << endl;
        // }

        if (verifyMatrices(D, E)) {
            cout << "D & E Match!" << endl;
        } else {
            cout << "D & E Do Not Match!" << endl;
        }

        cout << "-------------------------------------------" << endl;
    }
}

vector<float> flattenMatrix(const Matrix& matrix) {
    vector<float> flatMatrix;
    for (const auto &row : matrix) {
        flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }
    return flatMatrix;
}

Matrix unflattenMatrix(vector<float>& flatMatrix, int rows, int cols) {
    Matrix matrix(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flatMatrix[i * cols + j];
        }
    }
    return matrix;
}

__device__ float pseudoRandomNumber(int seed, int id, float min, float max) {
    int hash = id ^ seed;
    hash = (hash * 0x1e35a7bd) & 0xffffffff;
    float normalized = (hash / (float)UINT_MAX); // Normalize to [0, 1)
    return min + normalized * (max - min);       // Scale to [min, max]
}

__global__ void generateMatrixCUDAKernel(float *matrix, int rows, int cols, float min, float max, int seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Establish global ID
    int globalId = threadIdx.x + blockIdx.x * blockDim.x +
                   (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x;

    if (row < rows && col < cols) {
        matrix[row * cols + col] = pseudoRandomNumber(seed, globalId, min, max);
    }
}


void generateMatrixCUDA(Matrix &A, int rows, int cols, float min, float max) {
    size_t bytes = rows * cols * sizeof(float);
    float *d_matrix;

    // Allocate memory on the device
    cudaMalloc(&d_matrix, bytes);

    // Launch kernel
    dim3 threadsPerBlock(16, 16); // 256 threads
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    int seed = 1234; // Example seed value
    generateMatrixCUDAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols, min, max, seed);

    // Create a flattened matrix
    vector<float> flatMatrix(rows * cols);

    // Copy the result back to the host
    cudaMemcpy(flatMatrix.data(), d_matrix, bytes, cudaMemcpyDeviceToHost);

    // Reshape the flattened matrix into 2D
    A = unflattenMatrix(flatMatrix, rows, cols);

    // Free device memory
    cudaFree(d_matrix);
}



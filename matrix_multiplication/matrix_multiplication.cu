#include "matrix_multiplication.cuh"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

// Main Function
int main() {
    testSuite(10, 0, 10);
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
    flatA = flattenMatrix(A);
    flatB = flattenMatrix(B);

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

    unflattenMatrix(C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// CUDA shared memory kernel for matrix multiplication
__global__ void matrixMultiplySharedKernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    // Implementation will go here
}

// CUDA shared memory implementation of matrix multiplication
Matrix matrixMultiplyCUDA_Shared(const Matrix &A, const Matrix &B) {
    // Implementation will go here
    return Matrix(); // Placeholder
}

// Function to verify that two matrices are the same
bool verifyMatrices(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        cerr << "Error: Matrices have different dimensions." << endl;
        return false;
    }

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            if (abs(A[i][j] - B[i][j]) > 1e-5) {
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

// Test suite for matrix multiplication
void testSuite(int size, float min, float max) {
    Matrix A, B;

    generateMatrix(A, 2, 3, min, max);
    generateMatrix(B, 3, 2, min, max);

    Matrix C = matrixMultiplyCPU(A, B);

    printMatrix(A, "Matrix A");
    printMatrix(B, "Matrix B");
    printMatrix(C, "Matrix C");

    // if(verifyMatrices(A, B)){
    //     cout << "Match!";
    // }

}

vector<float> flattenMatrix(const Matrix &matrix) {
    vector<float> flatMatrix;
    for (const auto &row : matrix) {
        flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }
    return flatMatrix;
}


Matrix unflattenMatrix(const std::vector<float> &flatMatrix, int rows, int cols) {
    Matrix matrix(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flatMatrix[i * cols + j];
        }
    }
    return matrix;
}

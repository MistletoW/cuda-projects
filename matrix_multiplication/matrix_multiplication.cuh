#ifndef MATRIX_MULTIPLICATION_CUH
#define MATRIX_MULTIPLICATION_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

using namespace std;

// Matrix type definition
using Matrix = vector<vector<float>>;

// Function prototypes

// Generate a random matrix with given rows, columns, min, and max values
void generateMatrix(Matrix& matrix, int rows, int cols, float min, float max);

// CPU implementation of matrix multiplication
Matrix matrixMultiplyCPU(const Matrix& A, const Matrix& B);

// CUDA global memory implementation of matrix multiplication
__global__ void matrixMultiplyGlobalKernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB);
Matrix matrixMultiplyCUDA_Global(const Matrix& A, const Matrix& B);

// CUDA shared memory implementation of matrix multiplication
__global__ void matrixMultiplySharedKernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB);
Matrix matrixMultiplyCUDA_Shared(const Matrix& A, const Matrix& B);

// Verify that two matrices are the same (within a small tolerance)
bool verifyMatrices(const Matrix& A, const Matrix& B);

// Print a matrix
void printMatrix(const Matrix &matrix, const string& label);

// Test suite for matrix multiplication
void testSuite(int size, float min, float max, int matMin, int matMax);

//turn out 2d Matrix into a 1d matrix
vector<float> flattenMatrix(const Matrix& matrix);

//turn our 1d matrix into a 2d matrix
Matrix unflattenMatrix(vector<float>& flatMatrix, int rows, int cols);

#endif 
void generateTensor(Tensor& tensor, int depth, int rows, int cols, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    tensor.resize(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                tensor[d][i][j] = dis(gen);
            }
        }
    }
}

bool verifyTensors(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size() || A[0][0].size() != B[0][0].size()) {
        std::cerr << "Error: Tensors have different dimensions." << std::endl;
        return false;
    }

    for (size_t d = 0; d < A.size(); ++d) {
        for (size_t i = 0; i < A[d].size(); ++i) {
            for (size_t j = 0; j < A[d][i].size(); ++j) {
                if (std::abs(A[d][i][j] - B[d][i][j]) > 1e-5) {
                    std::cerr << "Error: Mismatch at element (" << d << ", " << i << ", " << j << "): "
                              << "A=" << A[d][i][j] << ", B=" << B[d][i][j] << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

void printTensor(const Tensor& tensor, const std::string& label) {
    std::cout << label << ":\n";
    for (size_t d = 0; d < tensor.size(); ++d) {
        std::cout << "Depth " << d << ":\n";
        for (const auto& row : tensor[d]) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }
}

std::vector<float> flattenTensor(const Tensor& tensor) {
    std::vector<float> flatTensor;
    for (const auto& matrix : tensor) {
        for (const auto& row : matrix) {
            flatTensor.insert(flatTensor.end(), row.begin(), row.end());
        }
    }
    return flatTensor;
}

Tensor unflattenTensor(const std::vector<float>& flatTensor, int depth, int rows, int cols) {
    Tensor tensor(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));
    int idx = 0;
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                tensor[d][i][j] = flatTensor[idx++];
            }
        }
    }
    return tensor;
}


__global__ void generateTensorCUDAKernel(float* tensor, int depth, int rows, int cols, float min, float max, int seed) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth && row < rows && col < cols) {
        int id = d * rows * cols + row * cols + col;
        int hash = id ^ seed;
        hash = (hash * 0x1e35a7bd) & 0xffffffff;
        float normalized = (hash / (float)UINT_MAX);
        tensor[d * rows * cols + row * cols + col] = min + normalized * (max - min);
    }
}

void generateTensorCUDA(Tensor& tensor, int depth, int rows, int cols, float min, float max) {
    size_t bytes = depth * rows * cols * sizeof(float);
    float* d_tensor;

    cudaMalloc(&d_tensor, bytes);

    dim3 threadsPerBlock(8, 8, 8); // 512 threads
    dim3 blocksPerGrid((cols + 7) / 8, (rows + 7) / 8, (depth + 7) / 8);

    int seed = 1234;
    generateTensorCUDAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tensor, depth, rows, cols, min, max, seed);

    std::vector<float> flatTensor(depth * rows * cols);
    cudaMemcpy(flatTensor.data(), d_tensor, bytes, cudaMemcpyDeviceToHost);

    tensor = unflattenTensor(flatTensor, depth, rows, cols);

    cudaFree(d_tensor);
}

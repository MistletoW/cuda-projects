# CUDA Projects Portfolio

Welcome to my CUDA Projects repository! This repository highlights two foundational CUDA programming projects designed to demonstrate GPU optimization techniques: **Vector Addition** and **Matrix Multiplication**.

## **Table of Contents**
- [Overview](#overview)
- [Vector Addition](#vector-addition)
  - [Project Description](#project-description)
  - [Features](#features)
  - [How to Run](#how-to-run)
- [Matrix Multiplication](#matrix-multiplication)
  - [Project Description](#project-description-1)
  - [Features](#features-1)
  - [How to Run](#how-to-run-1)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

---

## **Overview**

These projects are part of my journey in learning CUDA programming and GPU optimization techniques. By leveraging the power of NVIDIA GPUs, these projects demonstrate improvements in computational efficiency compared to traditional CPU-based implementations.

---

## **Vector Addition**

### **Project Description**
The Vector Addition project implements a simple addition of two vectors, leveraging CUDA kernels to parallelize computations across GPU threads. The project showcases how to structure and launch kernels efficiently for basic operations.

### **Features**
- Host and device memory management.
- Parallel computation using CUDA threads.
- Performance comparison between GPU and CPU implementations.

### **How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cuda-projects.git
   cd cuda-projects/vector-addition
   ```
2. Compile the code using `nvcc`:
   ```bash
   nvcc vector_addition.cu -o vector_add
   ```
3. Run the executable:
   ```bash
   ./vector_add
   ```
4. Observe the output for performance metrics and results.

---

## **Matrix Multiplication**

### **Project Description**
The Matrix Multiplication project performs matrix-matrix multiplication using CUDA. It demonstrates the use of shared memory and thread synchronization for optimizing memory access patterns.

### **Features**
- Efficient memory coalescing techniques.
- Use of shared memory to minimize global memory accesses.
- Comparison of naive and optimized implementations.

### **How to Run**
1. Navigate to the project directory:
   ```bash
   cd cuda-projects/matrix-multiplication
   ```
2. Compile the code using `nvcc`:
   ```bash
   nvcc matrix_multiplication.cu -o matrix_mult
   ```
3. Run the executable:
   ```bash
   ./matrix_mult
   ```
4. Review the performance metrics in the output.

---

## **Technologies Used**
- **CUDA Toolkit**: GPU programming framework.
- **C++**: Base language for CUDA development.
- **NVIDIA GPUs**: Hardware platform for testing and benchmarking.

---

## **Future Work**
- Implement more advanced GPU optimizations, such as warp-level primitives.
- Extend the matrix multiplication project to support larger datasets.
- Add visualization tools to display computational results.

---

Feel free to explore, use, and comment on these projects! If you encounter any issues or have suggestions, please open an issue or create a pull request.

**Contact**: Dylan Haase (haasedylan@gmail.com) | [LinkedIn](https://www.linkedin.com/in/dylan-haase-b72827279/)

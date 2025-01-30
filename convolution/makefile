# Compiler
NVCC = nvcc

# Output binary name
TARGET = convolve1D

# Source files
SRC = tensor.cu 1d_convolution.cu

# Object files
OBJ = $(SRC:.cu=.o)

# CUDA flags
NVCC_FLAGS = -arch=sm_60 -std=c++11 -O3 -Xcompiler "-Wall"

# Clean existing build files
clean:
	rm -rf $(TARGET) $(OBJ)

# Build object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link everything and create the final executable
build: $(OBJ)
	$(NVCC) -o $(TARGET) $(OBJ) -lstdc++

# Run the program after building
run: build
	./$(TARGET)

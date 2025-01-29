#include "1d_convolution.cuh"
#include "tensor.cuh"

using namespace std;

int main(){

}

Tensor convolve1D(Tensor& input, Tensor& kernel){
    if(input.getWidth() > kernel.getWidth()){
        throw runtime_error("Input tensor width must be greater than kernel width.");
    }

    if(input.getHeight() != 1 || kernel.getHeight() != 1){
        throw runtime_error("Height of both input and kernel tensors must be 1.");
    }

    if(input.getDepth() != 1 || kernel.getDepth() != 1){
        throw runtime_error("Depth of input and kernel tensors must be 1.");
    }
}
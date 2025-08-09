#include <stdio.h>
#include <math.h>
#include "globalTypes.h"

/*
| Category         | Examples                             |
| ---------------- | ------------------------------------ |
| Arithmetic       | `+`, `-`, `*`, `/`, `pow`, etc.      |
| Activation funcs | `relu`, `sigmoid`, `tanh`, `softmax` |
| Matrix ops       | `dot`, `matmul`, `transpose`         |
| Reductions       | `sum`, `mean`, `max`, `min`          |
| Gradients        | `d_relu`, `d_sigmoid`, etc.          |
| Utilities        | `normalize`, `clip`, `scale`         |
*/
//k is short for kernel 
//arithmetic:
__global__ void addk(float a[], float b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] + b[i];
    }
}
__global__ void subk(float a[], float b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] - b[i];
    }
}
__global__ void multiplyk(float a[], float b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] * b[i];
    }
}
__global__ void dividek(float a[], float b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] / b[i];
    }
}
__global__ void powk(float a[], int b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i];
        for (int j = 0; j <= b[i]; j++) {
            c[i] *= b[i];
        }
    }
}
__global__ void sumk(float a[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[0] += a[i];
    }
}
//simple algorithms:
//will finish later
__global__ void linearRegressionk(float a[], float b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        float theta = 0;
        
    }
}
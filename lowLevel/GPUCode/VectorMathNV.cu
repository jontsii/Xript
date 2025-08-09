#include <cuda_runtime.h>
#include <string.h>
#include "globalTypes.h"

typedef char* string;

__global__ void v3Addk(V3 a[], V3 b[], V3* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i].x = a[i].x + b[i].x;
        c[i].y = a[i].y + b[i].y;
        c[i].z = a[i].z + b[i].z;
    }
}
__global__ void v3Subk(V3 a[], V3 b[], V3* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i].x = a[i].x - b[i].x;
        c[i].y = a[i].y - b[i].y;
        c[i].z = a[i].z - b[i].z;
    }
}
__global__ void v3Dotk(V3 a[], V3 b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
    }
}
__global__ void v3Crossk(V3 a[], V3 b[], V3* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i].x = a[i].y * b[i].z - a[i].z * b[i].y;
        c[i].y = a[i].z * b[i].x - a[i].x * b[i].z;
        c[i].z = a[i].x * b[i].y - a[i].y * b[i].x;
    }                 
}
__global__ void v3Magnitudek(V3 a[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y + a[i].z * a[i].z);
    }
}
__global__ void v3Normalizek(V3 a[], V3* c, float mags[], int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        if (!mags[i] == 0.0f) {
            c[i].x = a[i].x / mags[i];
            c[i].y = a[i].y / mags[i];
            c[i].z = a[i].z / mags[i];
        }
        else {
            c[i] = a[i];
        }
    }
}
__global__ void v3Scalek(V3 a[], float scalar[], V3* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i].x = a[i].x * scalar[i];
        c[i].y = a[i].y * scalar[i];
        c[i].z = a[i].z * scalar[i];
    }
}
__global__ void v3Distancek(V3 a[], V3 b[], float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = sqrtf(powf((a[i].x - b[i].x), 2) + powf((a[i].y - b[i].y), 2) + powf((a[i].z - b[i].x), 2));
    }
}

//kernel launchers
//optimized for max performance on startup
extern "C" __declspec(dllexport) __host__ V3* v3Add(V3 a[], V3 b[], int amount) { //manages kernel calls, GPU memory and returning the answers
    float difficulty = 1.5f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = amount / blocks;

    V3* destA;
    V3* destB;
    V3* destC;

    cudaMalloc((void**)&destA, sizeof(V3) * amount); //alocate VRAM for arguments
    cudaMalloc((void**)&destB, sizeof(V3) * amount);
    cudaMalloc((void**)&destC, sizeof(V3) * amount);

    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);
    cudaMemcpy(destB, b, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Addk<<<blocks, threadsPerBlock>>>(destA, destB, destC, amount); //kernel call
    
    V3* hostC = (V3*)malloc(sizeof(V3) * amount);

    cudaMemcpy(hostC, destC, sizeof(V3) * amount, cudaMemcpyDeviceToHost);

    cudaFree(destC);
    cudaFree(destA);
    cudaFree(destB);
    return hostC;
}

extern "C" __declspec(dllexport) __host__ V3* v3Sub(V3 a[], V3 b[], int amount) { //manages kernel calls, GPU memory and returning the answers
    float difficulty = 2.2f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = amount / blocks;

    V3* destA;
    V3* destB;
    V3* destC;

    cudaMalloc((void**)&destA, sizeof(V3) * amount); //alocate ram for 
    cudaMalloc((void**)&destB, sizeof(V3) * amount);
    cudaMalloc((void**)&destC, sizeof(V3) * amount);

    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);
    cudaMemcpy(destB, b, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Subk<<<blocks, threadsPerBlock>>>(destA, destB, destC, amount); //kernel call
    
    V3* hostC = (V3*)malloc(sizeof(V3) * amount);

    cudaMemcpy(hostC, destC, sizeof(V3) * amount, cudaMemcpyDeviceToHost);

    cudaFree(destA);
    cudaFree(destB);
    cudaFree(destC);
    return hostC;
}

extern "C" __declspec(dllexport) __host__ V3* v3Normalize(V3 a[], int amount) { //manages kernel calls, GPU memory and returning the answers
    float difficulty = 4.0f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = amount / blocks;

    V3* destA;
    V3* destC;
    float* destMag;
    
    cudaMalloc((void**)&destA, sizeof(V3) * amount); //alocate VRAM for arguments
    cudaMalloc((void**)&destC, sizeof(V3) * amount);
    cudaMalloc((void**)&destMag, sizeof(float) * amount);

    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Magnitudek<<<blocks, threadsPerBlock>>>(a, destMag, amount);
    v3Normalizek<<<blocks, threadsPerBlock>>>(destA, destC, destMag, amount); //kernel call
    
    V3* hostC = (V3*)malloc(sizeof(V3) * amount);

    cudaMemcpy(hostC, destC, sizeof(V3) * amount, cudaMemcpyDeviceToHost);

    cudaFree(destA);
    cudaFree(destMag);
    cudaFree(destC);
    return hostC;
}

extern "C" __declspec(dllexport) __host__ V3* v3Scale(V3 a[], float scalar[], int amount) { //manages kernel calls, GPU memory and returning the answers
    float difficulty = 3.4f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = amount / blocks;

    V3* destA;
    V3* destC;

    cudaMalloc((void**)&destA, sizeof(V3) * amount); //alocate VRAM for arguments
    cudaMalloc((void**)&destC, sizeof(V3) * amount);

    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Scalek<<<blocks, threadsPerBlock>>>(destA, scalar, destC, amount); //kernel call
    
    V3* hostC = (V3*)malloc(sizeof(V3) * amount);

    cudaMemcpy(hostC, destC, sizeof(V3) * amount, cudaMemcpyDeviceToHost);

    cudaFree(destC);
    cudaFree(destA);
    return hostC;
}
extern "C" __declspec(dllexport) __host__ float* v3Dot(V3 a[], V3 b[], int amount) {
    float difficulty = 2.4f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = ceil(amount / blocks);
    
    V3* destA;
    V3* destB;
    float* destC;

    cudaMalloc((void**)&destA, sizeof(V3) * amount);
    cudaMalloc((void**)&destB, sizeof(V3) * amount);
    cudaMalloc((void**)&destC, sizeof(float) * amount);
    
    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);
    cudaMemcpy(destB, b, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Dotk<<<blocks, threadsPerBlock>>>(destA, destB, destC, amount);

    float* hostC = (float*)malloc(sizeof(float) * amount);
    cudaMemcpy(hostC, destC, sizeof(float) * amount, cudaMemcpyDeviceToHost);
    
    cudaFree(destA);
    cudaFree(destB);
    cudaFree(destC);
    return hostC;
}
extern "C" __declspec(dllexport) __host__ float* v3Magnitude(V3 a[], V3 b[], int amount) {
    float difficulty = 2.7f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = ceil(amount / blocks);
    
    V3* destA;

    float* destC;

    cudaMalloc((void**)destA, sizeof(V3) * amount);
    cudaMalloc((void**)destC, sizeof(float) * amount);
    
    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Magnitudek<<<blocks, threadsPerBlock>>>(destA, destC, amount);

    float* hostC = (float*)malloc(sizeof(float) * amount);
    cudaMemcpy(hostC, destC, sizeof(float) * amount, cudaMemcpyDeviceToHost);
    
    cudaFree(destA);
    cudaFree(destC);
    return hostC;
}
extern "C" __declspec(dllexport) __host__ float* v3Distance(V3 a[], V3 b[], int amount) {
    float difficulty = 2.4f;
    int blocks = ceil(amount / difficulty);
    int threadsPerBlock = ceil(amount / blocks);
    
    V3* destA;
    V3* destB;
    float* destC;
    
    cudaMalloc((void**)destA, sizeof(V3) * amount);
    cudaMalloc((void**)destB, sizeof(V3) * amount);
    cudaMalloc((void**)destC, sizeof(float) * amount);
    
    cudaMemcpy(destA, a, sizeof(V3) * amount, cudaMemcpyHostToDevice);
    cudaMemcpy(destB, b, sizeof(V3) * amount, cudaMemcpyHostToDevice);

    v3Distancek<<<blocks, threadsPerBlock>>>(destA, destB, destC, amount);

    float* hostC = (float*)malloc(sizeof(float) * amount);
    cudaMemcpy(hostC, destC, sizeof(float) * amount, cudaMemcpyDeviceToHost);
    
    cudaFree(destA);
    cudaFree(destB);
    cudaFree(destC);
    return hostC;
}
/*notes:
  GPU functions will be copy-pasted to the final executable from a dll, the block amount and threads per block will be calculated before called
  formula is:
   blocksAmount = amount of processes / difficulty. 
   threadPerBlock = amount of processes / blocksAmount.
  c[] is always empty except if said otherwise
*/ 
//make a __host__ function to launch kernels and be the GC
//#include "globalTypes.h"
#include <cuda_runtime.h>
#include <typeinfo>

//make desicion table for difficulties
typedef struct {
    int xLen;
    int yLen;
    int size;
    float* data;
} M2;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int size;
    float* data;
} M3;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int aLen;
    int size;
    float* data;
} M4;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int aLen;
    int bLen;
    int size;
    float* data;
} M5;

typedef struct {float addition = 0.1f;
                float subtraction = 0.1f;
                float multiplication = 0.1f;
                float division = 0.1f;
                float pow = 0.18f;
                float sqrt = 0.22f;
                float cosine = 0.2f;
                float exp = 0.23f;
                float sine = 0.18f;
                float log = 0.18f;
                float sum = 0.2f;
                float sigmoid = 0.23f;
                float add = 0.1f;
                float sub = 0.1f;
                float div = 0.1f;
                float mul = 0.1f;} diffTable;
//define creation functions for matrices
__host__ M2 M2Create(int xLen, int yLen) {
    M2 matrix = M2{xLen, yLen, NULL};
    matrix.data = (float*)malloc(sizeof(float) * xLen * yLen);  
    return matrix;
}
__host__ M3 M3Create(int xLen, int yLen, int zLen) {
    M3 matrix = M3{xLen, yLen, zLen, NULL};
    matrix.data = (float*)malloc(sizeof(float) * xLen * yLen * zLen);  
    return matrix;
}
__host__ M4 M4Create(int xLen, int yLen, int zLen, int aLen) {
    M4 matrix = M4{xLen, yLen, zLen, aLen, NULL};
    matrix.data = (float*)malloc(sizeof(float) * xLen * yLen * zLen * aLen);  
    return matrix;
}
__host__ M5 M5Create(int xLen, int yLen, int zLen, int aLen, int bLen) {
    M5 matrix = M5{xLen, yLen, zLen, aLen, bLen, NULL};
    matrix.data = (float*)malloc(sizeof(float) * xLen * yLen * zLen * aLen * bLen);  
    return matrix;
}

//function bodies
void MMultiplyk(M2 a, M2 b, M2* c, int amount) {
//Finish later
}
void MTransposek(M2 a, M2* c, int amount) {
//finish later
}

__global__ void MAdditionk(float* a, float* b, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] + b[i];
    }
}
__global__ void MSubtractionk(float* a, float* b, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] - b[i];
    }
}
__global__ void MMultiplicationk(float* a, float* b, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] * b[i];
    }
}
__global__ void MDivisionk(float* a, float* b, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] / b[i];
    }
}
__global__ void MPowk(float* a, int b, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = __powf(a[i], b);
    }
}
__global__ void MSqrtk(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = sqrtf(a[i]);
    }
}
__global__ void MSinek(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = __sinf(a[i]);
    }
}
__global__ void MCosinek(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = __cosf(a[i]);
    }
}
__global__ void MExpk(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = __expf(a[i]);
    }
}
__global__ void MLogk(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = __logf(a[i]);
    }
}
__host__ float MAccess(M2 a, int row, int column) {
    return a.data[row * a.yLen + column];
}

__global__ void MSumk(float* a, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[0] += a[i];
    }
}
__global__ void MSigmoidk(float* exp, float* c, int amount) { //exp is calculated in the kernel launcher and passed as an argument to here for performance
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = 1 / (1 + exp[i]);
    }
}
__host__ float MMean(M2 a, int amount) { //remember to copy data too!!! 
    M2 destB;
    float* destData;
    float destC;
    cudaMalloc((void**)&destB, sizeof(M2));
    cudaMalloc((void**)&destC, sizeof(float));
    cudaMalloc((void**)&destData, sizeof(float) * a.size);

    cudaMemcpy(&destB, &a, sizeof(M2), cudaMemcpyHostToDevice);
    cudaMemcpy(&destData, a.data, sizeof(float) * a.yLen * a.xLen, cudaMemcpyHostToDevice);

    MSumk<<<ceil(amount / 1024), ceil(amount / (amount / 1024))>>>(destB.data, &destC, amount);

    float result;
    cudaMemcpy(&result, &destC, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&destB);
    cudaFree(&destData);
    cudaFree(&destC);

    return result / (a.xLen * a.yLen);
}
//the next 4 kernels are overloaded
__global__ void MAddk(float* a, float n, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] + n;
    }
}
__global__ void MSubk(float* a, float n, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] - n;
    }
}
__global__ void MMulk(float* a, float n, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] * n;
    }
}
__global__ void MDivk(float* a, float n, float* c, int amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amount) {
        c[i] = a[i] / n;
    }
}
typedef enum {ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, POW, SQRT, COSINE, EXPONENTIAL,
              SINE, LOGARITHMIC, SUMMARY, SIGMOID, ADD, SUB, MUL, DIV} __declspec(dllexport) kernel;

int* calculateBlocksAndThreads(int amount, float difficulty) { //this returns an int* with 2 items: index 0 is blocks and index 1 is threads per block.
    int blocks = ceil(amount * difficulty);
    int threadsPerBlock = ceil(amount / blocks);
    
    if (threadsPerBlock > 1024) { //check if threadsPerBlock is over 1024, CUDA only supports up to 1024 threads per block
        blocks = ceil(amount / 1024);
    }
    int* ans = (int*)malloc(sizeof(int) * 2);
    ans[0] = blocks;
    ans[1] = threadsPerBlock;
    return ans;
}
M3 kernelLauncher(M3 a, M3 b, M3 c, float n, int d, kernel op); //make helper function to pass C correctly (c is a matrix pointer, it points to the correct matrix)
M4 kernelLauncher(M4 a, M4 b, M4 c, float n, int d, kernel op);
M5 kernelLauncher(M5 a, M5 b, M5 c, float n, int d, kernel op);

extern "C" __declspec(dllexport) M2 kernelLauncher(M2 a, M2 b, M2 c, float n, int d, kernel op) {
    float* destBData;
    float destN;
    int destD;

    diffTable table = diffTable{};
    float difficulty;
    int* initData;
    //allocate memory and copy it to VRAM
    if (b.data != NULL) {
        cudaMalloc((void**)&destBData, sizeof(float) * b.size);
        cudaMemcpy(destBData, b.data, sizeof(float) * b.size, cudaMemcpyHostToDevice);
    }
    if (n != NULL) {
        cudaMalloc((void**)&destN, sizeof(float));
        cudaMemcpy(&destN, &n, sizeof(float), cudaMemcpyHostToDevice);
    }
    if (d != NULL) {
        cudaMalloc((void**)&destD, sizeof(int));
        cudaMemcpy(&destD, &n, sizeof(int), cudaMemcpyHostToDevice);
    }
    float* destAData;
    cudaMalloc((void**)&destAData, sizeof(float) * a.size);
    cudaMemcpy(destAData, a.data, sizeof(float) * a.size, cudaMemcpyHostToDevice);
    //copy c´s data to VRAM
    float* destCData;
    cudaMalloc((void**)&destCData, sizeof(float) * a.size);

    switch (op) {
        case ADDITION:
            initData = calculateBlocksAndThreads(a.size, table.addition);
            MAdditionk<<<initData[0], initData[1]>>>(destAData, destBData, destCData, a.size); //read back values from VRAM later
            break;
        case SUBTRACTION:
            difficulty = 0.1f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MSubtractionk<<<initData[0], initData[1]>>>(destAData, destBData, destCData, a.size);
            break;
        case MULTIPLICATION:
            difficulty = 0.12f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MMultiplicationk<<<initData[0], initData[1]>>>(destAData, destBData, destCData, a.size);
            break;
        case DIVISION:
            difficulty = 0.15f; //0.15 because computers SUCK at division
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MDivisionk<<<initData[0], initData[1]>>>(destAData, destBData, destCData, a.size);
            break;
        case POW:
            difficulty = 0.18f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MPowk<<<initData[0], initData[1]>>>(destAData, destD, destCData, a.size);
            break;
        case SQRT:
            difficulty = 0.22f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MSqrtk<<<initData[0], initData[1]>>>(destAData, destCData, a.size);
            break;
        case COSINE:
            difficulty = 0.2f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MCosinek<<<initData[0], initData[1]>>>(destAData, destCData, a.size);
            break;
        case EXPONENTIAL:
            difficulty = 0.23f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MExpk<<<initData[0], initData[1]>>>(destAData, destCData, a.size);
            break;
        case SINE:
            difficulty = 0.18f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MSinek<<<initData[0], initData[1]>>>(destAData, destCData, a.size);
            break;
        case LOGARITHMIC:
            difficulty = 0.18f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MLogk<<<initData[0], initData[1]>>>(destAData, destCData, a.size);
            break;
        case SUMMARY:
            difficulty = 0.2f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            float* sum;
            cudaMalloc((void**)&sum, sizeof(float) * a.size);
            MSumk<<<initData[0], initData[1]>>>(destAData, sum, a.size);
            break;
        case SIGMOID:
            difficulty = 0.23f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            float* exp;
            cudaMalloc((void**)&exp, sizeof(float) * a.size);
            MExpk<<<initData[0], initData[1]>>>(destAData, exp, a.size);
            difficulty = 0.24;
            MSigmoidk<<<initData[0], initData[1]>>>(exp, destCData, a.size);
            break;
        case ADD:
            difficulty = 0.1f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MAddk<<<initData[0], initData[1]>>>(destAData, destN, destCData, a.size);
            break;
        case SUB:
            difficulty = 0.1f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MSubk<<<initData[0], initData[1]>>>(destAData, destN, destCData, a.size);
            break;
        case MUL:
            difficulty = 0.1f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MMulk<<<initData[0], initData[1]>>>(destAData, destN, destCData, a.size);
            break;
        case DIV:
            difficulty = 0.1f;
            initData = calculateBlocksAndThreads(a.size, difficulty);
            MDivk<<<initData[0], initData[1]>>>(destAData, destN, destCData, a.size);
            break;
    }
    //readback from VRAM and cleanup
    cudaMemcpy(c.data, destCData, sizeof(float) * a.size, cudaMemcpyDeviceToHost);

    cudaFree(destAData);
    cudaFree(destCData);
    if (b.data != NULL) {
        cudaFree(destBData);
    }
    if (n != NULL) {
        cudaFree(&destN);
    }
    if (d != NULL) {
        cudaFree(destAData);
    }
    free(initData);
    return c;
}
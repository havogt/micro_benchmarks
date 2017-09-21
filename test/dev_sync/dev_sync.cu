#include <iostream>
#include <stdio.h>

__global__ void child_launch(int* counter) { atomicAdd(counter, 1); }

__device__ void syncBlock(volatile unsigned int* lock, int cnt)
{
    atomicAdd((unsigned int*)lock, 1);
    while(gridDim.x * cnt != *lock)
        ;
}

class BlockSyncer {
private:
    bool dir;
    volatile unsigned int* lock_var;

public:
    __device__ BlockSyncer(volatile unsigned int* lock_var)
        : dir(false)
        , lock_var(lock_var){};
    __device__ void sync()
    {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            dir = !dir;
            atomicAdd((unsigned int*)lock_var, dir ? 1 : -1);
            while(gridDim.x * gridDim.y * gridDim.z * dir != *lock_var)
                ;
        }
        __syncthreads();
    }
};

__global__ void test_kernel(volatile unsigned int* counter)
{
    BlockSyncer blk(counter);
    if(blockIdx.x == 0 && threadIdx.x == 0)
        printf("counter = %d, gridDim.x = %d\n", *counter, gridDim.x);
    blk.sync();
    if(blockIdx.x == 0 && threadIdx.x == 0)
        printf("counter = %d\n", *counter);
    blk.sync();
    if(blockIdx.x == 0 && threadIdx.x == 0)
        printf("counter = %d\n", *counter);
}

int main()
{
    unsigned int* counter;
    cudaMalloc(&counter, sizeof(unsigned int));

    unsigned int zero = 0;
    cudaMemcpy(counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);

    test_kernel<<<32, 256>>>(counter);
    cudaDeviceSynchronize();
}

//__global__ void child_launch(int* data) { data[threadIdx.x] = data[threadIdx.x] + 1; }
//
//__global__ void parent_launch(int* data)
//{
//    data[threadIdx.x] = threadIdx.x;
//
//    __syncthreads();
//
//    if(threadIdx.x == 0) {
//        //        child_launch<<<1, 256>>>(data);
//        cudaDeviceSynchronize();
//    }
//
//    __syncthreads();
//}
//
// int main()
//{
//    int* data;
//    cudaMalloc(&data, sizeof(int) * 256);
//    parent_launch<<<1, 256>>>(data);
//}

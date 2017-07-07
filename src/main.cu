#include <iostream>
#include <stdio.h>

__global__ void test_kernel(int* tmp)
{
    printf("test from kernel\n");
    *tmp = 4;
}

int main()
{
    int* ptr;
    cudaMalloc(&ptr, sizeof(int));
    test_kernel<<<1, 1>>>(ptr);
    cudaDeviceSynchronize();
}

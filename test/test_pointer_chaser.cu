#include <iostream>
#include "src/pointer_chaser.hpp"

#include <boost/preprocessor/repetition/repeat.hpp>

#define MY_OPERATION(z, n, unused) p = (void*)*(pointer_chaser::ptr_type*)p;

__global__ void test_kernel(void** ptr, void* out)
{
    void* p = ptr[0];
    unsigned start = clock();
    //    BOOST_PP_REPEAT(256, MY_OPERATION, no_op)
    //    p = (void*)*(pointer_chaser::ptr_type*)p;
    printf("%p\n", p);
    unsigned stop = clock();
    out = p;
    printf("clocks: %d\n", (stop - start));
}

int main()
{
    pointer_chaser chase(1024 * 1024 * 1024, 1024 * 1024);
    std::cout << chase.max_steps << std::endl;
    std::cout << "host ptr: " << chase.h_array << std::endl;
    std::cout << "dev ptr: " << chase.d_array << std::endl;
    for(int i = 0; i < 10; ++i)
        std::cout << chase.h_array[i] << std::endl;

    // test 10 steps
    //    void** p = chase.first;
    //    for(int i = 0; i < 10; ++i) {
    //        p = *p;
    //    }
    //    std::cout << "pointer after 10 steps: " << *p << std::endl;

    void* out;
    cudaMalloc(&out, sizeof(void*));
    test_kernel<<<1, 1>>>(chase.d_array, out);
    cudaDeviceSynchronize();
}

#include <iostream>
#include <vector>

template <typename T>
__global__ void simple_fma(T* a, T* b, T* c, T* d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d[i] = a[i] * b[i] + c[i];
}

template <typename T>
__global__ void simple_mul(T* a, T* b, T* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}

template <typename T>
__global__ void simple_sub(T* a, T* b, T* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] - b[i];
}

template <typename T>
__global__ void simple_mul2(T* a, T* b, T* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}

template <int N, typename T>
__global__ void special_sin(T* a, T* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = sin(a[i]);
}

template <int N, typename T>
__global__ void special_exp(T* a, T* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = exp(a[i]);
}

__global__ void special_exp_intrinsic(float* a, float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = __expf(a[i]);
}

template <typename T>
void fill_all(T* data, size_t N, T value)
{
    std::vector<T> local(N);
    std::fill(local.begin(), local.end(), value);
    cudaMemcpy(data, local.data(), sizeof(T) * N, cudaMemcpyHostToDevice);
}

int main()
{
    using float_type = float;
    float_type* a;
    float_type* b;
    float_type* c;
    float_type* d;

    size_t blockSize = 32;
    size_t gridSize = 1;
    size_t N = blockSize * gridSize;

    cudaMalloc(&a, N * sizeof(float_type));
    cudaMalloc(&b, N * sizeof(float_type));
    cudaMalloc(&c, N * sizeof(float_type));
    cudaMalloc(&d, N * sizeof(float_type));

    std::cout << "Expected " << N << " fma\n" << std::endl;
    simple_fma<<<blockSize, gridSize>>>(a, b, c, d);
    cudaDeviceSynchronize();
    std::cout << "Expected " << N << " mul\n" << std::endl;
    simple_mul<<<blockSize, gridSize>>>(a, b, c);
    cudaDeviceSynchronize();
    std::cout << "Expected " << N << " add\n" << std::endl;
    simple_sub<<<blockSize, gridSize>>>(a, b, c);
    cudaDeviceSynchronize();

    std::cout << "Expected " << N / 2 << " mul\n" << std::endl;
    simple_mul2<<<blockSize / 2, gridSize>>>(a, b, c);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)1.0);
    special_sin<0><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)0.01);
    special_sin<1><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)3.1415);
    special_sin<2><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)0.0);
    special_exp<0><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)0.000001);
    special_exp<1><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)100000000);
    special_exp<2><<<1, 1>>>(a, b);
    cudaDeviceSynchronize();

    fill_all(a, 1, (float_type)1);
    special_exp_intrinsic<<<1, 1>>>(a, b);
    cudaDeviceSynchronize();
}

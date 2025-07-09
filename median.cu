#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <float.h>

#define THREADS 256
#define MAX_ITERS 1000

__device__ void device_min_max(const float* arr, int N, float& out_min, float& out_max) {
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
        float v = arr[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    __shared__ float block_min[THREADS];
    __shared__ float block_max[THREADS];

    block_min[threadIdx.x] = local_min;
    block_max[threadIdx.x] = local_max;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_min[threadIdx.x] = fminf(block_min[threadIdx.x], block_min[threadIdx.x + s]);
            block_max[threadIdx.x] = fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin((int*)&out_min, __float_as_int(block_min[0]));
        atomicMax((int*)&out_max, __float_as_int(block_max[0]));
    }
}

__global__ void median_device_kernel(const float* d_array, int N, float* d_median_out) {
    __shared__ float shared_min;
    __shared__ float shared_max;

    if (threadIdx.x == 0) {
        shared_min = FLT_MAX;
        shared_max = -FLT_MAX;
    }
    __syncthreads();

    device_min_max(d_array, N, shared_min, shared_max);
    __syncthreads();

    int k = N / 2;
    float pivot, min_val, max_val;

    if (threadIdx.x == 0) {
        min_val = shared_min;
        max_val = shared_max;
    }
    __syncthreads();

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        pivot = (min_val + max_val) * 0.5f;

        int local_less = 0, local_equal = 0, local_greater = 0;

        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
            float val = d_array[i];
            if (val < pivot) local_less++;
            else if (val > pivot) local_greater++;
            else local_equal++;
        }

        __shared__ int block_less[THREADS];
        __shared__ int block_equal[THREADS];
        __shared__ int block_greater[THREADS];

        block_less[threadIdx.x] = local_less;
        block_equal[threadIdx.x] = local_equal;
        block_greater[threadIdx.x] = local_greater;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_less[threadIdx.x] += block_less[threadIdx.x + s];
                block_equal[threadIdx.x] += block_equal[threadIdx.x + s];
                block_greater[threadIdx.x] += block_greater[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            int less = block_less[0];
            int equal = block_equal[0];

            if (k < less) {
                max_val = pivot;
            } else if (k < less + equal) {
                *d_median_out = pivot;
                return;
            } else {
                k -= (less + equal);
                min_val = pivot;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *d_median_out = pivot;  // fallback if max iters hit
    }
}

int main() {
    const int N = 10'000'000;
    float* h_array = new float[N];

    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < N; ++i)
        h_array[i] = static_cast<float>(std::rand()) / RAND_MAX;

    float* d_array;
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    float* d_median;
    float h_median;
    cudaMalloc(&d_median, sizeof(float));

    median_device_kernel<<<64, THREADS>>>(d_array, N, d_median);
    cudaMemcpy(&h_median, d_median, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Median: " << h_median << std::endl;

    cudaFree(d_array);
    cudaFree(d_median);
    delete[] h_array;

    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA kernel: reduce sum of a small array using shared memory
__global__ void reduce_sum(float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory (zero-padding if out-of-bounds)
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction loop (no warp-level code)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write final result to global memory
    if (tid == 0)
        *output = sdata[0];
}

int main() {
    const int N = 512; // Small array size (<= 1024)
    const int threads = 512; // Must be power of two and >= N

    // Allocate and initialize host array
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i)
        h_input[i] = 1.0f; // Example: sum should be N

    float h_output;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, `threads` threads, shared memory size = threads * sizeof(float)
    reduce_sum<<<1, threads, threads * sizeof(float)>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMem

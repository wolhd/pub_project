#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

__global__ void reduce_min(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = FLT_MAX;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val = fminf(val, input[i + blockDim.x]);
    sdata[tid] = val;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float find_min(float *h_input, int n) {
    float *d_input = nullptr, *d_output = nullptr;
    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    size_t sharedMemSize = threads * sizeof(float);

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, blocks * sizeof(float));

    reduce_min<<<blocks, threads, sharedMemSize>>>(d_input, d_output, n);

    // If needed, continue reducing until one value remains
    int num_elements = blocks;
    while (num_elements > 1) {
        int next_blocks = (num_elements + threads * 2 - 1) / (threads * 2);
        reduce_min<<<next_blocks, threads, sharedMemSize>>>(d_output, d_output, num_elements);
        num_elements = next_blocks;
    }

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return result;
}

int main() {
    const int N = 1 << 20;
    float *h_input = new float[N];

    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    h_input[12345] = -999.0f;  // insert known minimum

    float min_val = find_min(h_input, N);
    printf("Minimum value: %f\n", min_val);

    delete[] h_input;
    return 0;
}

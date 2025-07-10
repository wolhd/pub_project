#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

// Device function to merge two sorted subarrays
__device__ void mergeSequential(int *data, int *temp, int start, int mid, int end) {
    int i = start, j = mid, k = start;
    while (i < mid && j < end) {
        if (data[i] <= data[j])
            temp[k++] = data[i++];
        else
            temp[k++] = data[j++];
    }
    while (i < mid) temp[k++] = data[i++];
    while (j < end) temp[k++] = data[j++];
    for (int l = start; l < end; l++)
        data[l] = temp[l];
}

// Kernel: each thread block processes one merge segment
__global__ void mergeSortPassKernel(int *data, int *temp, int width, int n) {
    int tid = blockIdx.x;

    int start = tid * 2 * width;
    if (start >= n)
        return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    mergeSequential(data, temp, start, mid, end);
}

// Host function: Merge Sort for any array size
void mergeSort(int *h_data, int n) {
    int *d_data, *d_temp;
    size_t size = n * sizeof(int);

    cudaCheckError(cudaMalloc(&d_data, size));
    cudaCheckError(cudaMalloc(&d_temp, size));
    cudaCheckError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    for (int width = 1; width < n; width *= 2) {
        int numSegments = (n + (2 * width - 1)) / (2 * width);
        mergeSortPassKernel<<<numSegments, 1>>>(d_data, d_temp, width, n);
        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    cudaFree(d_data);
    cudaFree(d_temp);
}

// === Test ===

int main() {
    const int N = 1237;  // Arbitrary non-power-of-two size
    int *h_data = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        h_data[i] = rand() % 10000;

    printf("Unsorted sample:\n");
    for (int i = 0; i < 20; i++) printf("%d ", h_data[i]);
    printf("\n");

    mergeSort(h_data, N);

    printf("Sorted sample:\n");
    for (int i = 0; i < 20; i++) printf("%d ", h_data[i]);
    printf("\n");

    free(h_data);
    return 0;
}

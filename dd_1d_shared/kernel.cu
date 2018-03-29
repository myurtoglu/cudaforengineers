#include "kernel.h"
#define TPB 64
#define RAD 1 // radius of the stencil

__global__
void ddKernel(float *d_out, const float *d_in, int size, float h) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= size) return;

  const int s_idx = threadIdx.x + RAD;
  extern __shared__ float s_in[];

  // Regular cells
  s_in[s_idx] = d_in[i];

  // Halo cells
  if (threadIdx.x < RAD) {
    // careful: the two lines below will also access d_in[-1] and d_in[size+1] which 
    // are undefined! This bug is fixed in heat_2d (cf. idxClip function)
    s_in[s_idx - RAD] = d_in[i - RAD];
    s_in[s_idx + blockDim.x] = d_in[i + blockDim.x];
  }
  __syncthreads();
  d_out[i] = (s_in[s_idx-1] - 2.f*s_in[s_idx] + s_in[s_idx+1])/(h*h);
}

void ddParallel(float *out, const float *in, int n, float h) {
  float *d_in = 0, *d_out = 0;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

  // Set shared memory size in bytes
  const size_t smemSize = (TPB + 2 * RAD) * sizeof(float);

  ddKernel<<<(n + TPB - 1)/TPB, TPB, smemSize>>>(d_out, d_in, n, h);

  cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

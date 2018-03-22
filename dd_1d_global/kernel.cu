#include "kernel.h"
#define TPB 64

__global__
void ddKernel(float *d_out, const float *d_in, int size, float h) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i >= size - 1 || i == 0) return;
  d_out[i] = (d_in[i - 1] - 2.f*d_in[i] + d_in[i + 1]) / (h*h);
}

void ddParallel(float *out, const float *in, int n, float h) {
  float *d_in = 0, *d_out = 0;
  
  cudaMalloc(&d_in, n*sizeof(float));
  cudaMalloc(&d_out, n*sizeof(float));
  cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);
  
  ddKernel<<<(n + TPB - 1)/TPB, TPB>>>(d_out, d_in, n, h);
  
  cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

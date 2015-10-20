#include "kernel.h"
#include <stdio.h>
#define TPB 32

__device__
float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

__global__
void distanceKernel(float *d_out, float *d_in, float ref)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const float x = d_in[i];
  d_out[i] = distance(x, ref);
  printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

void distanceArray(float *out, float *in, float ref, int len)
{
  // Declare pointers to device arrays
  float *d_in = 0;
  float *d_out = 0;
 
  // Allocate memory for device arrays
  cudaMalloc(&d_in, len*sizeof(float));
  cudaMalloc(&d_out, len*sizeof(float));
 
  // Copy input data from host to device
  cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel to compute and store distance values
  distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);
 
  // Copy results from device to host
  cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);
 
  // Free the memory allocated for device arrays
  cudaFree(d_in);
  cudaFree(d_out);
}
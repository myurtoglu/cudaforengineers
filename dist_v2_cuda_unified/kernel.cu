#include <stdio.h>
#define N 64
#define TPB 32

float scale(int i, int n)
{
  return ((float)i)/(n - 1);
}

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

int main()
{
  const float ref = 0.5f;
  // Declare pointers for input and output arrays
  float *in = 0;
  float *out = 0;
  
  // Allocate managed memory for input and output arrays
  cudaMallocManaged(&in, N*sizeof(float));
  cudaMallocManaged(&out, N*sizeof(float));
  
  // Compute scaled input values
  for (int i = 0; i < N; ++i)
  {
    in[i] = scale(i, N);
  }
  
  // Launch kernel to compute and store distance values
  distanceKernel<<<N/TPB, TPB>>>(out, in, ref);
  cudaDeviceSynchronize();
  
  // Free the allocated memory
  cudaFree(in);
  cudaFree(out);
  
  return 0;
}
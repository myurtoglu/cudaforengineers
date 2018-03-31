#define W 500
#define H 500
#define TX 32 // number of threads per block along x-axis
#define TY 32 // number of threads per block along y-axis

__global__ void distanceKernel(float *d_out, int w, int h, float2 pos) 
{
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int i=r*w+c;

  if ((c >= w) || (r >= h)) return; 

  // Compute the distance and set d_out[i]
  d_out[i] = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));
}

int main() 
{
  float *out = (float*)calloc(W*H, sizeof(float));
  float *d_out; // pointer for device array

  cudaMalloc(&d_out,W*H*sizeof(float));

  const float2 pos = {0.0f, 0.0f}; // set reference position
  const dim3 blockSize(TX, TY);
  const int bx=(W+TX-1)/TX;
  const int by=(W+TY-1)/TY;
  const dim3 gridSize = dim3(bx,by);

  distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, pos);

  // Copy results to host.
  cudaMemcpy(out, d_out, W*H*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  free(out);
  return 0;
}
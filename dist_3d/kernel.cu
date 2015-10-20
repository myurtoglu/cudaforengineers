#define W 32
#define H 32
#define D 32
#define TX 8 // number of threads per block along x-axis
#define TY 8 // number of threads per block along y-axis
#define TZ 8 // number of threads per block along z-axis

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
float distance(int c, int r, int s, float3 pos) {
  return sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y) +
               (s - pos.z)*(s - pos.z));
}

__global__
void distanceKernel(float *d_out, int w, int h, int d, float3 pos) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // column
  const int r = blockIdx.y * blockDim.y + threadIdx.y; // row
  const int s = blockIdx.z * blockDim.z + threadIdx.z; // stack
  const int i = c + r*w + s*w*h;
  if ((c >= w) || (r >= h) || (s >= d)) return;
  d_out[i] = distance(c, r, s, pos); // compute and store result
}

int main() {
  float *out = (float*)calloc(W*H*D, sizeof(float));
  float *d_out = 0;
  cudaMalloc(&d_out, W*H*D*sizeof(float));
  const float3 pos = { 0.0f, 0.0f, 0.0f }; // set reference position
  const dim3 blockSize(TX, TY, TZ);
  const dim3 gridSize(divUp(W, TX), divUp(H, TY), divUp(D, TZ));
  distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, D, pos);
  cudaMemcpy(out, d_out, W*H*D*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  free(out);
  return 0;
}
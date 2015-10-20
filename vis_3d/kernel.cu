#include "kernel.h"
#include "device_funcs.cuh"
#include <helper_math.h>
#define TX_2D 32
#define TY_2D 32
#define TX 8
#define TY 8
#define TZ 8
#define NUMSTEPS 20

__global__
void renderKernel(uchar4 *d_out, float *d_vol, int w, int h,
  int3 volSize, int method, float zs, float theta, float threshold,
  float dist) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int i = c + r * w;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const uchar4 background = { 64, 0, 128, 0 };
  float3 source = { 0.f, 0.f, -zs };
  float3 pix = scrIdxToPos(c, r, w, h, 2 * volSize.z - zs);
  // apply viewing transformation: here rotate about y-axis
  source = yRotate(source, theta);
  pix = yRotate(pix, theta);
  // prepare inputs for ray-box intersection
  float t0, t1;
  const Ray pixRay = {source, pix - source};
  float3 center = {volSize.x/2.f, volSize.y/2.f, volSize.z/2.f};
  const float3 boxmin = -center;
  const float3 boxmax = {volSize.x - center.x, volSize.y - center.y,
                         volSize.z - center.z};
  // perform ray-box intersection test
  const bool hitBox = intersectBox(pixRay, boxmin, boxmax, &t0, &t1);
  uchar4 shade;
  if (!hitBox) shade = background; //miss box => background color
  else {
    if (t0 < 0.0f) t0 = 0.f; // clamp to 0 to avoid looking backward
    // bounded by points where the ray enters and leaves the box
    const Ray boxRay = { paramRay(pixRay, t0),
    paramRay(pixRay, t1) - paramRay(pixRay, t0) };
    if (method == 1) shade = 
      sliceShader(d_vol, volSize, boxRay, threshold, dist, source);
    else if (method == 2) shade =
      rayCastShader(d_vol, volSize, boxRay, threshold);
    else shade =
      volumeRenderShader(d_vol, volSize, boxRay, threshold, NUMSTEPS);
  }
  d_out[i] = shade;
}

__global__
void volumeKernel(float *d_vol, int3 volSize, int id, float4 params) {
  const int w = volSize.x, h = volSize.y, d = volSize.z;
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // column
  const int r = blockIdx.y * blockDim.y + threadIdx.y; // row
  const int s = blockIdx.z * blockDim.z + threadIdx.z; // stack
  const int i = c + r * w + s * w * h;
  if ((c >= w) || (r >= h) || (s >= d)) return;
  d_vol[i] = func(c, r, s, id, volSize, params); // compute and store
}

void kernelLauncher(uchar4 *d_out, float *d_vol, int w, int h,
  int3 volSize, int method, int zs, float theta, float threshold,
  float dist) {
  dim3 blockSize(TX_2D, TY_2D);
  dim3 gridSize(divUp(w, TX_2D), divUp(h, TY_2D));
  renderKernel<<<gridSize, blockSize>>>(d_out, d_vol, w, h, volSize,
    method, zs, theta, threshold, dist);
}

void volumeKernelLauncher(float *d_vol, int3 volSize, int id,
  float4 params) {
  dim3 blockSize(TX, TY, TZ);
  dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY),
  divUp(volSize.z, TZ));
  volumeKernel<<<gridSize, blockSize>>>(d_vol, volSize, id, params); 
}
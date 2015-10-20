#ifndef DEVICEFUNCS_CUH
#define DEVICEFUNCS_CUH

typedef struct {
float3 o, d; // origin and direction
} Ray;

__host__ int divUp(int a, int b);
__device__ float3 yRotate(float3 pos, float theta);
__device__ float func(int c, int r, int s, int id, int3 volSize,
  float4 params);
__device__ float3 paramRay(Ray r, float t);
__device__ float3 scrIdxToPos(int c, int r, int w, int h, float zs);
__device__ bool intersectBox(Ray r, float3 boxmin, float3 boxmax,
  float *tnear, float *tfar);
__device__ uchar4 sliceShader(float *d_vol, int3 volSize, Ray boxRay,
  float threshold, float dist, float3 norm);
__device__ uchar4 volumeRenderShader(float *d_vol, int3 volSize,
  Ray boxRay, float dist, int numSteps);
__device__ uchar4 rayCastShader(float *d_vol, int3 volSize,
  Ray boxRay, float dist);

#endif
#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int3;
struct float4;

void kernelLauncher(uchar4 *d_out, float *d_vol, int w, int h,
  int3 volSize, int method, int zs, float theta, float threshold,
  float dist);
 void volumeKernelLauncher(float *d_vol, int3 volSize, int id,
                           float4 params);
#endif
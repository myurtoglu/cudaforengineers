#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
// struct BC that contains all the boundary conditions
typedef struct {
  int x, y; // x and y location of pipe center
  float rad; // radius of pipe
  int chamfer; // chamfer
  float t_s, t_a, t_g; // temperatures in pipe, air, ground
} BC;

void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h,
                    BC bc);
void resetTemperature(float *d_temp, int w, int h, BC bc);

#endif
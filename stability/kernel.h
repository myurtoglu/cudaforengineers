#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s);

#endif
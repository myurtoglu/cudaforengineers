#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos);

#endif
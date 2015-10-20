#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;

void sharpenParallel(uchar4 *arr, int w, int h);
#endif
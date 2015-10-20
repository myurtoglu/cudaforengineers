#include "kernel.h"
#include <stdio.h>
#include <helper_math.h>
#define TPB 512

__global__
void centroidKernel(const uchar4 *d_img, int *d_centroidCol,
                    int *d_centroidRow, int *d_pixelCount,
                    int width, int height) {
  __shared__ uint4 s_img[TPB];
  
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int s_idx = threadIdx.x;
  const int row = idx / width;
  const int col = idx - row*width;
  
  if ((d_img[idx].x < 255 || d_img[idx].y < 255 ||
       d_img[idx].z < 255) && (idx < width*height)) {
    s_img[s_idx].x = col;
    s_img[s_idx].y = row;
    s_img[s_idx].z = 1;
  }
  else {
    s_img[s_idx].x = 0;
    s_img[s_idx].y = 0;
    s_img[s_idx].z = 0;
  }
  __syncthreads();
  
  // for (int s = 1; s < blockDim.x; s *= 2) {
  // int index = 2*s*s_idx;
  // if (index < blockDim.x) {
  // s_img[index] += s_img[index+s];
  // }
  // __syncthreads();
  // }
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (s_idx < s) {
      s_img[s_idx] += s_img[s_idx + s];
    }
    __syncthreads();
  }
  
  if (s_idx == 0) {
    atomicAdd(d_centroidCol, s_img[0].x);
    atomicAdd(d_centroidRow, s_img[0].y);
    atomicAdd(d_pixelCount, s_img[0].z);
  }
}

void centroidParallel(uchar4 *img, int width, int height) {
  uchar4 *d_img = 0;
  int *d_centroidRow = 0, *d_centroidCol = 0, *d_pixelCount = 0;
  int centroidRow = 0, centroidCol = 0, pixelCount = 0;
  
  // Allocate memory for device array and copy from host
  cudaMalloc(&d_img, width*height*sizeof(uchar4));
  cudaMemcpy(d_img, img, width*height*sizeof(uchar4),
  cudaMemcpyHostToDevice);
  
  // Allocate and set memory for three integers on the device
  cudaMalloc(&d_centroidRow, sizeof(int));
  cudaMalloc(&d_centroidCol, sizeof(int));
  cudaMalloc(&d_pixelCount, sizeof(int));
  cudaMemset(d_centroidRow, 0, sizeof(int));
  cudaMemset(d_centroidCol, 0, sizeof(int));
  cudaMemset(d_pixelCount, 0, sizeof(int));
  
  centroidKernel << <(width*height + TPB - 1) / TPB, TPB >> >(d_img,
  d_centroidCol, d_centroidRow, d_pixelCount, width, height);
  
  // Copy results from device to host.
  cudaMemcpy(&centroidRow, d_centroidRow, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&centroidCol, d_centroidCol, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&pixelCount, d_pixelCount, sizeof(int),
             cudaMemcpyDeviceToHost);
  
  centroidCol /= pixelCount;
  centroidRow /= pixelCount;
  
  printf("Centroid: {col = %d, row = %d} based on %d pixels\n",
         centroidCol, centroidRow, pixelCount);
  
  // Mark the centroid with red lines
  for (int col = 0; col < width; ++col) {
    img[centroidRow*width + col].x = 255;
    img[centroidRow*width + col].y = 0;
    img[centroidRow*width + col].z = 0;
  }
  for (int row = 0; row < height; ++row) {
    img[row*width + centroidCol].x = 255;
    img[row*width + centroidCol].y = 0;
    img[row*width + centroidCol].z = 0;
  }
   
  // Free the memory allocated
  cudaFree(d_img);
  cudaFree(d_centroidRow);
  cudaFree(d_centroidCol);
  cudaFree(d_pixelCount);
}
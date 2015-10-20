#define cimg_display 0
#include "CImg.h"
#include <cuda_runtime.h>
#include <npp.h>
#include <stdlib.h>
#define kNumCh 3

void sharpenNPP(Npp8u *arr, int w, int h) {
  Npp8u *d_in = 0, *d_out = 0;
  Npp32f *d_filter = 0;
  const Npp32f filter[9] = {-1.0, -1.0, -1.0,
                            -1.0,  9.0, -1.0,
                            -1.0, -1.0, -1.0};
  
  cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
  cudaMalloc(&d_filter, 9 * sizeof(Npp32f));
  cudaMemcpy(d_in, arr, kNumCh*w*h*sizeof(Npp8u),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filter, 9 * sizeof(Npp32f),
             cudaMemcpyHostToDevice);
  const NppiSize oKernelSize = {3, 3};
  const NppiPoint oAnchor = {1, 1};
  const NppiSize oSrcSize = {w, h};
  const NppiPoint oSrcOffset = {0, 0};
  const NppiSize oSizeROI = {w, h};
  
  nppiFilterBorder32f_8u_C3R(d_in, kNumCh*w*sizeof(Npp8u), oSrcSize,
    oSrcOffset, d_out, kNumCh*w*sizeof(Npp8u), oSizeROI, d_filter,
    oKernelSize, oAnchor, NPP_BORDER_REPLICATE);
  
  cudaMemcpy(arr, d_out, kNumCh*w*h*sizeof(Npp8u),
  cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_filter);
}

int main() {
  cimg_library::CImg<unsigned char> image("butterfly.bmp");
  const int w = image.width();
  const int h = image.height();
  Npp8u *arr = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
  
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      for (int ch = 0; ch < kNumCh; ++ch) {
        arr[kNumCh*(r*w + c) + ch] = image(c, r, ch);
      }
    }
  }
  
  sharpenNPP(arr, w, h);
  
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      for (int ch = 0; ch < kNumCh; ++ch) {
        image(c, r, ch) = arr[kNumCh*(r*w + c) + ch];
      }
    }
  }
  image.save_bmp("out.bmp");
  free(arr);
  return 0;
}
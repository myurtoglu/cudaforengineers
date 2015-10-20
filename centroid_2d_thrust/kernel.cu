#include "kernel.h"
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <stdio.h>

struct PixelFunctor {
  PixelFunctor(int width) : mWidth(width) {}
  template <typename T>
  __host__ __device__ int3 operator()(const T &el) {
    const int idx = thrust::get<0>(el);
    const uchar4 pixel = thrust::get<1>(el);
    const int r = idx / mWidth;
    const int c = idx - r*mWidth;
    int pixVal = (pixel.x < 255 || pixel.y < 255 || pixel.z < 255);
    return make_int3(pixVal*c, pixVal*r, pixVal);
    }
  int mWidth;
};

void centroidParallel(uchar4 *img, int width, int height) {
  thrust::device_vector<uchar4> invec(img, img + width*height);
  thrust::counting_iterator<int> first(0), last(invec.size());
  int3 res = thrust::transform_reduce(thrust::make_zip_iterator(
    thrust::make_tuple(first, invec.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(last, invec.end())),
    PixelFunctor(width), make_int3(0, 0, 0), thrust::plus<int3>());

  int centroidCol = res.x / res.z;
  int centroidRow = res.y / res.z;
  printf("Centroid: {col = %d, row = %d} based on %d pixels\n",
         centroidCol, centroidRow, res.z);
  
  for (int c = 0; c < width; ++c) {
    img[centroidRow*width + c].x = 255;
    img[centroidRow*width + c].y = 0;
    img[centroidRow*width + c].z = 0;
  }
  
  for (int r = 0; r < height; ++r) {
    img[r*width + centroidCol].x = 255;
    img[r*width + centroidCol].y = 0;
    img[r*width + centroidCol].z = 0;
  }
}
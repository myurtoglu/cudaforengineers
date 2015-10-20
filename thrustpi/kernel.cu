#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/count.h>
#include <stdio.h>
#define N (1 << 20)

int main() {
  thrust::host_vector<float> hvec_x(N), hvec_y(N);
  thrust::generate(hvec_x.begin(), hvec_x.end(), rand);
  thrust::generate(hvec_y.begin(), hvec_y.end(), rand);
  thrust::device_vector<float> dvec_x = hvec_x;
  thrust::device_vector<float> dvec_y = hvec_y;
  int insideCount =
    thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(
      dvec_x.begin(), dvec_y.begin())), thrust::make_zip_iterator(
      thrust::make_tuple(dvec_x.end(), dvec_y.end())),
      []__device__(const thrust::tuple<float, float> &el) {
        return (pow(thrust::get<0>(el)/RAND_MAX, 2) +
                pow(thrust::get<1>(el)/RAND_MAX, 2)) < 1.f; });
  printf("pi = %f\n", insideCount*4.f/N);
  return 0;
}
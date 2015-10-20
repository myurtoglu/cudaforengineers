#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <math.h>
#include <stdio.h>
#define N (1 << 20)

int main() {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
  thrust::device_vector<float>dvec_x(N);
  thrust::device_vector<float>dvec_y(N);
  float *ptr_x = thrust::raw_pointer_cast(&dvec_x[0]);
  float *ptr_y = thrust::raw_pointer_cast(&dvec_y[0]);
  curandGenerateUniform(gen, ptr_x, N);
  curandGenerateUniform(gen, ptr_y, N);
  curandDestroyGenerator(gen);
  int insideCount =
    thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(
      dvec_x.begin(), dvec_y.begin())), thrust::make_zip_iterator(
      thrust::make_tuple(dvec_x.end(), dvec_y.end())),
      []__device__(const thrust::tuple<float, float> &el) {
        return (pow(thrust::get<0>(el), 2) + 
                pow(thrust::get<1>(el), 2)) < 1.f; });
  printf("pi = %f\n", insideCount*4.f/N);
  return 0;
}
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <math.h>
#include <stdio.h>
#define N 64

int main() {
  const float ref = 0.5;
  thrust::device_vector<float> dvec_x(N);
  thrust::device_vector<float> dvec_dist(N);
  thrust::sequence(dvec_x.begin(), dvec_x.end());
  thrust::transform(dvec_x.begin(), dvec_x.end(), dvec_x.begin(),
    []__device__(float x){ return x / (N - 1); });
  thrust::transform(dvec_x.begin(), dvec_x.end(), dvec_dist.begin(),
    [=]__device__(float x){ return (x - ref)*(x - ref); });
  thrust::transform(dvec_dist.begin(), dvec_dist.end(),
    dvec_dist.begin(), []__device__(float x){ return sqrt(x); });
  thrust::host_vector<float> h_x = dvec_x;
  thrust::host_vector<float> h_dist = dvec_dist;
  for (int i = 0; i < N; ++i) {
    printf("x=%.3f, dist=%.3f\n", h_x[i], h_dist[i]);
  }
  return 0;
}
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <math.h>
#include <stdio.h>
#define N 64

// DistanceFrom(ref,n)(x)->sqrt((x/(n-1)-ref)*(x/(n-1)-ref))
struct DistanceFrom {
  DistanceFrom(float ref, int n) : mRef(ref), mN(n) {}
   
  __host__ __device__
  float operator()(const float &x) {
    float scaledX = x / (mN - 1);
    return std::sqrt((scaledX - mRef)*(scaledX - mRef));
  }
  float mRef;
  int mN;
};

int main() {
  const float ref = 0.5;
  thrust::device_vector<float> dvec_dist(N);
  thrust::transform(thrust::counting_iterator<float>(0),
    thrust::counting_iterator<float>(N), dvec_dist.begin(),
    DistanceFrom(ref, N));
  
  thrust::host_vector<float> hvec_dist = dvec_dist;
  float *ptr = thrust::raw_pointer_cast(&hvec_dist[0]); // debugging
  for (int i = 0; i < N; ++i) {
    printf("x[%d]=%.3f, dist=%.3f\n", i, 1.f*i / (N - 1), hvec_dist[i]);
  }
  return 0;
}
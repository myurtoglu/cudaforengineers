#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <math.h>
#include <stdio.h>
#define N 64

using namespace thrust::placeholders;

// Define transformation SqrtOf()(x) -> sqrt(x)
struct SqrtOf {
  __host__ __device__
  float operator()(float x) {
    return sqrt(x);
  }
};

int main() {
  const float ref = 0.5;
  thrust::device_vector<float> dvec_x(N);
  thrust::device_vector<float> dvec_dist(N);
  thrust::sequence(dvec_x.begin(), dvec_x.end());
  thrust::transform(dvec_x.begin(), dvec_x.end(),
                    dvec_x.begin(), _1/(N - 1));
  thrust::transform(dvec_x.begin(), dvec_x.end(),
                    dvec_dist.begin(), (_1 - ref)*(_1 - ref));
  thrust::transform(dvec_dist.begin(), dvec_dist.end(),
                    dvec_dist.begin(), SqrtOf());
  thrust::host_vector<float> h_x = dvec_x;
  thrust::host_vector<float> h_dist = dvec_dist;
  for (int i = 0; i < N; ++i) {
    printf("x=%3.3f, dist=%3.3f\n", h_x[i], h_dist[i]);
  }
  return 0;
}
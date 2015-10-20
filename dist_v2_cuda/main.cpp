#include "kernel.h"
#include <stdlib.h>
#define N 64

float scale(int i, int n)
{
  return ((float)i) / (n - 1);
}

int main()
{
  const float ref = 0.5f;

  float *in = (float*)calloc(N, sizeof(float));
  float *out = (float*)calloc(N, sizeof(float));

  // Compute scaled input values
  for (int i = 0; i < N; ++i)
  {
    in[i] = scale(i, N);
  }

  // Compute distances for the entire array
  distanceArray(out, in, ref, N);

  free(in);
  free(out);
  return 0;
}
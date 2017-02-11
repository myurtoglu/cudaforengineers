#include "aux_functions.h"
#define N 64 // Specify a constant value for array length.

int main()
{
  // Create 2 arrays of N floats (initialized to 0.0).
  // We will overwrite these values to store inputs and outputs.
  float in[N] = { 0.0f };
  float out[N] = { 0.0f };

  // Choose a reference value from which distances are measured.
  const float ref = 0.5f;

  // Iteration loop computes array of scaled input values.
  for (int i = 0; i < N; ++i)
  {
    in[i] = scale(i, N);
  }

  // Single function call to compute entire distance array.
  distanceArray(out, in, ref, N);

  return 0;
}
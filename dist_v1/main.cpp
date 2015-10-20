#include <math.h> //Include standard math library containing sqrt.
#define N 64 // Specify a constant value for array length.

// A scaling function to convert integers 0,1,...,N-1
// to evenly spaced floats ranging from 0 to 1.
float scale(int i, int n)
{
  return ((float)i) / (n - 1);
}

// Compute the distance between 2 points on a line.
float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

int main()
{
  // Create an array of N floats (initialized to 0.0).
  // We will overwrite these values to store our results.
  float out[N] = { 0.0f };

  // Choose a reference value from which distances are measured.
  const float ref = 0.5f;

  /* for loop to scale the index to obtain coordinate value,
   * compute the distance from the reference point,
   * and store the result in the corresponding entry in out. */
  for (int i = 0; i < N; ++i)
  {
    float x = scale(i, N);
    out[i] = distance(x, ref);
  }

  return 0;
}

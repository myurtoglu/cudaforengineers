#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#define N 1024

int main() {
  int cpu_res = 0;
  int gpu_res = 0;
  int *a = (int*)malloc(N*sizeof(int));
  int *b = (int*)malloc(N*sizeof(int));

  //Initialize input arrays
  for (int i = 0; i < N; ++i) {
    a[i] = 1;
    b[i] = 1;
  }

  for (int i = 0; i < N; ++i) {
    cpu_res += a[i] * b[i];
  }
  printf("cpu result = %d\n", cpu_res);

  dotLauncher(&gpu_res, a, b, N);
  printf("gpu result = %d\n", gpu_res);

  free(a);
  free(b);
  return 0;
}
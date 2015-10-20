#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

int main() {
  // Create A (m by n) and b (m by 1) on host and device.
  const int m = 9, n = 5;
  const int lda = m, ldb = m;
  
  float A[m*n] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    8.34, 23.64, 29.74, 19.07, 11.8, 13.97, 22.1, 14.47, 31.25,
    40.77, 58.49, 56.9, 49.69, 40.66, 39.16, 71.29, 41.76, 69.51,
    1010.84, 1011.4, 1007.15, 1007.22, 1017.13, 1016.05, 1008.2,
    1021.98, 1010.25,
    90.01, 74.2, 41.91, 76.79, 97.2, 84.6, 75.38, 78.41, 36.83};
  float b[m] = {
    480.48, 445.75, 438.76, 453.09, 464.43, 470.96, 442.35, 464, 428.77};

  float *d_A = 0, *d_b = 0;
  cudaMalloc(&d_A, m*n*sizeof(float));
  cudaMemcpy(d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_b, m*sizeof(float));
  cudaMemcpy(d_b, b, m*sizeof(float), cudaMemcpyHostToDevice);
  
  // Initialize the CUSOLVER and CUBLAS context.
  cusolverDnHandle_t cusolverDnH = 0;
  cublasHandle_t cublasH = 0;
  cusolverDnCreate(&cusolverDnH);
  cublasCreate(&cublasH);
  
  // Initialize solver parameters.
  float *tau = 0, *work = 0;
  int *devInfo = 0, Lwork = 0;
  cudaMalloc(&tau, MIN(m, n)*sizeof(float));
  cudaMalloc(&devInfo, sizeof(int));
  const float alpha = 1;
  
  // Calculate the size of work buffer needed.
  cusolverDnSgeqrf_bufferSize(cusolverDnH, m, n, d_A, lda, &Lwork);
  cudaMalloc(&work, Lwork*sizeof(float));
  
  // A = QR with CUSOLVER
  cusolverDnSgeqrf(cusolverDnH, m, n, d_A, lda, tau, work, Lwork,
                   devInfo);
  cudaDeviceSynchronize();
  
  // z = (Q^T)b with CUSOLVER, z is m x 1
  cusolverDnSormqr(cusolverDnH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 1,
                   MIN(m, n), d_A, lda, tau, d_b, ldb, work, Lwork,
                   devInfo);
  cudaDeviceSynchronize();
  
  // Solve Rx = z for x with CUBLAS, x is n x 1.
  cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
              CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1, &alpha, d_A,
              lda, d_b, ldb);
  // Copy the result and print.
  float x[n] = { 0.0 };
  cudaMemcpy(x, d_b, n*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) printf("x[%d] = %f\n", i, x[i]);
  
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverDnH);
  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(tau);
  cudaFree(devInfo);
  cudaFree(work);
  return 0;
}
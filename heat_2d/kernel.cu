#include "kernel.h"
#define TX 32
#define TY 32
#define RAD 1

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
  return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
  return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void resetKernel(float *d_temp, int w, int h, BC bc) {
  const int col = blockIdx.x*blockDim.x + threadIdx.x;
  const int row = blockIdx.y*blockDim.y + threadIdx.y;
  if ((col >= w) || (row >= h)) return;
  d_temp[row*w + col] = bc.t_a;
}

__global__
void tempKernel(uchar4 *d_out, float *d_temp, int w, int h, BC bc) {
  extern __shared__ float s_in[];
  // global indices
  const int col = threadIdx.x + blockDim.x * blockIdx.x;
  const int row = threadIdx.y + blockDim.y * blockIdx.y;
  if ((col >= w) || (row >= h)) return;
  const int idx = flatten(col, row, w, h);
  // local width and height
  const int s_w = blockDim.x + 2 * RAD;
  const int s_h = blockDim.y + 2 * RAD;
  // local indices
  const int s_col = threadIdx.x + RAD;
  const int s_row = threadIdx.y + RAD;
  const int s_idx = flatten(s_col, s_row, s_w, s_h);
  // assign default color values for d_out (black)
  d_out[idx].x = 0;
  d_out[idx].z = 0;
  d_out[idx].y = 0;
  d_out[idx].w = 255;
  
  // Load regular cells
  s_in[s_idx] = d_temp[idx];
  // Load halo cells
  if (threadIdx.x < RAD) {
    s_in[flatten(s_col - RAD, s_row, s_w, s_h)] =
      d_temp[flatten(col - RAD, row, w, h)];
    s_in[flatten(s_col + blockDim.x, s_row, s_w, s_h)] =
      d_temp[flatten(col + blockDim.x, row, w, h)];
  }
  if (threadIdx.y < RAD) {
    s_in[flatten(s_col, s_row - RAD, s_w, s_h)] =
      d_temp[flatten(col, row - RAD, w, h)];
    s_in[flatten(s_col, s_row + blockDim.y, s_w, s_h)] =
      d_temp[flatten(col, row + blockDim.y, w, h)];
  }

  // Calculate squared distance from pipe center
  float dSq = ((col - bc.x)*(col - bc.x) + (row - bc.y)*(row - bc.y));
  // If inside pipe, set temp to t_s and return
  if (dSq < bc.rad*bc.rad) {
    d_temp[idx] = bc.t_s;
    return;
  }
  // If outside plate, set temp to t_a and return
  if ((col == 0) || (col == w - 1) || (row == 0) ||
      (col + row < bc.chamfer) || (col - row > w - bc.chamfer)) {
    d_temp[idx] = bc.t_a;
    return;
  }
  // If point is below ground, set temp to t_g and return
  if (row == h - 1) {
    d_temp[idx] = bc.t_g;
    return;
  }
  __syncthreads();
  // For all the remaining points, find temperature and set colors.
  float temp = 0.25f*(s_in[flatten(s_col - 1, s_row, s_w, s_h)] +
  s_in[flatten(s_col + 1, s_row, s_w, s_h)] +
  s_in[flatten(s_col, s_row - 1, s_w, s_h)] +
  s_in[flatten(s_col, s_row + 1, s_w, s_h)]);
  d_temp[idx] = temp;
  const unsigned char intensity = clip((int)temp);
  d_out[idx].x = intensity; // higher temp -> more red
  d_out[idx].z = 255 - intensity; // lower temp -> more blue
}

void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h,
                    BC bc) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  const size_t smSz = (TX + 2 * RAD)*(TY + 2 * RAD)*sizeof(float);
  tempKernel<<<gridSize, blockSize, smSz>>>(d_out, d_temp, w, h, bc);
}

void resetTemperature(float *d_temp, int w, int h, BC bc) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  resetKernel<<<gridSize, blockSize>>>(d_temp, w, h, bc);
}
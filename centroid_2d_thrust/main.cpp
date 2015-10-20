#include "kernel.h"
#define cimg_display 0
#include "CImg.h"
#include <cuda_runtime.h>

int main()
{
  // Initializing input and output images
  cimg_library::CImg<unsigned char> inImage("wa_state.bmp");
  cimg_library::CImg<unsigned char> outImage(inImage, "xyzc", 0);
  int width = inImage.width();
  int height = inImage.height();

  // Initializing uchar array for image processing
  uchar4 *imgArray = (uchar4*)malloc(width*height*sizeof(uchar4));

  // Copying CImg data to image array
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      imgArray[row*width + col].x = inImage(col, row, 0);
      imgArray[row*width + col].y = inImage(col, row, 1);
      imgArray[row*width + col].z = inImage(col, row, 2);
    }
  }

  centroidParallel(imgArray, width, height);

  // Copying image array to CImg
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      outImage(col, row, 0) = imgArray[row*width + col].x;
      outImage(col, row, 1) = imgArray[row*width + col].y;
      outImage(col, row, 2) = imgArray[row*width + col].z;
    }
  }
  
  outImage.save("wa_state_out.bmp");
  free(imgArray);
}
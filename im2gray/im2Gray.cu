#include "im2Gray.h"
#include <math.h>

#define BLOCK 32



/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols)
{
	/*
		Your kernel here: Make sure to check for boundary conditions
	*/
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < numCols && row < numRows)
	{
		int idx = row * numCols + col;
		d_grey[idx] = 0.299 * d_in[idx].x + 0.587 * d_in[idx].y + 0.114 * d_in[idx].z;
	}

}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols)
{
    // configure launch params here 

    dim3 block(BLOCK,BLOCK,1);
    dim3 grid(ceil(numRows/(float)BLOCK),ceil(numCols/(float)BLOCK), 1);

    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

}






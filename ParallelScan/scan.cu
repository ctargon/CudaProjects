#include "im2Gray.h"
#include <math.h>

#define BLOCK 512



/*
 

 
 */
__global__ 
void scan(float *d_in, float *d_out, size_t length)
{
	/*
		Your kernel here: Make sure to check for boundary conditions
	*/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < length)
	{

	}
}




void launch_scan(float *d_in, float *d_out, size_t length)
{
		// configure launch params here 

		dim3 block(BLOCK, 1, 1);
		dim3 grid(ceil(length/(float)BLOCK), 1, 1);

		scan<<<grid,block>>>(d_in, d_out, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

}






#include "pde.h"
#include <math.h>



/*
 

 
 */
__global__ 
void pde(float *U, float *X, size_t length)
{
}


void launch_pde(float *d_in, float *d_out, float *d_sums, float *d_incs, size_t length)
{
		// configure launch params here 
		dim3 block(BLOCK, 1, 1);
		int grid_d = ceil(length / (2.0 * BLOCK));
		dim3 grid(grid_d, 1, 1);

		scan<<<grid,block>>>(d_in, d_out, d_sums, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		int grid_d2 = ceil(grid_d / (2.0 * BLOCK));
		dim3 grid2(grid_d2, 1, 1);
		scan<<<grid2,block>>>(d_sums, d_incs, NULL, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		increment_scan<<<grid,block>>>(d_in, d_out, d_incs, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
}






#include "pde.h"
#include <math.h>

#define BLOCK 32


/*
 

 
 */
__global__ 
void pde(float *U, float *U_out, size_t m, size_t n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float up, down, left, right;

	if (x < n && y < m)
	{
		up = down = left = right = 0;

		if (y - 1 >= 0) up = U[(y - 1) * n + x];
		if (y + 1 < m) down = U[(y + 1) * n + x];
		if (x - 1 >= 0) left = U[y * n + (x - 1)];
		if (x + 1 < n) right = U[y * n + (x + 1)];

		U_out[y * n + x] = (up + down + left + right) / 4;
	}
} 

#define TILE_WIDTH 32

__global__ 
void pde_shared(float *U, float *U_out, size_t m, size_t n)
{
	// __shared__ unsigned char window[TILE_WIDTH + fWidth - 1][TILE_WIDTH + fWidth - 1];	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float up, down, left, right;

	if (x < n && y < m)
	{
		up = down = left = right = 0;

		if (y - 1 >= 0) up = U[(y - 1) * n + x];
		if (y + 1 < m) down = U[(y + 1) * n + x];
		if (x - 1 >= 0) left = U[y * n + (x - 1)];
		if (x + 1 < n) right = U[y * n + (x + 1)];

		U_out[y * n + x] = (up + down + left + right) / 4;
	}
}


void launch_pde(float *U, float *U_out, size_t m, size_t n, size_t iters)
{
	int k = 0;
	float *tmp;
	// configure launch params here 
	dim3 block(BLOCK, BLOCK, 1);
	dim3 grid(ceil((float)m / BLOCK), ceil((float)n / BLOCK), 1);

	while (k < iters)
	{
		pde<<<grid,block>>>(U, U_out, m, n);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		k++;
		if (k < iters)
		{
			tmp = U_out;
			U_out = U;
			U = tmp;
		}
	}
}






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
        up = U[((y - 1 + m) % m)  * n + x];
        down = U[((y + 1) % m) * n + x];
        left = U[y * n + ((x - 1 + n) % n)];
        right = U[y * n + ((x + 1) % n)];
		U_out[y * n + x] = (up + down + left + right) / 4.0;
	}
} 


void pm(float *in, int m, int n)
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%.4f ", in[i * n + j]);
        }
        printf("\n");
    }

    printf("\n");
}

void launch_pde(float **U, float **U_out, size_t m, size_t n, size_t iters)
{
	int k = 0;
	float *tmp;
	// configure launch params here 
	dim3 block(BLOCK, BLOCK, 1);
	dim3 grid((n - 1) / BLOCK + 1, (m - 1) / BLOCK + 1, 1);

	while (k < iters)
	{
		pde<<<grid,block>>>(*U, *U_out, m, n);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		k++;
		if (k < iters)
		{
			tmp = *U_out;
			*U_out = *U;
			*U = tmp;
		}
	}
}






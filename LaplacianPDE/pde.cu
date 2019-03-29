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

	if (x > 0 && x < n - 1 && y < m)
	{
        up = U[(y - 1) * n + x];
        down = U[(y + 1) * n + x];
        left = U[y * n + ((x - 1 + n) % n)];
        right = U[y * n + ((x + 1) % n)];
		U_out[y * n + x] = (up + down + left + right) / 4.0;
	}
} 

__global__
void pde_multi(float *U, float *U_out, size_t m, size_t n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float up, down, left, right;

	if (x > 0 && x < n - 1 && y < m)
	{
        up = U[((y - 1 + m) % m) * n + x];
        down = U[((y + 1) % m) * n + x];
        left = U[y * n + ((x - 1 + n) % n)];
        right = U[y * n + ((x + 1) % n)];
		U_out[y * n + x] = (up + down + left + right) / 4.0;
	}
} 

void printf_matrix(float *in, int m, int n)
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

void launch_pde(float **hU, float **hU_out,float **U, float **U_out, size_t m, size_t n, size_t iters, int rank, int numprocs)
{
	int k = 0;
	float *tmp;
	// configure launch params here 
	dim3 block(BLOCK, BLOCK, 1);
	dim3 grid((n - 1) / BLOCK + 1, (m - 1) / BLOCK + 1, 1);

	int sub_rows = (m / numprocs) + 2;
	MPI_Status status;

	printf("rank: %d\n", rank);
	//printf_matrix(*U, sub_rows, n);
	printf("got here");
	while (k < iters)
	{
		printf("switching memory\n");

		MPI_Sendrecv(U[(sub_rows - 2) * n], n, MPI_FLOAT, (rank + 1) % numprocs, 0, U, n, MPI_FLOAT, (rank - 1 + numprocs) % numprocs, 0, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(U[n], n, MPI_FLOAT, (rank - 1 + numprocs) % numprocs, 0, U[(sub_rows - 1) * n], n, MPI_FLOAT, (rank + 1) % numprocs, 0, MPI_COMM_WORLD, &status);

		pde_multi<<<grid,block>>>(*U, *U_out, m, n);
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






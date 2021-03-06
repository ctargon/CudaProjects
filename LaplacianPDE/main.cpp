#include <iostream>
#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <string> 
#include <time.h>
#include "pde_single.h"


void checkResults(float *ref, float *gpu, size_t m, size_t n){
	for (int i = 0; i < (int)m; i++)
	{
		for(int j = 0; j < (int)n; j++)
		{
			if(fabs(ref[i * n + j] - gpu[i * n + j]) > 1e-3)
			{
				std::cerr << "Error at position " << i << "\n"; 
				std::cerr << "Reference:: " << std::setprecision(17) << + ref[i * n + j] <<"\n";
				std::cerr << "GPU:: " << +gpu[i * n + j] << "\n";
				exit(1);
			}
		}
	}
}


void print_matrix(float *in, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++)
		{
			printf("%f ", in[i * n + j]);
		}
		printf("\n");
	}

	printf("\n");
}


void serial_pde(float **U, float **U_out, int m, int n, int iters)
{
	int i, j, k = 0;

	float *tmp, up, down, left, right;

	while (k < iters)
	{
		for(i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
                up = (*U)[((i - 1 + m) % m) * n + j];
                down = (*U)[((i + 1) % m) * n + j];
                left = (*U)[i * n + ((j - 1 + n) % n)];
                right = (*U)[i * n + ((j + 1) % n)];
                (*U_out)[i * n + j] = (up + down + left + right) / 4.0;		
			}
		}
		tmp = *U;
		*U = *U_out;
		*U_out = tmp;
		k++;
	}
	tmp = *U;
	*U = *U_out;
	*U_out = tmp;
}

int main(int argc, char const *argv[])
{
	float *s_U, *s_U_out, *h_U, *d_U, *h_U_out, *d_U_out;
	int m = -1, n = -1, i, j, iters = 50;
	time_t t;

	srand((unsigned) time(&t));

	switch(argc)
	{
		case 3:
			m = atoi(argv[1]);
			n = atoi(argv[2]);
			break; 
		default: 
			std::cerr << "Usage ./pde <m> <n>\n";
			exit(1);
	}

	// set up array
	h_U = (float *) malloc (sizeof(float) * m * n);
	s_U = (float *) malloc (sizeof(float) * m * n);
	h_U_out = (float *) calloc (m * n, sizeof(float));
	s_U_out = (float *) calloc (m * n, sizeof(float));

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			float r = (float)(rand() / (float)RAND_MAX);	
			h_U[i * n + j] = r;
			s_U[i * n + j] = r;		
		}
	}

	float delta = 1.0;
	for (i = 0; i < m; i++)
	{
		float row_sum = 0;
		for (j = 0; j < n; j++)
		{
			row_sum += h_U[i * n + j];
		}
		if (i < n)
		{
			h_U[i * n + i] = row_sum + delta;
			s_U[i * n + i] = row_sum + delta;
		}
	}

	checkCudaErrors(cudaMalloc((void**)&d_U, sizeof(float) * m * n));
	checkCudaErrors(cudaMalloc((void**)&d_U_out, sizeof(float) * m * n));

	checkCudaErrors(cudaMemcpy(d_U, h_U, sizeof(float) * m * n, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_U_out, h_U_out, sizeof(float) * m * n, cudaMemcpyHostToDevice));

	// call the kernel 
	printf("launching kernel...\n");
	launch_pde(&d_U, &d_U_out, m, n, iters);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout << "Finished kernel launch \n";

	checkCudaErrors(cudaMemcpy(h_U_out, d_U_out, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
	struct timespec	tp1, tp2;

	clock_gettime(CLOCK_REALTIME, &tp1);
	serial_pde(&s_U, &s_U_out, m, n, iters);
	clock_gettime(CLOCK_REALTIME, &tp2);
	double d1 = tp1.tv_sec + tp1.tv_nsec / 1000000000.0;
	double d2 = tp2.tv_sec + tp2.tv_nsec / 1000000000.0;

	printf("Serial time (ms): %f\n", (d2 - d1) * 1000.0);
	// check if the caclulation was correct to a degree of tolerance
	checkResults(s_U_out, h_U_out, m, n);
	std::cout << "Results match.\n";

	cudaFree(d_U);
	cudaFree(d_U_out);
	free(h_U);
	free(h_U_out);
	free(s_U);
	free(s_U_out);

	return 0;
}




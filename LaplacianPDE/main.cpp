#include <iostream>
#include <cstdio>
#include <math.h>
#include <iomanip>
// #include <cuda.h>
// #include <cuda_runtime.h> 
#include <cassert>
#include <string> 
#include <time.h>
// #include "pde.h"


void checkResults(float *ref, float *gpu, size_t length){

	for(int i = 0; i < (int)length; i++){
		if(fabs(ref[i] - gpu[i]) > 1e-5){
			std::cerr << "Error at position " << i << "\n"; 

			std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
			std::cerr << "GPU:: " << +gpu[i] << "\n";

			exit(1);
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
			std::cout << in[i * n + j] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "\n";
}

void serial_pde(float *U, float *U_out, int m, int n, int iters)
{
	int i, j, k = 0;

	float up, down, left, right;

	while (k < iters)
	{
		for(i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				up = down = left = right = 0;
				if (i - 1 > 0)
				{
					if (k % 2) up = U_out[(i - 1) * n + j];
					else up = U[(i - 1) * n + j];
				}
				if (i + 1 < m)
				{
					if (k % 2) up = U_out[(i + 1) * n + j];
					else up = U[(i + 1) * n + j];				
				}
				if (j - 1 > 0)
				{
					if (k % 2) up = U_out[i * n + (j - 1)];
					else up = U[i * n + (j - 1)];				
				}
				if (j + 1 < n)
				{
					if (k % 2) up = U_out[i * n + (j + 1)];
					else up = U[i * n + (j + 1)];				
				}			

				if (k % 2) U_out[i * n + j] = (up + down + left + right) / 4;
				else U[i * n + j] = (up + down + left + right) / 4;
			}
		}
		k++;

		print_matrix(U, m, n);
		print_matrix(U_out, m, n);
	}

}



int main(int argc, char const *argv[])
{
	float *h_U, *d_U, *h_U_out, *d_U_out;
	int m = -1, n = -1, i, j, iters = 10;
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
	h_U_out = (float *) calloc (m * n, sizeof(float));


	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			h_U[i * n + j] = (float)(rand() % 1000) / 1000.0;			
		}
	}

	print_matrix(h_U, m, n);

	// checkCudaErrors(cudaMalloc((void**)&d_U, sizeof(float) * m * n));
	// checkCudaErrors(cudaMalloc((void**)&d_U_out, sizeof(float) * m * n));

	// checkCudaErrors(cudaMemcpy(d_U, h_U, sizeof(float) * length, cudaMemcpyHostToDevice)); 

	// // call the kernel 
	// launch_scan(d_U, d_U_out, m, n, iters);
	// cudaDeviceSynchronize();
	// checkCudaErrors(cudaGetLastError());

	// std::cout << "Finished kernel launch \n";

	// checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float) * length, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(sums, d_sums, sizeof(float) * num_sums, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(incs, d_incs, sizeof(float) * num_sums, cudaMemcpyDeviceToHost));

	struct timespec	tp1, tp2;
	clock_gettime(CLOCK_REALTIME, &tp1);
	serial_pde(h_U, h_U_out, m, n, iters);
	clock_gettime(CLOCK_REALTIME, &tp2);
	printf("Serial time (ns): %ld\n", tp2.tv_nsec-tp1.tv_nsec);

	print_matrix(h_U, m, n);
	print_matrix(h_U_out, m, n);

	// check if the caclulation was correct to a degree of tolerance
	// checkResults(s_out, h_out, length);
	// std::cout << "Results match.\n";

	// cudaFree(d_in);
	// cudaFree(d_out);
	free(h_U);
	free(h_U_out);

	return 0;
}




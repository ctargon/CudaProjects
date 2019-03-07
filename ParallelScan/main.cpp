#include <iostream>
#include <cstdio>
#include <math.h>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <string> 
#include <time.h>
#include "scan.h"


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



void serial_test(float *s_in, float *s_out, int length)
{
	int i;

	s_out[0] = s_in[0];

	for(i = 1; i < length; i++)
	{
		s_out[i] = s_out[i - 1] + s_in[i];
	}
}

void print_vector(float *in, int length)
{
	int i;

	for (i = 0; i < length; i++) std::cout << in[i] << " ";
	std::cout << "\n";
}



int main(int argc, char const *argv[])
{
	float *h_in, *d_in, *h_out, *d_out, *s_out, *sums, *d_sums, *incs, *d_incs;
	int length = -1, i;
	time_t t;

	srand((unsigned) time(&t));

	switch(argc)
	{
		case 2:
			length = atoi(argv[1]);
			break; 
		default: 
			std::cerr << "Usage ./scan <length>\n";
			exit(1);
	}

	// set up array
	h_in = (float *) malloc (sizeof(float) * length);
	h_out = (float *) malloc (sizeof(float) * length);
	s_out = (float *) malloc (sizeof(float) * length);
	int num_sums = ceil(length / (2.0 * BLOCK));
	sums = (float *) malloc (sizeof(float) * num_sums);
	incs = (float *) malloc (sizeof(float) * num_sums);

	for (i = 0; i < length; i++)
	{
		h_in[i] = (float)(rand() % 10);
	}

	checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(float) * length));
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float) * length));
	checkCudaErrors(cudaMalloc((void**)&d_sums, sizeof(float) * num_sums));
	checkCudaErrors(cudaMalloc((void**)&d_incs, sizeof(float) * num_sums));

	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(float) * length, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_sums, sums, sizeof(float) * num_sums, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_incs, incs, sizeof(float) * num_sums, cudaMemcpyHostToDevice));


	// call the kernel 
	launch_scan(d_in, d_out, d_sums, d_incs, length);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout << "Finished kernel launch \n";

	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float) * length, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(sums, d_sums, sizeof(float) * num_sums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(incs, d_incs, sizeof(float) * num_sums, cudaMemcpyDeviceToHost));

	struct timespec	tp1, tp2;
	clock_gettime(CLOCK_REALTIME, &tp1);
	serial_test(h_in, s_out, length);
	clock_gettime(CLOCK_REALTIME, &tp2);
	printf("Serial time (ns): %ld\n", tp2.tv_nsec-tp1.tv_nsec);

	//print_vector(h_in, length);
	//print_vector(s_out, length);
	//print_vector(h_out, length);
	//print_vector(sums, num_sums);
	//print_vector(incs, num_sums);

	// check if the caclulation was correct to a degree of tolerance
	checkResults(s_out, h_out, length);
	std::cout << "Results match.\n";

	cudaFree(d_in);
	cudaFree(d_out);
	free(h_in);
	free(h_out);
	free(s_out);

	return 0;
}




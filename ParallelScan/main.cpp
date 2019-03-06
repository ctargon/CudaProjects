#include <iostream>
#include <cstdio>
#include <math.h>
#include <iomanip>
// #include <cuda.h>
// #include <cuda_runtime.h> 
// #include <cassert>
// #include <string> 
// #include <opencv2/opencv.hpp> 
#include <time.h>
// #include "scan.h"



/*
	Driver program to test im2gray
*/


/*
	Process input image and allocate memory on host and 
	GPUs.
*/


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




int main(int argc, char const *argv[])
{
	float *h_in, *d_in, *h_out, *d_out;
	int length = -1, i;

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
	float vals[] = {3, 1, 7, 0, 4, 1, 6, 3};

	for (i = 0; i < length; i++)
	{
		h_in[i] = vals[i];
		std::cout << h_in[i] << " ";
	}
	


	// checkCudaErrors(cudaMalloc((void**)&d_imrgba, sizeof(uchar4)*numPixels));
	// checkCudaErrors(cudaMalloc((void**)&d_grey, sizeof(unsigned char)*numPixels));

	// checkCudaErrors(cudaMemcpy(d_imrgba, h_imrgba, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice)); 


	// // call the kernel 
	// launch_im2gray(d_imrgba, d_grey, img.rows, img.cols);
	// cudaDeviceSynchronize();
	// checkCudaErrors(cudaGetLastError());

	// std::cout << "Finished kernel launch \n";

	// checkCudaErrors(cudaMemcpy(h_grey, d_grey, numPixels*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	struct timespec	tp1, tp2;
	clock_gettime(CLOCK_REALTIME, &tp1);
	serial_test(h_in, h_out, length);
	clock_gettime(CLOCK_REALTIME, &tp2);
	printf("%ld\n", tp2.tv_nsec-tp1.tv_nsec);

	for (i = 0; i < length; i++)
	{
		std::cout << h_out[i] << " ";
	}

	// check if the caclulation was correct to a degree of tolerance
	//checkResult(reference, outfile, 1e-5);

	// cudaFree(d_imrgba);
	// cudaFree(d_grey);

	return 0;
}




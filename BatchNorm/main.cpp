#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <cmath> 
#include <iomanip>
#include <random>

#include "utils.h"
#include "batchnorm.h"

#define EPSILON 1e-3

void print_matrix(float *in, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++)
		{
			if (in[i * n + j]) printf("& ");
			else printf("  ");
		}
		printf("\n");
	}
	printf("\n");
}


void serial_batchnorm(float *X, float gamma, float beta, size_t batch_size, size_t rows, size_t cols, size_t depth)
{
	float *mean = (float *) calloc (rows * cols * depth, sizeof(float));
	float *var = (float *) malloc (sizeof(float) * rows * cols * depth);

	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t i = 0; i < rows * cols * depth; i++)
		{
			mean[i] += X[(b * rows * cols * depth) + i];
		}
	}

	for (size_t i = 0; i < rows * cols * depth; i++) mean[i] /= batch_size;

	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t i = 0; i < rows * cols * depth; i++)
		{
			var[i] += pow(X[(b * rows * cols * depth) + i] - mean[i], 2);
		}
	}

	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t i = 0; i < rows * cols * depth; i++)
		{
			X[(b * rows * cols * depth) + i] = (X[(b * rows * cols * depth) + i] - mean[i]) / sqrt(var[i] + EPSILON);
		}
	}

	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t i = 0; i < rows * cols * depth; i++)
		{
			X[(b * rows * cols * depth) + i] *= gamma;
			X[(b * rows * cols * depth) + i] += beta;
		}
	}
}


int main(int argc, char const *argv[])
{   
	size_t batch_size;
	std::string indir; 
	std::string outdir; 

	switch(argc)
	{
		case 4:
			batch_size = atoi(argv[1]);
			indir = std::string(argv[2]);
			outdir = std::string(argv[3]);
			break;
		default: 
			std::cerr << "Usage ./gblur batch_size <input_dir> <output_dir>\n";
			exit(1);
	}

	std::vector<cv::String> fn;
	std::vector<cv::Mat> images;
	cv::glob(indir, fn, false);

	// randomly shuffle the file names
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::shuffle(std::begin(fn), std::end(fn), std::default_random_engine(seed));

	// read in images from the data directory and save them to a vector in float form normalized between 0,1
	for (size_t i = 0; i < batch_size; i++)
	{
		cv::Mat img = cv::imread(fn[i], 0);
		cv::Mat f_img;
		if (img.empty())
		{
			std::cout << fn[i] << " is invalid!" << std::endl;
			continue;
		}
		img.convertTo(f_img, CV_32F, 1/255.0);
		images.push_back(f_img);
	}

	// get stats on the input batch
	batch_size = images.size();
	std::cout << "batch size: " << batch_size << std::endl;
	size_t rows = images[0].rows;
	std::cout << "rows: " << rows << std::endl;
	size_t cols = images[0].cols;
	std::cout << "cols: " << cols << std::endl;
	size_t channels = images[0].channels();
	std::cout << "channels: " << channels << std::endl;

	// allocate mem for the batch of data to be normalized
	float *batch_data = (float *) malloc (sizeof(float) * batch_size * rows * cols * channels);

	// memcpy the float data from the Mat object in the vector to the allocated array
	for (size_t i = 0; i < batch_size; i++)
	{
		size_t offset = i * rows * cols * channels;
		memcpy(&(batch_data[offset]), (float *)images[i].ptr<float>(0), sizeof(float) * rows * cols * channels);
	}

	// print images for debug
	//for (size_t i = 0; i < batch_size; i++)
	//{
	//	size_t offset = i * rows * cols * channels;
	//	print_matrix(&(batch_data[offset]), rows, cols);
	//}

	struct timespec	tp1, tp2;

	std::cout << "performing serial batchnorm..." << std::endl;

	clock_gettime(CLOCK_REALTIME, &tp1);
	serial_batchnorm(batch_data, 1.0, 0.0, batch_size, rows, cols, channels);
	clock_gettime(CLOCK_REALTIME, &tp2);

	double d1 = tp1.tv_sec + tp1.tv_nsec / 1000000000.0;
	double d2 = tp2.tv_sec + tp2.tv_nsec / 1000000000.0;

	printf("Serial time (ms): %f\n", (d2 - d1) * 1000.0);

	// write images
	for (size_t i = 0; i < batch_size; i++)
	{
		size_t offset = i * rows * cols * channels;
		cv::Mat o_img(rows, cols, CV_32F, &(batch_data[offset]));
		cv::Mat o_img_uint8;
		o_img.convertTo(o_img_uint8, CV_8U, 255);
		std::string fname = outdir + std::to_string(i) + ".png";
		cv::imwrite(fname, o_img_uint8);
	}

	


	return 0;
}




NVCC=nvcc 

OPENCV_LIBPATH=/usr/local/lib 

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O0 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o scan.o
	g++ -o scan main.o scan.o $(CUDA_LD_FLAGS)

main.o: main.cpp scan.h utils.h 
	g++ -c $(GCC_OPTS) main.cpp

scan.o: scan.cu scan.h utils.h
	$(NVCC) -c scan.cu $(NVCC_OPTS)

clean:
	rm *.o scan

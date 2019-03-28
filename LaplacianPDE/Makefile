NVCC=nvcc 

OPENCV_LIBPATH=/usr/local/lib 

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O0 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o pde.o
	g++ -o pde main.o pde.o $(CUDA_LD_FLAGS)

multi: main_mpi.o pde.o
	mpic++ -o pde_multi main_mpi.o pde.o $(CUDA_LD_FLAGS)

main_mpi.o: main_mpi.cpp pde.h utils.h
	mpic++ -c $(GCC_OPTS) main_mpi.cpp

main.o: main.cpp pde.h utils.h 
	g++ -c $(GCC_OPTS) main.cpp

pde.o: pde.cu pde.h utils.h
	$(NVCC) -c pde.cu $(NVCC_OPTS)

clean:
	rm *.o pde

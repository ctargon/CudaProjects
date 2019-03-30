NVCC=nvcc 

OPENCV_LIBPATH=/usr/local/lib 

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O0 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o pde_single.o
	g++ -o pde main.o pde_single.o $(CUDA_LD_FLAGS)

multi: multi.o
	mpic++ -o pde_multi multi.o $(CUDA_LD_FLAGS)

multi.o: multi.cu utils.h
	$(NVCC) -c multi.cu $(NVCC_OPTS)
#main_mpi.o: main_mpi.cpp pde.h utils.h
#	mpic++ -c $(GCC_OPTS) main_mpi.cpp

main.o: main.cpp pde_single.h utils.h 
	g++ -c $(GCC_OPTS) main.cpp

pde_single.o: pde_single.cu pde_single.h utils.h
	$(NVCC) -c pde_single.cu $(NVCC_OPTS)

#pde.o: pde.cu pde.h utils.h
#	$(NVCC) -c pde.cu $(NVCC_OPTS)

clean:
	rm *.o pde

#ifndef PDE_H_
#define PDE_H_

#include <mpi.h>
#include "utils.h"


void launch_pde(float **U, float **U_out, size_t m, size_t n, size_t iters, int rank, int numprocs);



#endif

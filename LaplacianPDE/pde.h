#ifndef PDE_H_
#define PDE_H_

#include "utils.h"

#define BLOCK 1024

void launch_pde(float *U, float *U_out, size_t m, size_t n, size_t iters);



#endif

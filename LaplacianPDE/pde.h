#ifndef PDE_H_
#define PDE_H_

#include "utils.h"

#define BLOCK 1024

void launch_pde(float *d_in, float *d_out, float *d_sums, float *d_incs, size_t length);



#endif

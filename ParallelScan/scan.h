#ifndef IM2_GRAY_H_
#define IM2_GRAY_H_

#include "utils.h"

#define BLOCK 1024

void launch_scan(float *d_in, float *d_out, float *d_sums, float *d_incs, size_t length);



#endif

#include <iostream>
#include "cuda_runtime.h"



__global__ void hello( void )
{
	printf( "Hello from thread %d\n", threadIdx.x );
}


int main( void )
{
	hello<<<1,10>>>();
	cudaDeviceSynchronize();

	return 0;
}

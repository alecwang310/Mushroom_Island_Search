#include <cuda_runtime.h>
extern "C" __global__ void ping(int *out) { if(threadIdx.x==0 && blockIdx.x==0) out[0]=42; }
extern "C" __declspec(dllexport) int run_ping(void *d_out) { ping<<<1,1>>>(d_out); return cudaDeviceSynchronize(); }

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern "C" { #include "../engine/continentalness.h"; }
#define MAX_OCTAVES 24
#define PERM_SIZE 256
#define THREADS 256
#define MAX_HITS_PER_SEED 512
extern "C" __global__ void tiered_scan(const uint8_t*,const float*,const float*,const float*,const float*,const float*,const uint8_t*,const float*,const float*,const int*,const float*,int,int,int,int,int*,int*,int*,unsigned long long*,unsigned long long*);
int main() {
    int n=8192; uint64_t *s=(uint64_t*)malloc(n*8);
    for(int i=0;i<n;i++)s[i]=12345+i;
    uint8_t *perm=(uint8_t*)malloc(n*MAX_OCTAVES*PERM_SIZE);
    float *oa=(float*)malloc(n*MAX_OCTAVES*4),*ob=(float*)malloc(n*MAX_OCTAVES*4),*oc=(float*)malloc(n*MAX_OCTAVES*4),*amp=(float*)malloc(n*MAX_OCTAVES*4),*lac=(float*)malloc(n*MAX_OCTAVES*4),*d2=(float*)malloc(n*MAX_OCTAVES*4),*t2=(float*)malloc(n*MAX_OCTAVES*4);
    uint8_t *h2=(uint8_t*)malloc(n*MAX_OCTAVES);
    int *rng=(int*)malloc(n*8*4); float *dbl=(float*)malloc(n*2*4);
    cont_batch_init(s,n,0,perm,oa,ob,oc,amp,lac,h2,d2,t2,rng,dbl);
    uint8_t *dp; float *doa,*dob,*doc,*damp,*dlac,*dd2,*dt2; uint8_t *dh2; int *drng; float *ddbl;
    cudaMalloc(&dp,n*MAX_OCTAVES*PERM_SIZE); cudaMalloc(&doa,n*MAX_OCTAVES*4); cudaMalloc(&dob,n*MAX_OCTAVES*4); cudaMalloc(&doc,n*MAX_OCTAVES*4); cudaMalloc(&damp,n*MAX_OCTAVES*4); cudaMalloc(&dlac,n*MAX_OCTAVES*4); cudaMalloc(&dh2,n*MAX_OCTAVES); cudaMalloc(&dd2,n*MAX_OCTAVES*4); cudaMalloc(&dt2,n*MAX_OCTAVES*4); cudaMalloc(&drng,n*8*4); cudaMalloc(&ddbl,n*2*4);
    cudaMemcpy(dp,perm,n*MAX_OCTAVES*PERM_SIZE,cudaMemcpyHostToDevice); cudaMemcpy(doa,oa,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(dob,ob,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(doc,oc,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(damp,amp,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(dlac,lac,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(dh2,h2,n*MAX_OCTAVES,cudaMemcpyHostToDevice); cudaMemcpy(dd2,d2,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(dt2,t2,n*MAX_OCTAVES*4,cudaMemcpyHostToDevice); cudaMemcpy(drng,rng,n*8*4,cudaMemcpyHostToDevice); cudaMemcpy(ddbl,dbl,n*2*4,cudaMemcpyHostToDevice);
    int *dc,*dgx,*dgz; cudaMalloc(&dc,n*4); cudaMalloc(&dgx,n*MAX_HITS_PER_SEED*4); cudaMalloc(&dgz,n*MAX_HITS_PER_SEED*4);
    unsigned long long *dtp,*dtd; cudaMalloc(&dtp,n*8); cudaMalloc(&dtd,n*8);
    cudaMemset(dc,0,n*4);
    // Warmup
    tiered_scan<<<n,THREADS>>>(dp,doa,dob,doc,damp,dlac,dh2,dd2,dt2,drng,ddbl,n,512,280,1,dc,dgx,dgz,dtp,dtd);
    cudaDeviceSynchronize();
    cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    for(int k=0;k<10;k++){cudaMemset(dc,0,n*4);tiered_scan<<<n,THREADS>>>(dp,doa,dob,doc,damp,dlac,dh2,dd2,dt2,drng,ddbl,n,512,280,1,dc,dgx,dgz,dtp,dtd);}
    cudaEventRecord(stop,0);cudaEventSynchronize(stop);
    float ms;cudaEventElapsedTime(&ms,start,stop);
    printf(\"TIERED ONLY: 10x8192 in %.1fms = %.0f seeds/s\n\",ms,10*n/(ms/1000));
    return 0;
}

/*
 * prefilter_host.cu — host-side prefilter wrapper (separate DLL).
 * Compile: nvcc -O3 -arch=sm_120 -shared -o prefilter.dll
 *               prefilter_host.cu prefilter_kernel.cu -lcudart
 *
 * Exports: prefilter_gpu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" __global__ void prefilter_seeds(
    const uint64_t *seeds, int n,
    float *scores, int *pass_idx, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes);

extern "C" __declspec(dllexport) int prefilter_gpu(
    const uint64_t *h_seeds, int n, float lo, float hi, const char *outfile)
{
    static uint64_t *d_seeds = NULL;
    static float    *d_scr   = NULL;
    static int      *d_pidx  = NULL, *d_pcnt = NULL;
    static int pf_cap = 0;

    if (n > pf_cap) {
        if (pf_cap > 0) { cudaFree(d_seeds); cudaFree(d_scr);
                          cudaFree(d_pidx); cudaFree(d_pcnt); }
        cudaMalloc(&d_seeds, n*8); cudaMalloc(&d_scr, n*4);
        cudaMalloc(&d_pidx, n*4);  cudaMalloc(&d_pcnt, 4);
        pf_cap = n;
    }

    cudaMemcpy(d_seeds, h_seeds, n*8, cudaMemcpyHostToDevice);
    cudaMemset(d_pcnt, 0, 4);
    prefilter_seeds<<<(n+255)/256, 256>>>(d_seeds, n, d_scr, d_pidx, d_pcnt, lo, hi, 0);
    cudaDeviceSynchronize();

    int pass = 0;
    cudaMemcpy(&pass, d_pcnt, 4, cudaMemcpyDeviceToHost);
    if (pass == 0) return 0;

    int *h_pidx = (int*)malloc(pass*4);
    cudaMemcpy(h_pidx, d_pidx, pass*4, cudaMemcpyDeviceToHost);
    uint64_t *surv = (uint64_t*)malloc(pass*8);
    for (int i = 0; i < pass; i++) surv[i] = h_seeds[h_pidx[i]];

    FILE *f = fopen(outfile, "wb");
    if (!f) { free(h_pidx); free(surv); return -1; }
    fwrite(surv, 8, pass, f);
    fclose(f);
    free(h_pidx); free(surv);
    return pass;
}

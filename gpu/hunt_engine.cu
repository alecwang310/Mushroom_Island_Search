/*
 * hunt_engine.cu — C++ tight-loop GPU hunt engine.
 * Eliminates Python overhead: seed gen, init, upload, kernel, download all in C++.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll hunt_engine.cu
 *        ../engine/continentalness.c -I../engine -lcudart
 *
 * Exports: hunt_batch(seeds, n, step, K, G, NG, radius, hits_out)
 *   Returns hit count. Python only does flood fill on returned hits.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- Include the C engine ----
extern "C" {
#include "../engine/continentalness.h"
}

// ---- GPU kernel declarations (same as sparse_kernel.cu) ----
#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define THREADS 256
#define MAX_GRIDS 32

// Forward declaration of kernel from sparse_kernel.cu
extern "C" __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    const int *grid_offsets, int num_grids, int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz, int *hit_grid);

// ---- Pre-allocated persistent buffers (GPU + host) ----
struct Buffers {
    // GPU
    uint8_t *d_perm; float *d_oa, *d_ob, *d_oc, *d_amp, *d_lac;
    uint8_t *d_h2; float *d_d2, *d_t2; int *d_ranges; float *d_dbl;
    int *d_offsets, *d_hit_flags, *d_hit_x, *d_hit_z, *d_hit_grid;
    // Host
    uint8_t *h_perms; float *h_oa, *h_ob, *h_oc, *h_amp, *h_lac;
    uint8_t *h_h2; float *h_d2, *h_t2; int *h_ranges; float *h_dbl;
    int *h_offsets, *h_hit_flags, *h_hit_x, *h_hit_z;
    int cap, cap_offsets;
} g = {0};

static void ensure_bufs(int n, int NG) {
    if (g.cap >= n && g.cap_offsets >= n * NG) return;
    #define REALLOC(p, sz, T) { if(p) free(p); p = (T*)malloc(sz); }
    int tb = n * MAX_OCTAVES * PERM_SIZE, sn = n * MAX_OCTAVES;
    REALLOC(g.h_perms, tb, uint8_t); REALLOC(g.h_oa, sn*4, float); REALLOC(g.h_ob, sn*4, float);
    REALLOC(g.h_oc, sn*4, float); REALLOC(g.h_amp, sn*4, float); REALLOC(g.h_lac, sn*4, float);
    REALLOC(g.h_h2, n*MAX_OCTAVES, uint8_t); REALLOC(g.h_d2, sn*4, float); REALLOC(g.h_t2, sn*4, float);
    REALLOC(g.h_ranges, n*32, int); REALLOC(g.h_dbl, n*8, float);
    REALLOC(g.h_offsets, n*NG*8, int); g.cap_offsets = n * NG;
    REALLOC(g.h_hit_flags, n*4, int); REALLOC(g.h_hit_x, n*4, int); REALLOC(g.h_hit_z, n*4, int);
    #define CUDA_R(p, sz) { if(p) cudaFree(p); cudaMalloc(&p, sz); }
    CUDA_R(g.d_perm, tb); CUDA_R(g.d_oa, sn*4); CUDA_R(g.d_ob, sn*4);
    CUDA_R(g.d_oc, sn*4); CUDA_R(g.d_amp, sn*4); CUDA_R(g.d_lac, sn*4);
    CUDA_R(g.d_h2, n*MAX_OCTAVES); CUDA_R(g.d_d2, sn*4); CUDA_R(g.d_t2, sn*4);
    CUDA_R(g.d_ranges, n*32); CUDA_R(g.d_dbl, n*8);
    CUDA_R(g.d_offsets, n*NG*8); CUDA_R(g.d_hit_flags, n*4);
    CUDA_R(g.d_hit_x, n*4); CUDA_R(g.d_hit_z, n*4); CUDA_R(g.d_hit_grid, n*4);
    g.cap = n;
}

// ---- Deterministic offset lattice ----
static void make_offsets(int *offsets, int n_seeds, int NG, int step, int radius) {
    int per_side = (int)sqrt((double)NG);
    int spacing = (radius * 2) / per_side;
    for (int s = 0; s < n_seeds; s++) {
        int seed_off = (s * 1234567) & 0x7FFFFFFF;  // deterministic per-seed shift
        for (int i = 0; i < NG; i++) {
            int gx = (i % per_side) * spacing - radius + (seed_off % spacing);
            int gz = (i / per_side) * spacing - radius + ((seed_off >> 16) % spacing);
            gx = (gx / step) * step; gz = (gz / step) * step;
            int idx = (s * NG + i) * 2;
            offsets[idx] = gx; offsets[idx+1] = gz;
        }
    }
}

// ---- Main entry point ----
extern "C" __declspec(dllexport) int hunt_batch(
    uint64_t start_seed, int n, int step, int K, int G, int NG, int radius,
    int64_t *hit_results)
{
    ensure_bufs(n, NG);
    int total_bytes = n * MAX_OCTAVES * PERM_SIZE;

    // Fill sequential seeds
    uint64_t *seeds_arr = (uint64_t*)malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds_arr[i] = start_seed + i;

    // Init
    cont_batch_init(seeds_arr, n, 0, g.h_perms, g.h_oa, g.h_ob, g.h_oc,
                    g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2, g.h_ranges, g.h_dbl);

    // Deterministic offsets
    make_offsets(g.h_offsets, n, NG, step, radius);

    // Upload
    cudaMemcpy(g.d_perm, g.h_perms, total_bytes, cudaMemcpyHostToDevice);
    int sn = n * MAX_OCTAVES;
    cudaMemcpy(g.d_oa, g.h_oa, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ob, g.h_ob, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oc, g.h_oc, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_amp, g.h_amp, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_lac, g.h_lac, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_h2, g.h_h2, n*MAX_OCTAVES, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_d2, g.h_d2, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_t2, g.h_t2, sn*4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ranges, g.h_ranges, n*32, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_dbl, g.h_dbl, n*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_offsets, g.h_offsets, n*NG*8, cudaMemcpyHostToDevice);
    cudaMemset(g.d_hit_flags, 0, n*4);

    // Kernel
    sparse_scan<<<n, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        g.d_offsets, NG, G, step, K,
        g.d_hit_flags, g.d_hit_x, g.d_hit_z, g.d_hit_grid);
    cudaDeviceSynchronize();

    // Download
    cudaMemcpy(g.h_hit_flags, g.d_hit_flags, n*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.h_hit_x, g.d_hit_x, n*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.h_hit_z, g.d_hit_z, n*4, cudaMemcpyDeviceToHost);

    int hit_count = 0;
    for (int i = 0; i < n; i++) {
        if (g.h_hit_flags[i]) {
            hit_results[hit_count*3]   = (int64_t)seeds_arr[i];
            hit_results[hit_count*3+1] = g.h_hit_x[i];
            hit_results[hit_count*3+2] = g.h_hit_z[i];
            hit_count++;
        }
    }
    free(seeds_arr);
    return hit_count;
}

extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g.d_perm) {
        cudaFree(g.d_perm); cudaFree(g.d_oa); cudaFree(g.d_ob);
        cudaFree(g.d_oc); cudaFree(g.d_amp); cudaFree(g.d_lac);
        cudaFree(g.d_h2); cudaFree(g.d_d2); cudaFree(g.d_t2);
        cudaFree(g.d_ranges); cudaFree(g.d_dbl);
        cudaFree(g.d_offsets);
        cudaFree(g.d_hit_flags); cudaFree(g.d_hit_x);
        cudaFree(g.d_hit_z); cudaFree(g.d_hit_grid);
        g.cap = 0;
    }
    if (g.h_perms) { free(g.h_perms); free(g.h_oa); free(g.h_ob); free(g.h_oc);
        free(g.h_amp); free(g.h_lac); free(g.h_h2); free(g.h_d2); free(g.h_t2);
        free(g.h_ranges); free(g.h_dbl); free(g.h_offsets);
        free(g.h_hit_flags); free(g.h_hit_x); free(g.h_hit_z); }
}

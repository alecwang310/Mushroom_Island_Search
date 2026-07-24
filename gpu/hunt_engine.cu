/*
 * hunt_engine.cu — C++ GPU hunt engine. No Python in the hot path.
 *
 * For each seed: one G×G grid centered at origin, every cell evaluated
 * on GPU. If any K×K block is all mushroom (cont < -1.05), hits are
 * reported back to the caller.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll hunt_engine.cu
 *        sparse_kernel.cu ../engine/continentalness.c -I../engine -lcudart
 *
 * Exports: hunt_batch(seeds, n, step, K, G, hits_out)
 *          hunt_cleanup()
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- C engine ----
extern "C" {
#include "../engine/continentalness.h"
}

// ---- Kernel signature (from sparse_kernel.cu) ----
#define MAX_OCTAVES 24
#define PERM_SIZE   257
#define THREADS     256

extern "C" __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz);

// ---- Persistent buffers (GPU + host) ----
struct Buffers {
    // GPU device pointers
    uint8_t *d_perm;
    float   *d_oa, *d_ob, *d_oc, *d_amp, *d_lac;
    uint8_t *d_h2;
    float   *d_d2, *d_t2;
    int     *d_ranges;
    float   *d_dbl;
    int     *d_hit_flags, *d_hit_x, *d_hit_z;

    // Host staging buffers
    uint8_t *h_perms;
    float   *h_oa, *h_ob, *h_oc, *h_amp, *h_lac;
    uint8_t *h_h2;
    float   *h_d2, *h_t2;
    int     *h_ranges;
    float   *h_dbl;
    int     *h_hit_flags, *h_hit_x, *h_hit_z;

    int cap;  // current capacity (seeds)
} g = {0};

// ---- Timing accumulators (ms) ----
static double t_init_sum = 0, t_upload_sum = 0, t_kernel_sum = 0, t_down_sum = 0;
static int t_batches = 0;
static cudaEvent_t ev_k_start, ev_k_stop;
static int ev_created = 0;

// ---- Allocate/reallocate buffers for n seeds ----
static void ensure_bufs(int n) {
    if (g.cap >= n) return;

    // Free existing if any
    #define FREE(p) if (p) { free(p); p = NULL; }
    FREE(g.h_perms); FREE(g.h_oa); FREE(g.h_ob); FREE(g.h_oc);
    FREE(g.h_amp);  FREE(g.h_lac); FREE(g.h_h2);
    FREE(g.h_d2);   FREE(g.h_t2);
    FREE(g.h_ranges); FREE(g.h_dbl);
    FREE(g.h_hit_flags); FREE(g.h_hit_x); FREE(g.h_hit_z);

    #define CUDA_FREE(p) if (p) { cudaFree(p); p = NULL; }
    CUDA_FREE(g.d_perm);
    CUDA_FREE(g.d_oa); CUDA_FREE(g.d_ob); CUDA_FREE(g.d_oc);
    CUDA_FREE(g.d_amp); CUDA_FREE(g.d_lac);
    CUDA_FREE(g.d_h2); CUDA_FREE(g.d_d2); CUDA_FREE(g.d_t2);
    CUDA_FREE(g.d_ranges); CUDA_FREE(g.d_dbl);
    CUDA_FREE(g.d_hit_flags); CUDA_FREE(g.d_hit_x); CUDA_FREE(g.d_hit_z);

    // Allocate new host buffers
    int perm_bytes = n * MAX_OCTAVES * PERM_SIZE;   // n*24*257
    int vec_elems  = n * MAX_OCTAVES;                // n*24 floats
    g.h_perms     = (uint8_t*)malloc(perm_bytes);
    g.h_oa        = (float*)  malloc(vec_elems * sizeof(float));
    g.h_ob        = (float*)  malloc(vec_elems * sizeof(float));
    g.h_oc        = (float*)  malloc(vec_elems * sizeof(float));
    g.h_amp       = (float*)  malloc(vec_elems * sizeof(float));
    g.h_lac       = (float*)  malloc(vec_elems * sizeof(float));
    g.h_h2        = (uint8_t*)malloc(n * MAX_OCTAVES);
    g.h_d2        = (float*)  malloc(vec_elems * sizeof(float));
    g.h_t2        = (float*)  malloc(vec_elems * sizeof(float));
    g.h_ranges    = (int*)    malloc(n * 8 * sizeof(int));
    g.h_dbl       = (float*)  malloc(n * 2 * sizeof(float));
    g.h_hit_flags = (int*)    malloc(n * sizeof(int));
    g.h_hit_x     = (int*)    malloc(n * sizeof(int));
    g.h_hit_z     = (int*)    malloc(n * sizeof(int));

    // Allocate new GPU buffers
    cudaMalloc(&g.d_perm,     perm_bytes);
    cudaMalloc(&g.d_oa,       vec_elems * sizeof(float));
    cudaMalloc(&g.d_ob,       vec_elems * sizeof(float));
    cudaMalloc(&g.d_oc,       vec_elems * sizeof(float));
    cudaMalloc(&g.d_amp,      vec_elems * sizeof(float));
    cudaMalloc(&g.d_lac,      vec_elems * sizeof(float));
    cudaMalloc(&g.d_h2,       n * MAX_OCTAVES);
    cudaMalloc(&g.d_d2,       vec_elems * sizeof(float));
    cudaMalloc(&g.d_t2,       vec_elems * sizeof(float));
    cudaMalloc(&g.d_ranges,   n * 8 * sizeof(int));
    cudaMalloc(&g.d_dbl,      n * 2 * sizeof(float));
    cudaMalloc(&g.d_hit_flags, n * sizeof(int));
    cudaMalloc(&g.d_hit_x,    n * sizeof(int));
    cudaMalloc(&g.d_hit_z,    n * sizeof(int));

    g.cap = n;
}

// ---- Main entry point ----
extern "C" __declspec(dllexport) int hunt_batch(
    uint64_t start_seed, int n, int step, int K, int G,
    int64_t *hit_results)
{
    ensure_bufs(n);

    // Create CUDA events once
    if (!ev_created) {
        cudaEventCreate(&ev_k_start);
        cudaEventCreate(&ev_k_stop);
        ev_created = 1;
    }

    // Build seed array
    uint64_t *seeds_arr = (uint64_t*)malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds_arr[i] = start_seed + i;

    // ---- Phase 1: CPU init ----
    clock_t c0 = clock();
    cont_batch_init(seeds_arr, n, 0,
        g.h_perms, g.h_oa, g.h_ob, g.h_oc,
        g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2,
        g.h_ranges, g.h_dbl);
    clock_t c1 = clock();
    double init_ms = ((double)(c1 - c0) / CLOCKS_PER_SEC) * 1000.0;

    // ---- Phase 2: Upload to GPU ----
    int perm_bytes = n * MAX_OCTAVES * PERM_SIZE;
    int vec_elems  = n * MAX_OCTAVES;
    #define H2D(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice)
    clock_t cu0 = clock();
    H2D(g.d_perm,  g.h_perms,  perm_bytes);
    H2D(g.d_oa,    g.h_oa,     vec_elems * sizeof(float));
    H2D(g.d_ob,    g.h_ob,     vec_elems * sizeof(float));
    H2D(g.d_oc,    g.h_oc,     vec_elems * sizeof(float));
    H2D(g.d_amp,   g.h_amp,    vec_elems * sizeof(float));
    H2D(g.d_lac,   g.h_lac,    vec_elems * sizeof(float));
    H2D(g.d_h2,    g.h_h2,     n * MAX_OCTAVES);
    H2D(g.d_d2,    g.h_d2,     vec_elems * sizeof(float));
    H2D(g.d_t2,    g.h_t2,     vec_elems * sizeof(float));
    H2D(g.d_ranges, g.h_ranges, n * 8 * sizeof(int));
    H2D(g.d_dbl,   g.h_dbl,    n * 2 * sizeof(float));
    cudaDeviceSynchronize();
    clock_t cu1 = clock();
    double upload_ms = ((double)(cu1 - cu0) / CLOCKS_PER_SEC) * 1000.0;

    // Zero hit output buffers
    cudaMemset(g.d_hit_flags, 0, n * sizeof(int));
    cudaMemset(g.d_hit_x,     0, n * sizeof(int));
    cudaMemset(g.d_hit_z,     0, n * sizeof(int));

    // ---- Phase 3: Kernel ----
    cudaEventRecord(ev_k_start, 0);
    sparse_scan<<<n, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        G, step, K,
        g.d_hit_flags, g.d_hit_x, g.d_hit_z);
    cudaEventRecord(ev_k_stop, 0);
    cudaEventSynchronize(ev_k_stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, ev_k_start, ev_k_stop);

    // ---- Phase 4: Download results ----
    clock_t cd0 = clock();
    #define D2H(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)
    D2H(g.h_hit_flags, g.d_hit_flags, n * sizeof(int));
    D2H(g.h_hit_x,     g.d_hit_x,     n * sizeof(int));
    D2H(g.h_hit_z,     g.d_hit_z,     n * sizeof(int));
    clock_t cd1 = clock();
    double down_ms = ((double)(cd1 - cd0) / CLOCKS_PER_SEC) * 1000.0;

    // Accumulate timings
    t_init_sum   += init_ms;
    t_upload_sum += upload_ms;
    t_kernel_sum += kernel_ms;
    t_down_sum   += down_ms;
    t_batches++;

    // Pack hits into output array
    int hit_count = 0;
    for (int i = 0; i < n; i++) {
        if (g.h_hit_flags[i]) {
            hit_results[hit_count * 3]     = (int64_t)seeds_arr[i];
            hit_results[hit_count * 3 + 1] = (int64_t)g.h_hit_x[i];
            hit_results[hit_count * 3 + 2] = (int64_t)g.h_hit_z[i];
            hit_count++;
        }
    }

    free(seeds_arr);
    return hit_count;
}

// ---- Timing query ----
extern "C" __declspec(dllexport) void hunt_get_timings(float *out) {
    if (t_batches == 0) {
        out[0] = out[1] = out[2] = out[3] = out[4] = 0;
        return;
    }
    out[0] = (float)(t_init_sum   / t_batches);  // init_ms
    out[1] = (float)(t_upload_sum / t_batches);  // upload_ms
    out[2] = (float)(t_kernel_sum / t_batches);  // kernel_ms
    out[3] = (float)(t_down_sum   / t_batches);  // download_ms
    out[4] = (float)t_batches;
}

extern "C" __declspec(dllexport) void hunt_reset_timings() {
    t_init_sum = t_upload_sum = t_kernel_sum = t_down_sum = 0;
    t_batches = 0;
}

// ---- Cleanup ----
extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g.d_perm) {
        cudaFree(g.d_perm); cudaFree(g.d_oa); cudaFree(g.d_ob);
        cudaFree(g.d_oc); cudaFree(g.d_amp); cudaFree(g.d_lac);
        cudaFree(g.d_h2); cudaFree(g.d_d2); cudaFree(g.d_t2);
        cudaFree(g.d_ranges); cudaFree(g.d_dbl);
        cudaFree(g.d_hit_flags); cudaFree(g.d_hit_x); cudaFree(g.d_hit_z);
        g.cap = 0;
    }
    #define FREE_IF(p) if (p) { free(p); p = NULL; }
    FREE_IF(g.h_perms); FREE_IF(g.h_oa); FREE_IF(g.h_ob); FREE_IF(g.h_oc);
    FREE_IF(g.h_amp);  FREE_IF(g.h_lac); FREE_IF(g.h_h2);
    FREE_IF(g.h_d2);   FREE_IF(g.h_t2);
    FREE_IF(g.h_ranges); FREE_IF(g.h_dbl);
    FREE_IF(g.h_hit_flags); FREE_IF(g.h_hit_x); FREE_IF(g.h_hit_z);
}

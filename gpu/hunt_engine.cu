/*
 * hunt_engine.cu — GPU hunt engine DLL.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll hunt_engine.cu
 *        sparse_kernel.cu warp_shuffle.cu tiered_kernel.cu
 *        ../engine/continentalness.c -I../engine -lcudart
 *
 * Exports: hunt_batch (baseline K=2), hunt_batch_ws (warp-shuffle),
 *          hunt_batch_tiered (hex O6+O15 multi-hit), hunt_cleanup,
 *          hunt_get_timings, hunt_reset_timings.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "../engine/continentalness.h"
}

#define MAX_OCTAVES 24
#define PERM_SIZE   256     // 32 lanes × 8 bytes, 8-byte aligned
#define THREADS     256

// ---- Debug timing struct (must match sparse_kernel.cu) ----
typedef struct {
    unsigned long long t_perm_load;    // cycles: loading s_perm + __syncthreads
    unsigned long long t_perlin;       // cycles: all perlin calls
    unsigned long long t_kx_check;     // cycles: KxK detection
    unsigned long long t_sync;         // cycles: __syncthreads outside other phases
    int n_perlin_calls;                // perlin calls made by thread 0
    int n_octaves;                     // octaves processed
    int n_tiles;                       // tiles processed before hit/exit
    int n_perm_loads;                  // number of perm loads
} DebugTiming;

// ---- Kernel signatures (from sparse_kernel.cu and warp_shuffle.cu) ----
extern "C" __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz,
    DebugTiming *debug);

// Warp-shuffle variant (from warp_shuffle.cu)
extern "C" __global__ void sparse_scan_ws(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz);

// Tiered two-stage variant (from tiered_kernel.cu)
extern "C" __global__ void tiered_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step_2x, int K_coarse,
    int *hit_counts, int *hit_gx, int *hit_gz,
    unsigned long long *t_perlin, unsigned long long *t_detect);

// Pre-filter variant (from prefilter_kernel.cu)
extern "C" __global__ void prefilter_seeds(
    const uint64_t *seeds, int n,
    float *scores, int *pass_idx, int *pass_count,
    float threshold, int large_biomes);

// ---- Persistent buffers ----
struct Buffers {
    uint8_t *d_perm;
    float   *d_oa, *d_ob, *d_oc, *d_amp, *d_lac;
    uint8_t *d_h2;
    float   *d_d2, *d_t2;
    int     *d_ranges;
    float   *d_dbl;
    int     *d_hit_flags, *d_hit_x, *d_hit_z;
    DebugTiming *d_debug;
    // Tiered multi-hit buffers
    int     *d_tier_counts, *d_tier_gx, *d_tier_gz;

    uint8_t *h_perms;
    float   *h_oa, *h_ob, *h_oc, *h_amp, *h_lac;
    uint8_t *h_h2;
    float   *h_d2, *h_t2;
    int     *h_ranges;
    float   *h_dbl;
    int     *h_hit_flags, *h_hit_x, *h_hit_z;
    DebugTiming *h_debug;
    // Tiered host buffers
    int     *h_tier_counts, *h_tier_gx, *h_tier_gz;

    int cap;
} g = {0};

// ---- Timing accumulators (ms) ----
static double t_init_sum = 0, t_upload_sum = 0, t_kernel_sum = 0, t_down_sum = 0;
static int t_batches = 0;
static cudaEvent_t ev_k_start, ev_k_stop;
static int ev_created = 0;

// Separate timing accumulators for warp-shuffle variant
static double ws_init_sum = 0, ws_upload_sum = 0, ws_kernel_sum = 0, ws_down_sum = 0;
static int ws_batches = 0;

// ---- Debug timing accumulators (averaged across batches) ----
static double d_perm_load = 0, d_perlin = 0, d_kx_check = 0, d_sync = 0;
static double d_n_perlin_calls = 0, d_n_octaves = 0, d_n_tiles = 0, d_n_perm_loads = 0;

static void ensure_bufs(int n) {
    if (g.cap >= n) return;

    #define FREE(p) if (p) { free(p); p = NULL; }
    FREE(g.h_perms); FREE(g.h_oa); FREE(g.h_ob); FREE(g.h_oc);
    FREE(g.h_amp);  FREE(g.h_lac); FREE(g.h_h2);
    FREE(g.h_d2);   FREE(g.h_t2);
    FREE(g.h_ranges); FREE(g.h_dbl);
    FREE(g.h_hit_flags); FREE(g.h_hit_x); FREE(g.h_hit_z);
    FREE(g.h_debug);

    #define CUDA_FREE(p) if (p) { cudaFree(p); p = NULL; }
    CUDA_FREE(g.d_perm);
    CUDA_FREE(g.d_oa); CUDA_FREE(g.d_ob); CUDA_FREE(g.d_oc);
    CUDA_FREE(g.d_amp); CUDA_FREE(g.d_lac);
    CUDA_FREE(g.d_h2); CUDA_FREE(g.d_d2); CUDA_FREE(g.d_t2);
    CUDA_FREE(g.d_ranges); CUDA_FREE(g.d_dbl);
    CUDA_FREE(g.d_hit_flags); CUDA_FREE(g.d_hit_x); CUDA_FREE(g.d_hit_z);
    CUDA_FREE(g.d_debug);

    int perm_bytes = n * MAX_OCTAVES * PERM_SIZE;
    int vec_elems  = n * MAX_OCTAVES;
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
    g.h_debug     = (DebugTiming*)malloc(n * sizeof(DebugTiming));

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
    cudaMalloc(&g.d_debug,    n * sizeof(DebugTiming));

    g.cap = n;
}

extern "C" __declspec(dllexport) int hunt_batch(
    uint64_t start_seed, int n, int step, int K, int G,
    int64_t *hit_results)
{
    ensure_bufs(n);

    if (!ev_created) {
        cudaEventCreate(&ev_k_start);
        cudaEventCreate(&ev_k_stop);
        ev_created = 1;
    }

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

    // ---- Phase 2: Upload ----
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

    // Zero output buffers
    cudaMemset(g.d_hit_flags, 0, n * sizeof(int));
    cudaMemset(g.d_hit_x,     0, n * sizeof(int));
    cudaMemset(g.d_hit_z,     0, n * sizeof(int));
    cudaMemset(g.d_debug,     0, n * sizeof(DebugTiming));

    // ---- Phase 3: Kernel ----
    cudaEventRecord(ev_k_start, 0);
    sparse_scan<<<n, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        G, step, K,
        g.d_hit_flags, g.d_hit_x, g.d_hit_z,
        g.d_debug);
    cudaEventRecord(ev_k_stop, 0);
    cudaEventSynchronize(ev_k_stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, ev_k_start, ev_k_stop);

    // ---- Phase 4: Download ----
    clock_t cd0 = clock();
    #define D2H(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)
    D2H(g.h_hit_flags, g.d_hit_flags, n * sizeof(int));
    D2H(g.h_hit_x,     g.d_hit_x,     n * sizeof(int));
    D2H(g.h_hit_z,     g.d_hit_z,     n * sizeof(int));
    D2H(g.h_debug,     g.d_debug,     n * sizeof(DebugTiming));
    clock_t cd1 = clock();
    double down_ms = ((double)(cd1 - cd0) / CLOCKS_PER_SEC) * 1000.0;

    // Accumulate host-side timings
    t_init_sum   += init_ms;
    t_upload_sum += upload_ms;
    t_kernel_sum += kernel_ms;
    t_down_sum   += down_ms;
    t_batches++;

    // Accumulate kernel-phase timings (aggregate across all seeds in batch)
    double sum_pl = 0, sum_pe = 0, sum_kx = 0, sum_sy = 0;
    double sum_np = 0, sum_no = 0, sum_nt = 0, sum_npl = 0;
    for (int i = 0; i < n; i++) {
        sum_pl += (double)g.h_debug[i].t_perm_load;
        sum_pe += (double)g.h_debug[i].t_perlin;
        sum_kx += (double)g.h_debug[i].t_kx_check;
        sum_sy += (double)g.h_debug[i].t_sync;
        sum_np += g.h_debug[i].n_perlin_calls;
        sum_no += g.h_debug[i].n_octaves;
        sum_nt += g.h_debug[i].n_tiles;
        sum_npl += g.h_debug[i].n_perm_loads;
    }
    d_perm_load   += sum_pl;
    d_perlin      += sum_pe;
    d_kx_check    += sum_kx;
    d_sync        += sum_sy;
    d_n_perlin_calls += sum_np;
    d_n_octaves     += sum_no;
    d_n_tiles       += sum_nt;
    d_n_perm_loads  += sum_npl;

    // Pack hits
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

// ---- Warp-shuffle variant entry point ----
// Same logic as hunt_batch but calls sparse_scan_ws (no debug timing support).
extern "C" __declspec(dllexport) int hunt_batch_ws(
    uint64_t start_seed, int n, int step, int K, int G,
    int64_t *hit_results)
{
    ensure_bufs(n);

    if (!ev_created) {
        cudaEventCreate(&ev_k_start);
        cudaEventCreate(&ev_k_stop);
        ev_created = 1;
    }

    uint64_t *seeds_arr = (uint64_t*)malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds_arr[i] = start_seed + i;

    // CPU init
    clock_t c0 = clock();
    cont_batch_init(seeds_arr, n, 0,
        g.h_perms, g.h_oa, g.h_ob, g.h_oc,
        g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2,
        g.h_ranges, g.h_dbl);
    clock_t c1 = clock();
    double init_ms = ((double)(c1 - c0) / CLOCKS_PER_SEC) * 1000.0;

    // Upload
    int perm_bytes = n * MAX_OCTAVES * PERM_SIZE;
    int vec_elems  = n * MAX_OCTAVES;
    #define H2D_WS(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice)
    clock_t cu0 = clock();
    H2D_WS(g.d_perm,  g.h_perms,  perm_bytes);
    H2D_WS(g.d_oa,    g.h_oa,     vec_elems * sizeof(float));
    H2D_WS(g.d_ob,    g.h_ob,     vec_elems * sizeof(float));
    H2D_WS(g.d_oc,    g.h_oc,     vec_elems * sizeof(float));
    H2D_WS(g.d_amp,   g.h_amp,    vec_elems * sizeof(float));
    H2D_WS(g.d_lac,   g.h_lac,    vec_elems * sizeof(float));
    H2D_WS(g.d_h2,    g.h_h2,     n * MAX_OCTAVES);
    H2D_WS(g.d_d2,    g.h_d2,     vec_elems * sizeof(float));
    H2D_WS(g.d_t2,    g.h_t2,     vec_elems * sizeof(float));
    H2D_WS(g.d_ranges, g.h_ranges, n * 8 * sizeof(int));
    H2D_WS(g.d_dbl,   g.h_dbl,    n * 2 * sizeof(float));
    cudaDeviceSynchronize();
    clock_t cu1 = clock();
    double upload_ms = ((double)(cu1 - cu0) / CLOCKS_PER_SEC) * 1000.0;

    // Zero hit buffers
    cudaMemset(g.d_hit_flags, 0, n * sizeof(int));
    cudaMemset(g.d_hit_x,     0, n * sizeof(int));
    cudaMemset(g.d_hit_z,     0, n * sizeof(int));

    // Kernel — warp-shuffle variant
    cudaEventRecord(ev_k_start, 0);
    sparse_scan_ws<<<n, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        G, step, K,
        g.d_hit_flags, g.d_hit_x, g.d_hit_z);
    cudaEventRecord(ev_k_stop, 0);
    cudaEventSynchronize(ev_k_stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, ev_k_start, ev_k_stop);

    // Download
    clock_t cd0 = clock();
    #define D2H_WS(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)
    D2H_WS(g.h_hit_flags, g.d_hit_flags, n * sizeof(int));
    D2H_WS(g.h_hit_x,     g.d_hit_x,     n * sizeof(int));
    D2H_WS(g.h_hit_z,     g.d_hit_z,     n * sizeof(int));
    clock_t cd1 = clock();
    double down_ms = ((double)(cd1 - cd0) / CLOCKS_PER_SEC) * 1000.0;

    ws_init_sum   += init_ms;
    ws_upload_sum += upload_ms;
    ws_kernel_sum += kernel_ms;
    ws_down_sum   += down_ms;
    ws_batches++;

    // Pack hits
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

// ---- Hex grid O6+O15 tiered hunt (multi-hit output) ----
#define MAX_HITS_PER_SEED 512
extern "C" __declspec(dllexport) int hunt_batch_tiered(
    uint64_t start_seed, int n, int step_2x, int K_coarse, int G,
    int *hit_counts_out, int64_t *hit_results)
{
    ensure_bufs(n);

    // Allocate tiered GPU buffers once (lazily, via ensure_bufs side-effect)
    static int *d_tier_counts, *d_tier_gx, *d_tier_gz;
    static int *h_tier_counts, *h_tier_gx, *h_tier_gz;
    static int tier_cap = 0;
    if (n > tier_cap) {
        if (tier_cap > 0) {
            cudaFree(d_tier_counts); cudaFree(d_tier_gx); cudaFree(d_tier_gz);
            free(h_tier_counts); free(h_tier_gx); free(h_tier_gz);
        }
        cudaMalloc(&d_tier_counts, n * sizeof(int));
        cudaMalloc(&d_tier_gx, n * MAX_HITS_PER_SEED * sizeof(int));
        cudaMalloc(&d_tier_gz, n * MAX_HITS_PER_SEED * sizeof(int));
        h_tier_counts = (int*)malloc(n * sizeof(int));
        h_tier_gx = (int*)malloc(n * MAX_HITS_PER_SEED * sizeof(int));
        h_tier_gz = (int*)malloc(n * MAX_HITS_PER_SEED * sizeof(int));
        tier_cap = n;
    }

    uint64_t *seeds_arr = (uint64_t*)malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds_arr[i] = start_seed + i;

    cont_batch_init(seeds_arr, n, 0,
        g.h_perms, g.h_oa, g.h_ob, g.h_oc,
        g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2,
        g.h_ranges, g.h_dbl);

    // Kernel hardcodes O6+O15 — ranges not used by tiered_scan.

    int perm_bytes = n * MAX_OCTAVES * PERM_SIZE;
    int vec_elems  = n * MAX_OCTAVES;
    cudaMemcpy(g.d_perm,  g.h_perms,  perm_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oa,    g.h_oa,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ob,    g.h_ob,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oc,    g.h_oc,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_amp,   g.h_amp,    vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_lac,   g.h_lac,    vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_h2,    g.h_h2,     n * MAX_OCTAVES, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_d2,    g.h_d2,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_t2,    g.h_t2,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ranges, g.h_ranges, n * 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_dbl,   g.h_dbl,    n * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaMemset(d_tier_counts, 0, n * sizeof(int));

    // Timing buffers (allocated once)
    static unsigned long long *d_tperlin, *d_tdetect;
    static unsigned long long *h_tperlin, *h_tdetect;
    static int timer_cap = 0;
    if (n > timer_cap) {
        if (timer_cap > 0) { cudaFree(d_tperlin); cudaFree(d_tdetect); free(h_tperlin); free(h_tdetect); }
        cudaMalloc(&d_tperlin, n * sizeof(unsigned long long));
        cudaMalloc(&d_tdetect, n * sizeof(unsigned long long));
        h_tperlin = (unsigned long long*)malloc(n * sizeof(unsigned long long));
        h_tdetect = (unsigned long long*)malloc(n * sizeof(unsigned long long));
        timer_cap = n;
    }

    tiered_scan<<<n, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        G, step_2x, K_coarse,
        d_tier_counts, d_tier_gx, d_tier_gz,
        d_tperlin, d_tdetect);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tier_counts, d_tier_counts, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tier_gx, d_tier_gx, n * MAX_HITS_PER_SEED * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tier_gz, d_tier_gz, n * MAX_HITS_PER_SEED * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tperlin, d_tperlin, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tdetect, d_tdetect, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // GPU per-seed timing
    if (n > 0) {
        unsigned long long tpl = h_tperlin[0], tdt = h_tdetect[0];
        printf("GPU cycles (seed 0): perlin=%llu detect=%llu total=%llu\n", tpl, tdt, tpl+tdt);
    }
    int total = 0;
    for (int i = 0; i < n; i++) {
        int cnt = h_tier_counts[i];
        if (cnt > MAX_HITS_PER_SEED) cnt = MAX_HITS_PER_SEED;
        hit_counts_out[i] = cnt;
        int base = i * MAX_HITS_PER_SEED;
        for (int j = 0; j < cnt; j++) {
            hit_results[total * 3]     = (int64_t)seeds_arr[i];
            hit_results[total * 3 + 1] = (int64_t)h_tier_gx[base + j];
            hit_results[total * 3 + 2] = (int64_t)h_tier_gz[base + j];
            total++;
        }
    }

    free(seeds_arr);
    return total;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-filter + tiered hunt: filter seeds by O6+O15 variance before GPU scan
// ═══════════════════════════════════════════════════════════════════════════
extern "C" __declspec(dllexport) int hunt_batch_prefilter(
    uint64_t start_seed, int n, int step_2x, int G,
    float threshold, int large_biomes,
    int *hit_counts_out, int64_t *hit_results)
{
    // ── Pre-filter GPU buffers (static, lazy allocation) ──────────────
    static uint64_t *d_seeds = NULL;
    static float    *d_scores = NULL;
    static int      *d_pass_idx = NULL;
    static int      *d_pass_count = NULL;
    static uint64_t *h_seeds = NULL;
    static float    *h_scores = NULL;
    static int      *h_pass_idx = NULL;
    static int pf_cap = 0;

    if (n > pf_cap) {
        if (pf_cap > 0) {
            cudaFree(d_seeds); cudaFree(d_scores);
            cudaFree(d_pass_idx); cudaFree(d_pass_count);
            free(h_seeds); free(h_scores); free(h_pass_idx);
        }
        cudaMalloc(&d_seeds,      n * sizeof(uint64_t));
        cudaMalloc(&d_scores,     n * sizeof(float));
        cudaMalloc(&d_pass_idx,   n * sizeof(int));
        cudaMalloc(&d_pass_count, 1 * sizeof(int));
        h_seeds    = (uint64_t*)malloc(n * sizeof(uint64_t));
        h_scores   = (float*)   malloc(n * sizeof(float));
        h_pass_idx = (int*)     malloc(n * sizeof(int));
        pf_cap = n;
    }

    // ── Generate seed array ───────────────────────────────────────────
    for (int i = 0; i < n; i++)
        h_seeds[i] = start_seed + (uint64_t)i;

    // ── Upload seeds, launch prefilter kernel ────────────────────────
    cudaMemcpy(d_seeds, h_seeds, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_pass_count, 0, sizeof(int));

    int blocks = (n + 255) / 256;
    prefilter_seeds<<<blocks, 256>>>(
        d_seeds, n, d_scores, d_pass_idx, d_pass_count,
        threshold, large_biomes);
    cudaDeviceSynchronize();

    // ── Download results ──────────────────────────────────────────────
    int pass_count = 0;
    cudaMemcpy(&pass_count, d_pass_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (pass_count == 0) return 0;

    cudaMemcpy(h_pass_idx, d_pass_idx, pass_count * sizeof(int), cudaMemcpyDeviceToHost);

    // ── Compact: collect surviving seeds ──────────────────────────────
    uint64_t *surviving = (uint64_t*)malloc(pass_count * sizeof(uint64_t));
    for (int i = 0; i < pass_count; i++)
        surviving[i] = h_seeds[h_pass_idx[i]];

    // ── Run standard tiered pipeline on survivors only ────────────────
    // The tiered pipeline uses persistent buffers at size cap >= n.
    // We fill only the first pass_count entries.
    ensure_bufs(n);  // keep full-size buffers for tiered scan output

    cont_batch_init(surviving, pass_count, large_biomes,
        g.h_perms, g.h_oa, g.h_ob, g.h_oc,
        g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2,
        g.h_ranges, g.h_dbl);

    // Upload only pass_count entries
    int perm_bytes = pass_count * MAX_OCTAVES * PERM_SIZE;
    int vec_elems  = pass_count * MAX_OCTAVES;
    cudaMemcpy(g.d_perm,  g.h_perms,  perm_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oa,    g.h_oa,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ob,    g.h_ob,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oc,    g.h_oc,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_amp,   g.h_amp,    vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_lac,   g.h_lac,    vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_h2,    g.h_h2,     pass_count * MAX_OCTAVES, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_d2,    g.h_d2,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_t2,    g.h_t2,     vec_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ranges, g.h_ranges, pass_count * 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_dbl,   g.h_dbl,    pass_count * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Reuse tiered hit buffers (already sized for n, which is >= pass_count)
    static int *d_tier_counts, *d_tier_gx, *d_tier_gz;
    static int *h_tier_counts, *h_tier_gx, *h_tier_gz;
    static int tier_cap = 0;
    if (n > tier_cap) {
        if (tier_cap > 0) {
            cudaFree(d_tier_counts); cudaFree(d_tier_gx); cudaFree(d_tier_gz);
            free(h_tier_counts); free(h_tier_gx); free(h_tier_gz);
        }
        cudaMalloc(&d_tier_counts, n * sizeof(int));
        cudaMalloc(&d_tier_gx, n * MAX_HITS_PER_SEED * sizeof(int));
        cudaMalloc(&d_tier_gz, n * MAX_HITS_PER_SEED * sizeof(int));
        h_tier_counts = (int*)malloc(n * sizeof(int));
        h_tier_gx = (int*)malloc(n * MAX_HITS_PER_SEED * sizeof(int));
        h_tier_gz = (int*)malloc(n * MAX_HITS_PER_SEED * sizeof(int));
        tier_cap = n;
    }

    cudaMemset(d_tier_counts, 0, pass_count * sizeof(int));

    tiered_scan<<<pass_count, THREADS>>>(
        g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, pass_count,
        G, step_2x, 1,
        d_tier_counts, d_tier_gx, d_tier_gz,
        NULL, NULL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tier_counts, d_tier_counts, pass_count * sizeof(int), cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i = 0; i < pass_count; i++) {
        int cnt = h_tier_counts[i];
        if (cnt > MAX_HITS_PER_SEED) cnt = MAX_HITS_PER_SEED;
        hit_counts_out[i] = cnt;
        // NOTE: hit_gx/hit_gz use global buffer indexed by the SURVIVING seed index.
        // We read from the start of the tier buffer.
        int base = i * MAX_HITS_PER_SEED;
        for (int j = 0; j < cnt; j++) {
            // We need to download gx/gz too. For now, allocate temp arrays.
            // Actual reading done below...
            hit_results[total * 3]     = (int64_t)surviving[i];
            hit_results[total * 3 + 1] = 0;  // placeholder
            hit_results[total * 3 + 2] = 0;
            total++;
        }
    }

    // Download the actual hit coordinates
    if (total > 0) {
        cudaMemcpy(h_tier_gx, d_tier_gx, pass_count * MAX_HITS_PER_SEED * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tier_gz, d_tier_gz, pass_count * MAX_HITS_PER_SEED * sizeof(int), cudaMemcpyDeviceToHost);

        total = 0;
        for (int i = 0; i < pass_count; i++) {
            int cnt = hit_counts_out[i];
            if (cnt > MAX_HITS_PER_SEED) cnt = MAX_HITS_PER_SEED;
            int base = i * MAX_HITS_PER_SEED;
            for (int j = 0; j < cnt; j++) {
                hit_results[total * 3]     = (int64_t)surviving[i];
                hit_results[total * 3 + 1] = (int64_t)h_tier_gx[base + j];
                hit_results[total * 3 + 2] = (int64_t)h_tier_gz[base + j];
                total++;
            }
        }
    }

    free(surviving);
    return total;
}

// ---- Host-side timing queries ----
extern "C" __declspec(dllexport) void hunt_get_timings(float *out) {
    if (t_batches == 0) { out[0]=out[1]=out[2]=out[3]=out[4]=0; return; }
    out[0] = (float)(t_init_sum   / t_batches);
    out[1] = (float)(t_upload_sum / t_batches);
    out[2] = (float)(t_kernel_sum / t_batches);
    out[3] = (float)(t_down_sum   / t_batches);
    out[4] = (float)t_batches;
}

extern "C" __declspec(dllexport) void hunt_get_timings_ws(float *out) {
    if (ws_batches == 0) { out[0]=out[1]=out[2]=out[3]=out[4]=0; return; }
    out[0] = (float)(ws_init_sum   / ws_batches);
    out[1] = (float)(ws_upload_sum / ws_batches);
    out[2] = (float)(ws_kernel_sum / ws_batches);
    out[3] = (float)(ws_down_sum   / ws_batches);
    out[4] = (float)ws_batches;
}

extern "C" __declspec(dllexport) void hunt_reset_timings() {
    t_init_sum = t_upload_sum = t_kernel_sum = t_down_sum = 0;
    t_batches = 0;
    ws_init_sum = ws_upload_sum = ws_kernel_sum = ws_down_sum = 0;
    ws_batches = 0;
    d_perm_load = d_perlin = d_kx_check = d_sync = 0;
    d_n_perlin_calls = d_n_octaves = d_n_tiles = d_n_perm_loads = 0;
}

// ---- Kernel-phase debug timing query ----
// Returns 10 doubles: [perm_load, perlin, kx_check, sync,
//                       n_perlin_calls, n_octaves, n_tiles, n_perm_loads,
//                       cycles_per_perlin, cycles_per_perm_load]
extern "C" __declspec(dllexport) void hunt_get_debug_timings(double *out) {
    if (t_batches == 0) {
        for (int i = 0; i < 10; i++) out[i] = 0;
        return;
    }
    double nb = (double)t_batches;
    out[0] = d_perm_load   / nb;   // avg cycles/batch: perm load
    out[1] = d_perlin      / nb;   // avg cycles/batch: all perlin
    out[2] = d_kx_check    / nb;   // avg cycles/batch: KxK check
    out[3] = d_sync        / nb;   // avg cycles/batch: __syncthreads
    out[4] = d_n_perlin_calls / nb;  // perlin calls per batch
    out[5] = d_n_octaves     / nb;  // octaves per batch
    out[6] = d_n_tiles       / nb;  // tiles per batch
    out[7] = d_n_perm_loads  / nb;  // perm loads per batch
    // Per-call averages
    out[8] = d_n_perlin_calls > 0 ? d_perlin / d_n_perlin_calls : 0;
    out[9] = d_n_perm_loads  > 0 ? d_perm_load / d_n_perm_loads : 0;
}

// ---- Cleanup ----
extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g.d_perm) {
        cudaFree(g.d_perm); cudaFree(g.d_oa); cudaFree(g.d_ob);
        cudaFree(g.d_oc); cudaFree(g.d_amp); cudaFree(g.d_lac);
        cudaFree(g.d_h2); cudaFree(g.d_d2); cudaFree(g.d_t2);
        cudaFree(g.d_ranges); cudaFree(g.d_dbl);
        cudaFree(g.d_hit_flags); cudaFree(g.d_hit_x); cudaFree(g.d_hit_z);
        cudaFree(g.d_debug);
        g.cap = 0;
    }
    #define FREE_IF(p) if (p) { free(p); p = NULL; }
    FREE_IF(g.h_perms); FREE_IF(g.h_oa); FREE_IF(g.h_ob); FREE_IF(g.h_oc);
    FREE_IF(g.h_amp);  FREE_IF(g.h_lac); FREE_IF(g.h_h2);
    FREE_IF(g.h_d2);   FREE_IF(g.h_t2);
    FREE_IF(g.h_ranges); FREE_IF(g.h_dbl);
    FREE_IF(g.h_hit_flags); FREE_IF(g.h_hit_x); FREE_IF(g.h_hit_z);
    FREE_IF(g.h_debug);
}

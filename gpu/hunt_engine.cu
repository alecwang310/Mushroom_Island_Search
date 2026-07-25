/*
 * hunt_engine.cu — GPU hex grid tiered hunt + prefilter DLL.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll
 *        hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu
 *        ../engine/continentalness.c -I../engine -lcudart
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

#define MAX_OCTAVES      24
#define PERM_SIZE        256
#define THREADS          256
#define MAX_HITS_PER_SEED 512
#define TIER_CHUNK       8192

// ── Kernels ────────────────────────────────────────────────────────────

extern "C" __global__ void tiered_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step_2x, int K_coarse,
    int *hit_counts, int *hit_gx, int *hit_gz,
    unsigned long long *t_perlin, unsigned long long *t_detect);

extern "C" __global__ void prefilter_seeds(
    const uint64_t *seeds, int n,
    float *scores, int *pass_idx, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes);

// ── Persistent buffers (shared by tiered hunt) ─────────────────────────

struct Buffers {
    uint8_t *d_perm;   float *d_oa, *d_ob, *d_oc, *d_amp, *d_lac;
    uint8_t *d_h2;     float *d_d2, *d_t2;
    int *d_ranges;     float *d_dbl;
    uint8_t *h_perms;  float *h_oa, *h_ob, *h_oc, *h_amp, *h_lac;
    uint8_t *h_h2;     float *h_d2, *h_t2;
    int *h_ranges;     float *h_dbl;
    int cap;
} g = {0};

static void ensure_bufs(int n) {
    if (g.cap >= n) return;
    #define FREE(p) if(p){free(p);p=NULL;}
    FREE(g.h_perms);FREE(g.h_oa);FREE(g.h_ob);FREE(g.h_oc);FREE(g.h_amp);FREE(g.h_lac);
    FREE(g.h_h2);FREE(g.h_d2);FREE(g.h_t2);FREE(g.h_ranges);FREE(g.h_dbl);
    #define CUDA_FREE(p) if(p){cudaFree(p);p=NULL;}
    CUDA_FREE(g.d_perm);CUDA_FREE(g.d_oa);CUDA_FREE(g.d_ob);CUDA_FREE(g.d_oc);
    CUDA_FREE(g.d_amp);CUDA_FREE(g.d_lac);CUDA_FREE(g.d_h2);
    CUDA_FREE(g.d_d2);CUDA_FREE(g.d_t2);CUDA_FREE(g.d_ranges);CUDA_FREE(g.d_dbl);
    int pb = n * MAX_OCTAVES * PERM_SIZE, ve = n * MAX_OCTAVES;
    g.h_perms  = (uint8_t*)malloc(pb);
    g.h_oa     = (float*)  malloc(ve*4); g.h_ob    = (float*)malloc(ve*4);
    g.h_oc     = (float*)  malloc(ve*4); g.h_amp   = (float*)malloc(ve*4);
    g.h_lac    = (float*)  malloc(ve*4); g.h_h2    = (uint8_t*)malloc(n*MAX_OCTAVES);
    g.h_d2     = (float*)  malloc(ve*4); g.h_t2    = (float*)malloc(ve*4);
    g.h_ranges = (int*)    malloc(n*8*4);g.h_dbl   = (float*)malloc(n*2*4);
    cudaMalloc(&g.d_perm,   pb);        cudaMalloc(&g.d_oa,    ve*4);
    cudaMalloc(&g.d_ob,     ve*4);      cudaMalloc(&g.d_oc,    ve*4);
    cudaMalloc(&g.d_amp,    ve*4);      cudaMalloc(&g.d_lac,   ve*4);
    cudaMalloc(&g.d_h2,     n*MAX_OCTAVES);
    cudaMalloc(&g.d_d2,     ve*4);      cudaMalloc(&g.d_t2,    ve*4);
    cudaMalloc(&g.d_ranges, n*8*4);     cudaMalloc(&g.d_dbl,   n*2*4);
    g.cap = n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Prefilter: seeds in RAM → GPU filter → survivors to disk
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// Tiered scan — internal (shared by hunt_batch_tiered and hunt_batch_from_file)
// ═══════════════════════════════════════════════════════════════════════════

static int tiered_chunk(uint64_t *seeds, int n, int step_2x, int G,
                         int *hit_counts_out, int64_t *hit_results) {
    ensure_bufs(n);

    // Tiered hit buffers (local statics — exact match to original working code)
    static int *d_tcnt, *d_tgx, *d_tgz;
    static int *h_tcnt, *h_tgx, *h_tgz;
    static int tier_cap = 0;
    if (n > tier_cap) {
        if (tier_cap > 0) { cudaFree(d_tcnt);cudaFree(d_tgx);cudaFree(d_tgz);
                            free(h_tcnt);free(h_tgx);free(h_tgz); }
        cudaMalloc(&d_tcnt, n*4); cudaMalloc(&d_tgx, n*MAX_HITS_PER_SEED*4);
        cudaMalloc(&d_tgz, n*MAX_HITS_PER_SEED*4);
        h_tcnt = (int*)malloc(n*4);
        h_tgx  = (int*)malloc(n*MAX_HITS_PER_SEED*4);
        h_tgz  = (int*)malloc(n*MAX_HITS_PER_SEED*4);
        tier_cap = n;
    }

    cont_batch_init(seeds, n, 0, g.h_perms, g.h_oa, g.h_ob, g.h_oc,
                    g.h_amp, g.h_lac, g.h_h2, g.h_d2, g.h_t2,
                    g.h_ranges, g.h_dbl);

    int pb = n * MAX_OCTAVES * PERM_SIZE, ve = n * MAX_OCTAVES;
    cudaMemcpy(g.d_perm,  g.h_perms,  pb,  cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oa,    g.h_oa,     ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ob,    g.h_ob,     ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_oc,    g.h_oc,     ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_amp,   g.h_amp,    ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_lac,   g.h_lac,    ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_h2,    g.h_h2,     n*MAX_OCTAVES,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_d2,    g.h_d2,     ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_t2,    g.h_t2,     ve*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ranges,g.h_ranges, n*8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_dbl,   g.h_dbl,    n*2*4,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Timing buffers (kernel writes to these unconditionally — cannot be NULL)
    static unsigned long long *d_tperlin, *d_tdetect;
    static unsigned long long *h_tperlin, *h_tdetect;
    static int timer_cap = 0;
    if (n > timer_cap) {
        if (timer_cap > 0) { cudaFree(d_tperlin); cudaFree(d_tdetect);
                              free(h_tperlin); free(h_tdetect); }
        cudaMalloc(&d_tperlin, n*8); cudaMalloc(&d_tdetect, n*8);
        h_tperlin = (unsigned long long*)malloc(n*8);
        h_tdetect = (unsigned long long*)malloc(n*8);
        timer_cap = n;
    }

    cudaMemset(d_tcnt, 0, n*4);
    tiered_scan<<<n, THREADS>>>(g.d_perm, g.d_oa, g.d_ob, g.d_oc, g.d_amp, g.d_lac,
        g.d_h2, g.d_d2, g.d_t2, g.d_ranges, g.d_dbl, n,
        G, step_2x, 1, d_tcnt, d_tgx, d_tgz, d_tperlin, d_tdetect);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tcnt, d_tcnt, n*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tgx,  d_tgx,  n*MAX_HITS_PER_SEED*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tgz,  d_tgz,  n*MAX_HITS_PER_SEED*4, cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i = 0; i < n; i++) {
        int c = h_tcnt[i]; if (c > MAX_HITS_PER_SEED) c = MAX_HITS_PER_SEED;
        hit_counts_out[i] = c;
        int base = i * MAX_HITS_PER_SEED;
        for (int j = 0; j < c; j++) {
            hit_results[total*3]   = (int64_t)seeds[i];
            hit_results[total*3+1] = (int64_t)h_tgx[base+j];
            hit_results[total*3+2] = (int64_t)h_tgz[base+j];
            total++;
        }
    }
    return total;
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __declspec(dllexport) int hunt_batch_tiered(
    uint64_t start_seed, int n, int step_2x, int K_coarse, int G,
    int *hit_counts_out, int64_t *hit_results)
{
    uint64_t *seeds = (uint64_t*)malloc(n*8);
    for (int i = 0; i < n; i++) seeds[i] = start_seed + i;
    int total = tiered_chunk(seeds, n, step_2x, G, hit_counts_out, hit_results);
    free(seeds);
    return total;
}

// Tiered scan on seeds from host memory (≤8192 seeds). No disk I/O.
extern "C" __declspec(dllexport) int tiered_scan_mem(
    const uint64_t *h_seeds, int n, int step_2x, int G,
    int *hit_counts_out, int64_t *hit_results)
{
    if (n > TIER_CHUNK) return -1;
    uint64_t *seeds = (uint64_t*)malloc(n*8);
    for (int i = 0; i < n; i++) seeds[i] = h_seeds[i];
    int total = tiered_chunk(seeds, n, step_2x, G, hit_counts_out, hit_results);
    free(seeds);
    return total;
}

extern "C" __declspec(dllexport) int hunt_batch_from_file(
    const char *seed_file, int step_2x, int G, const char *hits_file)
{
    FILE *f = fopen(seed_file, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long fsz = ftell(f); rewind(f);
    int n = (int)(fsz/8);
    if (n <= 0 || fsz%8 != 0) { fclose(f); return -1; }

    uint64_t *seeds = (uint64_t*)malloc(n*8);
    if (fread(seeds, 8, n, f) != (size_t)n) { fclose(f); free(seeds); return -1; }
    fclose(f);

    FILE *out = fopen(hits_file, "wb");
    if (!out) { free(seeds); return -1; }

    int *chunk_cnt = (int*)malloc(TIER_CHUNK*4);
    int64_t *chunk_hits = (int64_t*)malloc(TIER_CHUNK*MAX_HITS_PER_SEED*3*8);
    int total = 0;

    for (int off = 0; off < n; off += TIER_CHUNK) {
        int sz = (n - off < TIER_CHUNK) ? (n - off) : TIER_CHUNK;
        int hc = tiered_chunk(seeds + off, sz, step_2x, G, chunk_cnt, chunk_hits);
        for (int i = 0; i < hc; i++) {
            int64_t hit[3] = {chunk_hits[i*3], chunk_hits[i*3+1], chunk_hits[i*3+2]};
            fwrite(hit, 8, 3, out);
        }
        total += hc;
    }

    free(chunk_cnt); free(chunk_hits); free(seeds); fclose(out);
    return total;
}

extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g.d_perm) {
        cudaFree(g.d_perm);cudaFree(g.d_oa);cudaFree(g.d_ob);cudaFree(g.d_oc);
        cudaFree(g.d_amp);cudaFree(g.d_lac);cudaFree(g.d_h2);
        cudaFree(g.d_d2);cudaFree(g.d_t2);cudaFree(g.d_ranges);cudaFree(g.d_dbl);
        g.cap=0;
    }
    #define F(p) if(p){free(p);p=NULL;}
    F(g.h_perms);F(g.h_oa);F(g.h_ob);F(g.h_oc);F(g.h_amp);F(g.h_lac);
    F(g.h_h2);F(g.h_d2);F(g.h_t2);F(g.h_ranges);F(g.h_dbl);
}

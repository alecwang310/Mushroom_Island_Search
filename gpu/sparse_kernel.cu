/*
 * sparse_kernel.cu — Tiled per-seed GPU kernel with phase-level profiling.
 *
 * One block per seed, 256 threads. G×G grid, K×K mushroom detection.
 * Shared-memory perm table (260 bytes/aligned), direct-index perlin.
 *
 * Profiling: clock64() measures 4 outer phases (fenced by __syncthreads):
 *   1. Perm load  — global→shared copy + __syncthreads
 *   2. Perlin     — all perlin calls per octave (hash chain + gradient + interp)
 *   3. K×K check  — scan s_grid for mushroom blocks
 *   4. Sync stall — __syncthreads outside the above
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_OCTAVES 24
#define PERM_SIZE   256     // 32 lanes × 8 bytes, 8-byte aligned
#define THREADS     256
#define TILE        32
#define MAXC        4

// Debug output struct — one per seed
typedef struct {
    unsigned long long t_perm_load;    // cycles: loading s_perm + __syncthreads
    unsigned long long t_perlin;       // cycles: all perlin calls across all octaves
    unsigned long long t_kx_check;     // cycles: K×K detection scan
    unsigned long long t_sync;         // cycles: __syncthreads outside other phases
    int n_perlin_calls;                // perlin calls made by thread 0
    int n_octaves;                     // octaves processed
    int n_tiles;                       // tiles processed before hit/exit
    int n_perm_loads;                  // number of perm loads into shared memory
} DebugTiming;

// ---- Gradient table ----
__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

// ---- Perlin noise (no inner timing — reliable outer timing via caller) ----
__device__ __forceinline__ float perlin(
    const uint8_t *perm, float oa, float ob, float oc,
    uint8_t cached_h2, float cached_d2, float cached_t2,
    float x, float y, float z)
{
    float d1 = x + oa, d2 = y + ob, d3 = z + oc;
    uint8_t h2; float t2;
    if (y == 0.0f) {
        h2 = cached_h2; d2 = cached_d2; t2 = cached_t2;
    } else {
        float i2 = floorf(d2); d2 -= i2;
        h2 = (uint8_t)((int)i2 & 0xFF);
        t2 = d2*d2*d2 * (d2*(d2*6.0f - 15.0f) + 10.0f);
    }
    float i1 = floorf(d1), i3 = floorf(d3); d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF, h3 = ((int)i3) & 0xFF;
    float t1 = d1*d1*d1 * (d1*(d1*6.0f - 15.0f) + 10.0f);
    float t3 = d3*d3*d3 * (d3*(d3*6.0f - 15.0f) + 10.0f);

    // Hash chain: 14 dependent s_perm[] lookups
    int va = perm[h1] + h2, vb = perm[h1+1] + h2;
    int v2a = perm[va & 0xFF] + h3;
    int v2b = perm[(va & 0xFF)+1] + h3;
    int v3a = perm[vb & 0xFF] + h3;
    int v3b = perm[(vb & 0xFF)+1] + h3;
    int v4a = perm[v2a & 0xFF];
    int v4b = perm[(v2a & 0xFF)+1];
    int v5a = perm[v2b & 0xFF];
    int v5b = perm[(v2b & 0xFF)+1];
    int v6a = perm[v3a & 0xFF];
    int v6b = perm[(v3a & 0xFF)+1];
    int v7a = perm[v3b & 0xFF];
    int v7b = perm[(v3b & 0xFF)+1];

    // Gradient LERP (8 gradient lookups × 3 MACs each = 24 FMAs)
    // + trilinear interpolation (7 lerp steps)
    #define L(i,a,b,c) (c_grad[(i)&0xF][0]*(a) + \
                         c_grad[(i)&0xF][1]*(b) + \
                         c_grad[(i)&0xF][2]*(c))
    float l1 = L(v4a, d1, d2, d3),   l5 = L(v4b, d1, d2, d3-1);
    float l2 = L(v6a, d1-1, d2, d3), l6 = L(v6b, d1-1, d2, d3-1);
    float l3 = L(v5a, d1, d2-1, d3), l7 = L(v5b, d1, d2-1, d3-1);
    float l4 = L(v7a, d1-1, d2-1, d3), l8 = L(v7b, d1-1, d2-1, d3-1);
    #undef L

    l1 += t1*(l2 - l1); l3 += t1*(l4 - l3);
    l5 += t1*(l6 - l5); l7 += t1*(l8 - l7);
    l1 += t2*(l3 - l1); l5 += t2*(l7 - l5);
    return l1 + t3*(l5 - l1);
}


// ============================================================================
// Main kernel
// ============================================================================
extern "C" __launch_bounds__(THREADS, 4) __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz,
    DebugTiming *debug)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f / 331.0f;

    // ---- Thread 0 timing accumulators ----
    unsigned long long t_perm_load = 0, t_perlin = 0;
    unsigned long long t_kx_check = 0, t_sync = 0;
    int n_perlin_calls = 0, n_octaves = 0, n_tiles = 0, n_perm_loads = 0;
    long long t0, t1;

    // ---- Octave params in shared memory ----
    __shared__ float s_oa[MAX_OCTAVES], s_ob[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    if (tid < MAX_OCTAVES) {
        int src = seed * MAX_OCTAVES + tid;
        s_oa[tid]  = oa[src];  s_ob[tid]  = ob[src];  s_oc[tid] = oc[src];
        s_amp[tid] = amp[src]; s_lac[tid] = lac[src];
        s_h2[tid]  = h2[src];  s_d2[tid]  = d2[src];  s_t2[tid] = t2[src];
    }
    __shared__ int s_ranges[8]; __shared__ float s_dbl[2];
    if (tid < 8) s_ranges[tid] = ranges[seed * 8 + tid];
    if (tid < 2) s_dbl[tid]   = dbl_amps[seed * 2 + tid];

    // ---- Shared memory for perm, tile grid, found flag ----
    __shared__ uint8_t s_perm[PERM_SIZE + 1];  // +1 for perm[256] = perm[0]
    __shared__ float s_grid[TILE][TILE];
    __shared__ int s_found;

    // Sync 1: octave params → all threads
    t0 = clock64();
    __syncthreads();
    t1 = clock64();
    if (tid == 0) t_sync += (unsigned long long)(t1 - t0);

    int sh_as = s_ranges[0], sh_ac = s_ranges[1];
    int sh_bs = s_ranges[2], sh_bc = s_ranges[3];
    int ct_as = s_ranges[4], ct_ac = s_ranges[5];
    int ct_bs = s_ranges[6], ct_bc = s_ranges[7];
    int total_oct = sh_ac + sh_bc + ct_ac + ct_bc;
    float sh_amp = s_dbl[0], ct_amp = s_dbl[1];
    float sh4 = sh_amp * 4.0f;

    int center = -(G / 2) * step;
    int tiles_dim = (G + TILE - 1) / TILE;

    if (tid == 0) s_found = 0;
    t0 = clock64();
    __syncthreads();
    t1 = clock64();
    if (tid == 0) t_sync += (unsigned long long)(t1 - t0);

    int seed_perm_off = seed * MAX_OCTAVES * PERM_SIZE;

    // Per-cell state
    float my_dx[MAXC], my_dz[MAXC], my_cont[MAXC];
    float my_x[MAXC], my_z[MAXC];
    int my_cx[MAXC], my_cz[MAXC];
    int my_ncells;

    // ---- Tiled scan ----
    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {
            n_tiles++;

            int tox = center + tx * TILE * step;
            int toz = center + tz * TILE * step;
            int tile_w = (tx == tiles_dim - 1) ? G - tx * TILE : TILE;
            int tile_h = (tz == tiles_dim - 1) ? G - tz * TILE : TILE;
            int tile_cells = tile_w * tile_h;

            // Enumerate cells
            my_ncells = 0;
            for (int i = tid; i < tile_cells && my_ncells < MAXC; i += THREADS) {
                my_cx[my_ncells] = i % tile_w;
                my_cz[my_ncells] = i / tile_w;
                my_x[my_ncells]   = (float)(tox + my_cx[my_ncells] * step);
                my_z[my_ncells]   = (float)(toz + my_cz[my_ncells] * step);
                my_ncells++;
            }

            #pragma unroll
            for (int c = 0; c < MAXC; c++) {
                my_dx[c] = 0; my_dz[c] = 0; my_cont[c] = 0;
            }

            // ================================================================
            // Helper lambda (as macro): process one group of octaves
            // ================================================================
            #define DO_OCTAVES(j_start, j_count, X_EXPR, Z_EXPR, Y_EXPR,         \
                               ACCUM_DX, ACCUM_DZ, ACCUM_CONT, USE_FR)           \
            for (int j = j_start; j < j_start + j_count; j++) {                   \
                /* --- Phase: perm load --- */                                    \
                t0 = clock64();                                                   \
                const uint32_t *src = (const uint32_t*)(perm + seed_perm_off      \
                                                        + j * PERM_SIZE);        \
                if (tid < PERM_SIZE)                                              \
                    s_perm[tid] = ((const uint8_t*)src)[tid];                     \
                if (tid == 0) s_perm[256] = s_perm[0];  /* perm[256]==perm[0] */  \
                __syncthreads();                                                  \
                t1 = clock64();                                                   \
                if (tid == 0) {                                                   \
                    t_perm_load += (unsigned long long)(t1 - t0);                 \
                    n_perm_loads++;                                               \
                }                                                                 \
                n_octaves++;                                                      \
                                                                                  \
                float lf = s_lac[j], aj = s_amp[j];                               \
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];               \
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];      \
                float use_fr = (USE_FR) ? fr : 1.0f;                              \
                                                                                  \
                /* --- Phase: perlin --- */                                       \
                t0 = clock64();                                                   \
                _Pragma("unroll")                                                 \
                for (int c = 0; c < MAXC; c++) {                                  \
                    if (c < my_ncells) {                                          \
                        float _x = X_EXPR, _z = Z_EXPR;                           \
                        ACCUM_DX;                                                 \
                        ACCUM_CONT;                                               \
                        float v1 = perlin(s_perm, joa, job, joc,                  \
                            jh2, jd2, jt2, _z, (Y_EXPR) == 0.0f ? _x : 0.0f, 0.0f);\
                        ACCUM_DZ;                                                 \
                        n_perlin_calls += 2;                                      \
                    }                                                              \
                }                                                                  \
                t1 = clock64();                                                   \
                if (tid == 0) t_perlin += (unsigned long long)(t1 - t0);          \
            }

            // ---- Shift octave A (2 perlin calls: dx + dz) ----
            DO_OCTAVES(sh_as, sh_ac,
                my_x[c] * lf, my_z[c] * lf, 0.0f,
                my_dx[c] += aj * perlin(s_perm, joa, job, joc, jh2, jd2, jt2,
                                        _x, 0.0f, _z),
                my_dz[c] += aj * v1,
                , false);

            // ---- Shift octave B ----
            DO_OCTAVES(sh_bs, sh_bc,
                my_x[c] * lf * fr, my_z[c] * lf * fr, 0.0f,
                my_dx[c] += aj * perlin(s_perm, joa, job, joc, jh2, jd2, jt2,
                                        _x, 0.0f, _z),
                my_dz[c] += aj * v1,
                , false);

            // ---- Continentalness octave A (1 perlin call per cell) ----
            DO_OCTAVES(ct_as, ct_ac,
                (my_x[c] + my_dx[c] * sh4) * lf,
                (my_z[c] + my_dz[c] * sh4) * lf,
                0.0f,
                ,
                ,
                my_cont[c] += aj * perlin(s_perm, joa, job, joc, jh2, jd2, jt2,
                                          _x, 0.0f, _z),
                false);

            // ---- Continentalness octave B ----
            DO_OCTAVES(ct_bs, ct_bc,
                (my_x[c] + my_dx[c] * sh4) * lf * fr,
                (my_z[c] + my_dx[c] * sh4) * lf * fr,
                0.0f,
                ,
                ,
                my_cont[c] += aj * perlin(s_perm, joa, job, joc, jh2, jd2, jt2,
                                          _x, 0.0f, _z),
                false);

            #undef DO_OCTAVES

            // ---- Write to s_grid ----
            #pragma unroll
            for (int c = 0; c < MAXC; c++) {
                if (c < my_ncells) {
                    s_grid[my_cz[c]][my_cx[c]] = my_cont[c] * ct_amp;
                }
            }
            t0 = clock64();
            __syncthreads();
            t1 = clock64();
            if (tid == 0) t_sync += (unsigned long long)(t1 - t0);

            // ---- K×K detection ----
            if (!s_found) {
                int bpa_x = tile_w - K + 1, bpa_z = tile_h - K + 1;
                if (bpa_x > 0 && bpa_z > 0) {
                    t0 = clock64();
                    int tblks = bpa_x * bpa_z;
                    for (int i = tid; i < tblks; i += THREADS) {
                        int bx = i % bpa_x, bz = i / bpa_x;
                        int hit = 1;
                        #pragma unroll
                        for (int dz2 = 0; dz2 < K && hit; dz2++)
                            #pragma unroll
                            for (int dx2 = 0; dx2 < K; dx2++)
                                if (s_grid[bz+dz2][bx+dx2] >= -1.05f)
                                    { hit = 0; break; }
                        if (hit) {
                            if (atomicExch(&s_found, 1) == 0) {
                                hit_flags[seed] = 1;
                                hit_gx[seed] = tox + (bx + K/2) * step;
                                hit_gz[seed] = toz + (bz + K/2) * step;
                            }
                        }
                    }
                    t1 = clock64();
                    if (tid == 0) t_kx_check += (unsigned long long)(t1 - t0);
                }
            }

            t0 = clock64();
            __syncthreads();
            t1 = clock64();
            if (tid == 0) t_sync += (unsigned long long)(t1 - t0);

            if (s_found) goto write_debug;
        }
    }

write_debug:
    // Write debug timings (thread 0 only)
    if (tid == 0 && debug != NULL) {
        debug[seed].t_perm_load   = t_perm_load;
        debug[seed].t_perlin      = t_perlin;
        debug[seed].t_kx_check    = t_kx_check;
        debug[seed].t_sync        = t_sync;
        debug[seed].n_perlin_calls = n_perlin_calls;
        debug[seed].n_octaves     = n_octaves;
        debug[seed].n_tiles       = n_tiles;
        debug[seed].n_perm_loads  = n_perm_loads;
    }
}

// ---- Host wrapper ----
extern "C" __declspec(dllexport) int gpu_scan_seeds(
    void *d_perm, void *d_oa, void *d_ob, void *d_oc,
    void *d_amp, void *d_lac, void *d_h2, void *d_d2, void *d_t2,
    void *d_ranges, void *d_dbl_amps, int num_seeds,
    int G, int step, int K,
    void *d_hit_flags, void *d_hit_x, void *d_hit_z,
    void *d_debug)
{
    sparse_scan<<<num_seeds, THREADS>>>(
        (const uint8_t*)d_perm, (const float*)d_oa, (const float*)d_ob,
        (const float*)d_oc, (const float*)d_amp, (const float*)d_lac,
        (const uint8_t*)d_h2, (const float*)d_d2, (const float*)d_t2,
        (const int*)d_ranges, (const float*)d_dbl_amps, num_seeds,
        G, step, K,
        (int*)d_hit_flags, (int*)d_hit_x, (int*)d_hit_z,
        (DebugTiming*)d_debug);
    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : 1;
}

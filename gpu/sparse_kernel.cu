/*
 * sparse_kernel.cu — Tiled per-seed GPU kernel for G×G mushroom detection.
 *
 * One block per seed, 256 threads. Single G×G grid centered at origin.
 * Tiled in TILE×TILE cells to handle large G (up to 512+).
 *
 * Tradeoff: K×K blocks that straddle tile boundaries are NOT detected.
 * This trades ~2/TILE of hits for the ability to scan arbitrarily large grids.
 * For TILE=32, K=2: miss rate ≈ 6.25%.
 *
 * Shared memory: ~11 KB (TILE=32, includes 6.2KB perm table).
 * Launch bounds: 256 threads, 4 blocks/SM for good occupancy.
 *
 * Future: register-stored perm tables with warp shuffling to eliminate
 * shared memory latency on the perm lookups.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_OCTAVES 24
#define PERM_SIZE  257
#define THREADS    256
#define TILE       32    // 32×32 cells/tile = 1024 cells = 4 cells/thread

// ---- Gradient table (constant memory, cached) ----
__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

// ---- 3D Perlin noise sample (y=0 fast path from cubiomes) ----
// __noinline__ keeps perlin registers from spilling into the main kernel.
__device__ __noinline__ float perlin(
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

    int va = perm[h1] + h2, vb = perm[h1+1] + h2;
    int v2a = perm[va & 0xFF] + h3, v2b = perm[(va & 0xFF)+1] + h3;
    int v3a = perm[vb & 0xFF] + h3, v3b = perm[(vb & 0xFF)+1] + h3;
    int v4a = perm[v2a & 0xFF],    v4b = perm[(v2a & 0xFF)+1];
    int v5a = perm[v2b & 0xFF],    v5b = perm[(v2b & 0xFF)+1];
    int v6a = perm[v3a & 0xFF],    v6b = perm[(v3a & 0xFF)+1];
    int v7a = perm[v3b & 0xFF],    v7b = perm[(v3b & 0xFF)+1];

    #define L(i,a,b,c) (c_grad[(i)&0xF][0]*(a) + \
                         c_grad[(i)&0xF][1]*(b) + \
                         c_grad[(i)&0xF][2]*(c))
    float l1 = L(v4a, d1, d2, d3);
    float l5 = L(v4b, d1, d2, d3-1);
    float l2 = L(v6a, d1-1, d2, d3);
    float l6 = L(v6b, d1-1, d2, d3-1);
    float l3 = L(v5a, d1, d2-1, d3);
    float l7 = L(v5b, d1, d2-1, d3-1);
    float l4 = L(v7a, d1-1, d2-1, d3);
    float l8 = L(v7b, d1-1, d2-1, d3-1);
    #undef L

    l1 += t1*(l2 - l1); l3 += t1*(l4 - l3);
    l5 += t1*(l6 - l5); l7 += t1*(l8 - l7);
    l1 += t2*(l3 - l1); l5 += t2*(l7 - l5);
    return l1 + t3*(l5 - l1);
}


// ============================================================================
// Main kernel: scan one G×G grid centered at origin for each seed.
// One block per seed, 256 threads, tiled in TILE×TILE chunks.
// ============================================================================
extern "C" __launch_bounds__(THREADS, 4) __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f / 331.0f;

    // ---- Coalesced load: perm table into shared memory ----
    __shared__ uint8_t s_perm[MAX_OCTAVES][PERM_SIZE];
    int total_bytes = MAX_OCTAVES * PERM_SIZE;
    for (int i = tid; i < total_bytes; i += THREADS)
        ((uint8_t*)s_perm)[i] = perm[seed * total_bytes + i];

    // ---- Load per-octave parameters ----
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
    __syncthreads();

    // ---- Unpack ranges ----
    int sh_as = s_ranges[0], sh_ac = s_ranges[1];
    int sh_bs = s_ranges[2], sh_bc = s_ranges[3];
    int ct_as = s_ranges[4], ct_ac = s_ranges[5];
    int ct_bs = s_ranges[6], ct_bc = s_ranges[7];
    float sh_amp = s_dbl[0], ct_amp = s_dbl[1];

    // Grid origin: -(G/2)*step so the grid covers [-G*step/2, +G*step/2)
    int center = -(G / 2) * step;

    // ---- Shared tile buffer and found flag ----
    __shared__ float s_grid[TILE][TILE];
    __shared__ int s_found;

    int tiles_dim = (G + TILE - 1) / TILE;

    // Init found flag
    if (tid == 0) s_found = 0;
    __syncthreads();

    // ---- Tiled scan ----
    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {

            // Tile bounds
            int tox = center + tx * TILE * step;
            int toz = center + tz * TILE * step;
            int tile_w = (tx == tiles_dim - 1) ? G - tx * TILE : TILE;
            int tile_h = (tz == tiles_dim - 1) ? G - tz * TILE : TILE;
            int tile_cells = tile_w * tile_h;

            // ---- Phase 1: continentalness for each cell in this tile ----
            for (int i = tid; i < tile_cells; i += THREADS) {
                int cx = i % tile_w, cz = i / tile_w;
                float x = (float)(tox + cx * step);
                float z = (float)(toz + cz * step);

                // Shift distortion (spline) — dx, dz
                float dx = 0, dz = 0;
                for (int j = sh_as; j < sh_as + sh_ac; j++) {
                    float lf = s_lac[j];
                    dx += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], x*lf, 0.0f, z*lf);
                    dz += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], z*lf, x*lf, 0.0f);
                }
                for (int j = sh_bs; j < sh_bs + sh_bc; j++) {
                    float lf = s_lac[j];
                    dx += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], x*lf*fr, 0.0f, z*lf*fr);
                    dz += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], z*lf*fr, x*lf*fr, 0.0f);
                }
                dx *= sh_amp; dz *= sh_amp;
                float px = x + dx * 4.0f, pz = z + dz * 4.0f;

                // Continentalness at shifted position
                float cont = 0;
                for (int j = ct_as; j < ct_as + ct_ac; j++) {
                    float lf = s_lac[j];
                    cont += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], px*lf, 0.0f, pz*lf);
                }
                for (int j = ct_bs; j < ct_bs + ct_bc; j++) {
                    float lf = s_lac[j];
                    cont += s_amp[j] * perlin(s_perm[j], s_oa[j], s_ob[j], s_oc[j],
                        s_h2[j], s_d2[j], s_t2[j], px*lf*fr, 0.0f, pz*lf*fr);
                }
                s_grid[cz][cx] = cont * ct_amp;
            }
            __syncthreads();

            // ---- Phase 2: K×K detection within this tile ----
            // NOTE: hits straddling tile boundaries are missed (speed tradeoff).
            // For TILE=32, K=2: ~6.25% of potential 2×2 blocks are at boundaries.
            if (!s_found) {
                int bpa_x = tile_w - K + 1, bpa_z = tile_h - K + 1;
                if (bpa_x > 0 && bpa_z > 0) {
                    int tblks = bpa_x * bpa_z;
                    for (int i = tid; i < tblks; i += THREADS) {
                        int bx = i % bpa_x, bz = i / bpa_x;
                        int hit = 1;
                        for (int dz2 = 0; dz2 < K && hit; dz2++)
                            for (int dx2 = 0; dx2 < K; dx2++)
                                if (s_grid[bz+dz2][bx+dx2] >= -1.05f)
                                    { hit = 0; break; }
                        if (hit) {
                            hit_flags[seed] = 1;
                            hit_gx[seed] = tox + (bx + K/2) * step;
                            hit_gz[seed] = toz + (bz + K/2) * step;
                            s_found = 1;
                        }
                    }
                }
            }
            __syncthreads();

            // All threads agree on s_found after __syncthreads — safe to return.
            if (s_found) return;
        }
    }
}


// ============================================================================
// Host wrapper: exposed to Python via ctypes
// ============================================================================
extern "C" __declspec(dllexport) int gpu_scan_seeds(
    void *d_perm, void *d_oa, void *d_ob, void *d_oc,
    void *d_amp, void *d_lac, void *d_h2, void *d_d2, void *d_t2,
    void *d_ranges, void *d_dbl_amps, int num_seeds,
    int G, int step, int K,
    void *d_hit_flags, void *d_hit_x, void *d_hit_z)
{
    sparse_scan<<<num_seeds, THREADS>>>(
        (const uint8_t*)d_perm, (const float*)d_oa, (const float*)d_ob,
        (const float*)d_oc, (const float*)d_amp, (const float*)d_lac,
        (const uint8_t*)d_h2, (const float*)d_d2, (const float*)d_t2,
        (const int*)d_ranges, (const float*)d_dbl_amps, num_seeds,
        G, step, K,
        (int*)d_hit_flags, (int*)d_hit_x, (int*)d_hit_z);
    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : 1;
}

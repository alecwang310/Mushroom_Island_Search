/*
 * sparse_kernel.cu — Tiled per-seed GPU kernel for G×G mushroom detection.
 *
 * One block per seed, 256 threads. Single G×G grid centered at origin.
 *
 * OPTIMIZATIONS:
 *   1. Octave-major ordering: outer loop over octaves, inner over cells.
 *      Octave params + perm table stay in smem/registers per octave.
 *   2. Cooperative perm load: 257 bytes loaded into shared memory per octave
 *      (one coalesced transaction). Zero __ldg overhead.
 *   3. #pragma unroll with compile-time MAXC=4: register arrays never spill.
 *   4. Launch bounds (256,4): 64 regs/thread, 4 blocks/SM.
 *
 * Tradeoff: K×K blocks straddling tile boundaries are NOT detected (~6% miss).
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_OCTAVES 24
#define PERM_SIZE   257
#define THREADS     256
#define TILE        32    // 32×32 cells/tile = 4 cells/thread
#define MAXC        4     // max cells per thread = ceil(1024/256)

// ---- Gradient table (constant memory) ----
__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

// ---- Perlin: perm from shared memory, octave params in registers ----
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

    #define P(idx) ((int)(perm[(idx)]))

    int va = P(h1) + h2, vb = P(h1+1) + h2;
    int v2a = P(va & 0xFF) + h3, v2b = P((va & 0xFF)+1) + h3;
    int v3a = P(vb & 0xFF) + h3, v3b = P((vb & 0xFF)+1) + h3;
    int v4a = P(v2a & 0xFF),    v4b = P((v2a & 0xFF)+1);
    int v5a = P(v2b & 0xFF),    v5b = P((v2b & 0xFF)+1);
    int v6a = P(v3a & 0xFF),    v6b = P((v3a & 0xFF)+1);
    int v7a = P(v3b & 0xFF),    v7b = P((v3b & 0xFF)+1);
    #undef P

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
    int *hit_flags, int *hit_gx, int *hit_gz)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f / 331.0f;

    // ---- Per-octave params (loaded once, ~700 bytes) ----
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

    // ---- Per-octave perm table (257 bytes, reloaded per octave) ----
    __shared__ uint8_t s_perm[PERM_SIZE];

    // ---- Tile output + found flag ----
    __shared__ float s_grid[TILE][TILE];
    __shared__ int s_found;

    __syncthreads();

    int sh_as = s_ranges[0], sh_ac = s_ranges[1];
    int sh_bs = s_ranges[2], sh_bc = s_ranges[3];
    int ct_as = s_ranges[4], ct_ac = s_ranges[5];
    int ct_bs = s_ranges[6], ct_bc = s_ranges[7];
    float sh_amp = s_dbl[0], ct_amp = s_dbl[1];
    float sh4 = sh_amp * 4.0f;

    int center = -(G / 2) * step;
    int tiles_dim = (G + TILE - 1) / TILE;

    if (tid == 0) s_found = 0;
    __syncthreads();

    // ---- Per-thread register accumulators ----
    // MAXC = compile-time constant → #pragma unroll → no local memory spill.
    float my_dx[MAXC], my_dz[MAXC], my_cont[MAXC];
    float my_x[MAXC], my_z[MAXC];
    int my_cx[MAXC], my_cz[MAXC];
    int my_ncells;

    // Base offset into global perm array for this seed
    int seed_perm_off = seed * MAX_OCTAVES * PERM_SIZE;

    // ---- Tiled scan ----
    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {

            int tox = center + tx * TILE * step;
            int toz = center + tz * TILE * step;
            int tile_w = (tx == tiles_dim - 1) ? G - tx * TILE : TILE;
            int tile_h = (tz == tiles_dim - 1) ? G - tz * TILE : TILE;
            int tile_cells = tile_w * tile_h;

            // Determine cells for this thread
            my_ncells = 0;
            for (int i = tid; i < tile_cells && my_ncells < MAXC; i += THREADS) {
                my_cx[my_ncells] = i % tile_w;
                my_cz[my_ncells] = i / tile_w;
                my_x[my_ncells]   = (float)(tox + my_cx[my_ncells] * step);
                my_z[my_ncells]   = (float)(toz + my_cz[my_ncells] * step);
                my_ncells++;
            }

            // Zero accumulators — unrolled, stays in registers
            #pragma unroll
            for (int c = 0; c < MAXC; c++) {
                my_dx[c] = 0; my_dz[c] = 0; my_cont[c] = 0;
            }

            // ---- Phase 1: Shift octave A ----
            for (int j = sh_as; j < sh_as + sh_ac; j++) {
                // Cooperative load: 256 threads → 257 bytes in 1 pass
                const uint8_t *pj = perm + seed_perm_off + j * PERM_SIZE;
                if (tid < 256) s_perm[tid] = pj[tid];
                if (tid == 0)   s_perm[256] = pj[256];
                __syncthreads();

                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];

                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells) {
                        float X = my_x[c], Z = my_z[c];
                        my_dx[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, X*lf, 0.0f, Z*lf);
                        my_dz[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, Z*lf, X*lf, 0.0f);
                    }
                }
            }

            // ---- Phase 2: Shift octave B ----
            for (int j = sh_bs; j < sh_bs + sh_bc; j++) {
                const uint8_t *pj = perm + seed_perm_off + j * PERM_SIZE;
                if (tid < 256) s_perm[tid] = pj[tid];
                if (tid == 0)   s_perm[256] = pj[256];
                __syncthreads();

                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];

                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells) {
                        float X = my_x[c], Z = my_z[c];
                        my_dx[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, X*lf*fr, 0.0f, Z*lf*fr);
                        my_dz[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, Z*lf*fr, X*lf*fr, 0.0f);
                    }
                }
            }

            // ---- Phase 3: Continentalness octave A ----
            for (int j = ct_as; j < ct_as + ct_ac; j++) {
                const uint8_t *pj = perm + seed_perm_off + j * PERM_SIZE;
                if (tid < 256) s_perm[tid] = pj[tid];
                if (tid == 0)   s_perm[256] = pj[256];
                __syncthreads();

                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];

                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells) {
                        float px = my_x[c] + my_dx[c] * sh4;
                        float pz = my_z[c] + my_dz[c] * sh4;
                        my_cont[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, px*lf, 0.0f, pz*lf);
                    }
                }
            }

            // ---- Phase 4: Continentalness octave B ----
            for (int j = ct_bs; j < ct_bs + ct_bc; j++) {
                const uint8_t *pj = perm + seed_perm_off + j * PERM_SIZE;
                if (tid < 256) s_perm[tid] = pj[tid];
                if (tid == 0)   s_perm[256] = pj[256];
                __syncthreads();

                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];

                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells) {
                        float px = my_x[c] + my_dx[c] * sh4;
                        float pz = my_z[c] + my_dz[c] * sh4;
                        my_cont[c] += aj * perlin(s_perm, joa, job, joc,
                            jh2, jd2, jt2, px*lf*fr, 0.0f, pz*lf*fr);
                    }
                }
            }

            // ---- Write to shared memory ----
            #pragma unroll
            for (int c = 0; c < MAXC; c++) {
                if (c < my_ncells) {
                    s_grid[my_cz[c]][my_cx[c]] = my_cont[c] * ct_amp;
                }
            }
            __syncthreads();

            // ---- K×K detection ----
            if (!s_found) {
                int bpa_x = tile_w - K + 1, bpa_z = tile_h - K + 1;
                if (bpa_x > 0 && bpa_z > 0) {
                    int tblks = bpa_x * bpa_z;
                    for (int i = tid; i < tblks; i += THREADS) {
                        int bx = i % bpa_x, bz = i / bpa_x;
                        int hit = 1;
                        #pragma unroll
                        for (int dz2 = 0; dz2 < K; dz2++)
                            #pragma unroll
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
            if (s_found) return;
        }
    }
}

// ---- Host wrapper ----
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

/*
 * warp_shuffle.cu — Warp-shuffle perm table, register-pinned via dual-shuffle asm.
 *
 * Perm table distributed across 32-lane warp: 2 uint32_t registers per thread.
 * Lane L owns bytes [L*8 .. L*8+7]. 32 lanes × 8 bytes = 256 bytes = perfect fit.
 *
 * KEY INSIGHT: The compiler spills PermRegs to local memory because only ONE
 * of {r0, r1} appears in each __shfl_sync call — the "unused" one looks dead
 * to the register allocator. By putting BOTH r0 and r1 as asm inputs on
 * EVERY perm_get call, the compiler cannot prove either is dead and MUST
 * keep both in registers across the entire hash chain.
 *
 * Cost: 2 shuffles per lookup instead of 1 (~5 extra cycles per lookup).
 * Benefit: zero local-memory spills, __shfl_sync reads from actual registers.
 *
 * ALL threads must call perlin_ws for MAXC iterations (dummy coords for
 * unused cells) to keep __shfl_sync(0xFFFFFFFF) valid.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_OCTAVES 24
#define PERM_STRIDE 256    // 32 lanes × 8 bytes, 8-byte aligned
#define THREADS     256
#define TILE        32
#define MAXC        4

__constant__ float c_grad_ws[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

// ---- Perm lookup: dual-shuffle with both r0/r1 as direct asm inputs ----
// By putting BOTH r0 and r1 as direct inputs to the asm volatile, the
// compiler MUST keep both in registers (no spill to local memory). Two
// shuffles execute, then the correct result is selected in C. The "wasted"
// shuffle costs ~5 cycles but prevents the __shfl_sync register-spill
// bug on sm_120. Without both registers pinned via the asm, the compiler
// spills the "unused" one to local memory where __shfl_sync misbehaves.
__device__ __forceinline__ uint8_t perm_get(uint32_t r0, uint32_t r1, int idx) {
    int lane = (idx & 0xFF) >> 3;
    int off  = idx & 7;
    int word = off >> 2;
    int shift = (off & 3) * 8;

    uint32_t v0, v1;
    asm volatile(
        "shfl.sync.idx.b32 %0, %2, %4, 31, 0xFFFFFFFF;\n\t"
        "shfl.sync.idx.b32 %1, %3, %4, 31, 0xFFFFFFFF;"
        : "=&r"(v0), "=&r"(v1)
        : "r"(r0), "r"(r1), "r"(lane)
        : "memory"
    );

    uint32_t val = word ? v1 : v0;
    return (uint8_t)(val >> shift);
}

// ---- Perlin noise — rp0/rp1 passed by value, kept in registers by perm_get ----
__device__ __forceinline__ float perlin_ws(
    uint32_t rp0, uint32_t rp1,
    float oa, float ob, float oc,
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

    // Hash chain: 14 perm lookups, each with dual-shuffle (28 total shuffles).
    // Both rp0/rp1 are live at every perm_get → no register spill.
    int va  = perm_get(rp0, rp1, h1)   + h2;
    int vb  = perm_get(rp0, rp1, h1+1) + h2;
    int v2a = perm_get(rp0, rp1, va & 0xFF) + h3;
    int v2b = perm_get(rp0, rp1, (va & 0xFF)+1) + h3;
    int v3a = perm_get(rp0, rp1, vb & 0xFF) + h3;
    int v3b = perm_get(rp0, rp1, (vb & 0xFF)+1) + h3;
    int v4a = perm_get(rp0, rp1, v2a & 0xFF);
    int v4b = perm_get(rp0, rp1, (v2a & 0xFF)+1);
    int v5a = perm_get(rp0, rp1, v2b & 0xFF);
    int v5b = perm_get(rp0, rp1, (v2b & 0xFF)+1);
    int v6a = perm_get(rp0, rp1, v3a & 0xFF);
    int v6b = perm_get(rp0, rp1, (v3a & 0xFF)+1);
    int v7a = perm_get(rp0, rp1, v3b & 0xFF);
    int v7b = perm_get(rp0, rp1, (v3b & 0xFF)+1);

    #define L(i,a,b,c) (c_grad_ws[(i)&0xF][0]*(a) + \
                         c_grad_ws[(i)&0xF][1]*(b) + \
                         c_grad_ws[(i)&0xF][2]*(c))
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
extern "C" __launch_bounds__(THREADS, 4) __global__ void sparse_scan_ws(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    int lane = tid & 31;
    float fr = 337.0f / 331.0f;

    // ---- Octave params ----
    __shared__ float s_oa[MAX_OCTAVES], s_ob[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    if (tid < MAX_OCTAVES) {
        int src_idx = seed * MAX_OCTAVES + tid;
        s_oa[tid]  = oa[src_idx];  s_ob[tid]  = ob[src_idx];  s_oc[tid] = oc[src_idx];
        s_amp[tid] = amp[src_idx]; s_lac[tid] = lac[src_idx];
        s_h2[tid]  = h2[src_idx];  s_d2[tid]  = d2[src_idx];  s_t2[tid] = t2[src_idx];
    }
    __shared__ int s_ranges[8]; __shared__ float s_dbl[2];
    if (tid < 8) s_ranges[tid] = ranges[seed * 8 + tid];
    if (tid < 2) s_dbl[tid]   = dbl_amps[seed * 2 + tid];

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

    int seed_perm_off = seed * MAX_OCTAVES * PERM_STRIDE;

    float my_dx[MAXC], my_dz[MAXC], my_cont[MAXC];
    float my_x[MAXC], my_z[MAXC];
    int my_cx[MAXC], my_cz[MAXC];
    int my_ncells;

    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {

            int tox = center + tx * TILE * step;
            int toz = center + tz * TILE * step;
            int tile_w = (tx == tiles_dim - 1) ? G - tx * TILE : TILE;
            int tile_h = (tz == tiles_dim - 1) ? G - tz * TILE : TILE;
            int tile_cells = tile_w * tile_h;

            my_ncells = 0;
            for (int i = tid; i < tile_cells && my_ncells < MAXC; i += THREADS) {
                my_cx[my_ncells] = i % tile_w;
                my_cz[my_ncells] = i / tile_w;
                my_x[my_ncells] = (float)(tox + my_cx[my_ncells] * step);
                my_z[my_ncells] = (float)(toz + my_cz[my_ncells] * step);
                my_ncells++;
            }

            #pragma unroll
            for (int c = 0; c < MAXC; c++) {
                my_dx[c] = 0; my_dz[c] = 0; my_cont[c] = 0;
            }

            // ---- Shift octave A ----
            for (int j = sh_as; j < sh_as + sh_ac; j++) {
                const uint32_t *src = (const uint32_t*)(perm + seed_perm_off + j * PERM_STRIDE);
                uint32_t rp0 = src[lane * 2];
                uint32_t rp1 = src[lane * 2 + 1];
                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    int valid = (c < my_ncells);
                    float X = valid ? my_x[c] : 0.0f;
                    float Z = valid ? my_z[c] : 0.0f;
                    float dxv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, X*lf, 0.0f, Z*lf);
                    float dzv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, Z*lf, X*lf, 0.0f);
                    if (valid) { my_dx[c] += dxv; my_dz[c] += dzv; }
                }
            }

            // ---- Shift octave B ----
            for (int j = sh_bs; j < sh_bs + sh_bc; j++) {
                const uint32_t *src = (const uint32_t*)(perm + seed_perm_off + j * PERM_STRIDE);
                uint32_t rp0 = src[lane * 2];
                uint32_t rp1 = src[lane * 2 + 1];
                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    int valid = (c < my_ncells);
                    float X = valid ? my_x[c] : 0.0f;
                    float Z = valid ? my_z[c] : 0.0f;
                    float dxv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, X*lf*fr, 0.0f, Z*lf*fr);
                    float dzv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, Z*lf*fr, X*lf*fr, 0.0f);
                    if (valid) { my_dx[c] += dxv; my_dz[c] += dzv; }
                }
            }

            // ---- Continentalness octave A (1 perlin call per cell — no waste) ----
            for (int j = ct_as; j < ct_as + ct_ac; j++) {
                const uint32_t *src = (const uint32_t*)(perm + seed_perm_off + j * PERM_STRIDE);
                uint32_t rp0 = src[lane * 2];
                uint32_t rp1 = src[lane * 2 + 1];
                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    int valid = (c < my_ncells);
                    float px = valid ? (my_x[c] + my_dx[c] * sh4) : 0.0f;
                    float pz = valid ? (my_z[c] + my_dz[c] * sh4) : 0.0f;
                    float cv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, px*lf, 0.0f, pz*lf);
                    if (valid) my_cont[c] += cv;
                }
            }

            // ---- Continentalness octave B ----
            for (int j = ct_bs; j < ct_bs + ct_bc; j++) {
                const uint32_t *src = (const uint32_t*)(perm + seed_perm_off + j * PERM_STRIDE);
                uint32_t rp0 = src[lane * 2];
                uint32_t rp1 = src[lane * 2 + 1];
                float lf = s_lac[j], aj = s_amp[j];
                float joa = s_oa[j], job = s_ob[j], joc = s_oc[j];
                uint8_t jh2 = s_h2[j]; float jd2 = s_d2[j], jt2 = s_t2[j];
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    int valid = (c < my_ncells);
                    float px = valid ? (my_x[c] + my_dx[c] * sh4) : 0.0f;
                    float pz = valid ? (my_z[c] + my_dz[c] * sh4) : 0.0f;
                    float cv = aj * perlin_ws(rp0, rp1, joa, job, joc, jh2, jd2, jt2, px*lf*fr, 0.0f, pz*lf*fr);
                    if (valid) my_cont[c] += cv;
                }
            }

            // ---- Write to s_grid ----
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
                }
            }
            __syncthreads();
            if (s_found) return;
        }
    }
}

/*
 * tiered_kernel.cu — Hex grid O6+O15 mushroom prefilter.
 *
 * Grid: hex lattice (staggered rows, D=step_2x, 6 neighbors).
 * Only 2 octaves: O6 (cont A first) + O15 (cont B first).
 * No shift distortion — my_dx=my_dz=0, continentalness sampled at raw (x,z).
 * Threshold: -1.00 combined.
 *
 * Phase 1: continentalness at hex grid points → s_grid[32][32].
 * Phase 2: 6-neighbor adjacent pair detection → global hit buffer.
 * Timing: clock64() perlin vs detection.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS     256
#define TILE        32
#define MAXC        4
#define MAX_HITS    512
#define PERM_SIZE   256

__constant__ float c_grad_tier[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ __forceinline__ float perlin32(
    const uint32_t *perm, float oa, float ob, float oc,
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

    // All lookups via perm[idx] & 0xFF — uint32_t array, no bank conflicts
    int va  = (perm[h1] & 0xFF)   + h2, vb  = (perm[h1+1] & 0xFF) + h2;
    int v2a = (perm[va & 0xFF] & 0xFF) + h3, v2b = (perm[(va & 0xFF)+1] & 0xFF) + h3;
    int v3a = (perm[vb & 0xFF] & 0xFF) + h3, v3b = (perm[(vb & 0xFF)+1] & 0xFF) + h3;
    int v4a = (perm[v2a & 0xFF] & 0xFF),     v4b = (perm[(v2a & 0xFF)+1] & 0xFF);
    int v5a = (perm[v2b & 0xFF] & 0xFF),     v5b = (perm[(v2b & 0xFF)+1] & 0xFF);
    int v6a = (perm[v3a & 0xFF] & 0xFF),     v6b = (perm[(v3a & 0xFF)+1] & 0xFF);
    int v7a = (perm[v3b & 0xFF] & 0xFF),     v7b = (perm[(v3b & 0xFF)+1] & 0xFF);

    #define L(i,a,b,c) (c_grad_tier[(i)&0xF][0]*(a) + \
                         c_grad_tier[(i)&0xF][1]*(b) + \
                         c_grad_tier[(i)&0xF][2]*(c))
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


extern "C" __launch_bounds__(THREADS, 4) __global__ void tiered_scan(
    const uint8_t *perm, const float *oa, const float *ob, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    int G, int step_2x, int unused_K,
    int *hit_counts, int *hit_gx, int *hit_gz,
    unsigned long long *t_perlin, unsigned long long *t_detect)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    unsigned long long t_perlin_acc = 0, t_detect_acc = 0, t_phase;

    // ---- Perm tables as uint32 to eliminate bank conflicts ----
    // Each byte stored in its own 32-bit word. 43% conflicts → 0%.
    // s_perm[257] because perm[256] == perm[0] (hash chain wrap).
    __shared__ uint32_t s_perm6[257], s_perm15[257];
    __shared__ float s_grid[TILE][TILE];
    __syncthreads();

    // ---- Read octave 6 (cont A) params from global ----
    int s6 = seed * 24 + 6;
    float oa6 = oa[s6], ob6 = ob[s6], oc6 = oc[s6];
    float amp6 = amp[s6], lac6 = lac[s6];
    uint8_t h26 = h2[s6]; float d26 = d2[s6], t26 = t2[s6];

    // ---- Read octave 15 (cont B) params from global ----
    int s15 = seed * 24 + 15;
    float oa15 = oa[s15], ob15 = ob[s15], oc15 = oc[s15];
    float amp15 = amp[s15], lac15 = lac[s15];
    uint8_t h215 = h2[s15]; float d215 = d2[s15], t215 = t2[s15];

    // ---- Read dbl_amps (ct_amp only, shift is 0) ----
    float ct_amp = dbl_amps[seed * 2 + 1];
    float fr = 337.0f / 331.0f;  // only used for O15 (cont B)

    // ---- Hex grid constants ----
    float D = (float)step_2x;
    float D_sqrt3_2 = D * 0.8660254037844386f;
    int center_x = (int)(-(G / 2) * D);
    int center_z = (int)(-(G / 2) * D_sqrt3_2);
    int tiles_dim = (G + TILE - 1) / TILE;

    int seed_perm_off = seed * 24 * PERM_SIZE;
    int my_base = seed * MAX_HITS;
    __shared__ int s_hit_count;
    if (tid == 0) s_hit_count = 0;
    __syncthreads();

    float my_cont[MAXC], my_x[MAXC], my_z[MAXC];
    int my_cx[MAXC], my_cz[MAXC], my_ncells;

    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {

            float tox = center_x + tx * TILE * D;
            float toz = center_z + tz * TILE * D_sqrt3_2;
            int tile_w = (tx == tiles_dim-1) ? G - tx*TILE : TILE;
            int tile_h = (tz == tiles_dim-1) ? G - tz*TILE : TILE;
            int tile_cells = tile_w * tile_h;

            if (tid == 0) t_phase = clock64();
            my_ncells = 0;
            for (int i = tid; i < tile_cells && my_ncells < MAXC; i += THREADS) {
                my_cx[my_ncells] = i % tile_w;
                my_cz[my_ncells] = i / tile_w;
                float hx = tox + my_cx[my_ncells] * D;
                float hz = toz + my_cz[my_ncells] * D_sqrt3_2;
                if (my_cz[my_ncells] & 1) hx += D * 0.5f;
                my_x[my_ncells] = hx;
                my_z[my_ncells] = hz;
                my_ncells++;
            }
            for (int c = 0; c < MAXC; c++) my_cont[c] = 0;

            // ---- Load both perm tables at once (single __syncthreads) ----
            {
                const uint8_t *pb6 = perm + seed_perm_off + 6 * PERM_SIZE;
                const uint8_t *pb15 = perm + seed_perm_off + 15 * PERM_SIZE;
                if (tid < PERM_SIZE) {
                    s_perm6[tid] = (uint32_t)pb6[tid];
                    s_perm15[tid] = (uint32_t)pb15[tid];
                }
                if (tid == 0) { s_perm6[256] = s_perm6[0]; s_perm15[256] = s_perm15[0]; }
                __syncthreads();

                // Octave 6 (cont A) — no barrier needed
                float lf6 = lac6, aj6 = amp6;
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells)
                        my_cont[c] += aj6 * perlin32(s_perm6, oa6, ob6, oc6, h26, d26, t26,
                                                     my_x[c]*lf6, 0.0f, my_z[c]*lf6);
                }

                // Octave 15 (cont B, fr detuned) — no barrier
                float lf15 = lac15, aj15 = amp15;
                #pragma unroll
                for (int c = 0; c < MAXC; c++) {
                    if (c < my_ncells)
                        my_cont[c] += aj15 * perlin32(s_perm15, oa15, ob15, oc15, h215, d215, t215,
                                                      my_x[c]*lf15*fr, 0.0f, my_z[c]*lf15*fr);
                }
            }

            // Write to s_grid
            #pragma unroll
            for (int c = 0; c < MAXC; c++)
                if (c < my_ncells) s_grid[my_cz[c]][my_cx[c]] = my_cont[c] * ct_amp;
            __syncthreads();

            // ---- Detection ----
            if (tid == 0) { unsigned long long t_now = clock64(); t_perlin_acc += t_now - t_phase; t_phase = t_now; }
            for (int i = tid; i < tile_cells; i += THREADS) {
                int sx = i % tile_w, sz = i / tile_w;
                if (s_grid[sz][sx] >= -1.00f) continue;

                int has_nb = 0;
                int hx[6] = {1, -1, 0, 0, 1, -1};
                int hz[6] = {0, 0, -1, 1, -1, 1};
                for (int k = 0; k < 6 && !has_nb; k++) {
                    int nx = sx + hx[k], nz = sz + hz[k];
                    if (nx >= 0 && nx < tile_w && nz >= 0 && nz < tile_h)
                        if (s_grid[nz][nx] < -1.00f) has_nb = 1;
                }
                if (!has_nb) continue;

                int idx = atomicAdd(&s_hit_count, 1);
                if (idx < MAX_HITS) {
                    float hx_v = tox + sx * D + (sz & 1) * (D * 0.5f);
                    float hz_v = toz + sz * D_sqrt3_2;
                    hit_gx[my_base + idx] = (int)hx_v;
                    hit_gz[my_base + idx] = (int)hz_v;
                }
            }
            if (tid == 0) { unsigned long long t_now = clock64(); t_detect_acc += t_now - t_phase; t_phase = t_now; }
            __syncthreads();
        }
    }

    if (tid == 0) {
        hit_counts[seed] = (s_hit_count < MAX_HITS) ? s_hit_count : MAX_HITS;
        t_perlin[seed] = t_perlin_acc;
        t_detect[seed] = t_detect_acc;
    }
}

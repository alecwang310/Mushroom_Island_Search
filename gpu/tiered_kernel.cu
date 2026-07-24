/*
 * tiered_kernel.cu — Multi-hit GPU coarse scan + adjacent-pair detection.
 *
 * Phase 1: Continentalness at 2× step → s_grid.
 * Phase 2: Find ALL adjacent pairs in s_grid, write to global hit
 *          buffer via atomicAdd per seed. No early exit — collects
 *          all candidates so CPU can verify all of them.
 *
 * Output: hit_counts[seed] = number of pairs found
 *         hits[seed * MAX_HITS + i] = {gx, gz} for i = 0..hit_counts[seed]-1
 */
#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_OCTAVES 24
#define PERM_SIZE   256
#define THREADS     256
#define TILE        32
#define MAXC        4
#define MAX_HITS    256    // max adjacent pairs per seed (1-neighbor mode)

__constant__ float c_grad_tier[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ __forceinline__ float perlin_tier(
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

    int va  = perm[h1]   + h2, vb  = perm[h1+1] + h2;
    int v2a = perm[va & 0xFF] + h3, v2b = perm[(va & 0xFF)+1] + h3;
    int v3a = perm[vb & 0xFF] + h3, v3b = perm[(vb & 0xFF)+1] + h3;
    int v4a = perm[v2a & 0xFF],     v4b = perm[(v2a & 0xFF)+1];
    int v5a = perm[v2b & 0xFF],     v5b = perm[(v2b & 0xFF)+1];
    int v6a = perm[v3a & 0xFF],     v6b = perm[(v3a & 0xFF)+1];
    int v7a = perm[v3b & 0xFF],     v7b = perm[(v3b & 0xFF)+1];

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
    int *hit_counts, int *hit_gx, int *hit_gz)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f / 331.0f;

    __shared__ float s_oa[MAX_OCTAVES], s_ob[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    if (tid < MAX_OCTAVES) {
        int s = seed * MAX_OCTAVES + tid;
        s_oa[tid]=oa[s]; s_ob[tid]=ob[s]; s_oc[tid]=oc[s];
        s_amp[tid]=amp[s]; s_lac[tid]=lac[s];
        s_h2[tid]=h2[s]; s_d2[tid]=d2[s]; s_t2[tid]=t2[s];
    }
    __shared__ int s_ranges[8]; __shared__ float s_dbl[2];
    if (tid < 8) s_ranges[tid] = ranges[seed * 8 + tid];
    if (tid < 2) s_dbl[tid]   = dbl_amps[seed * 2 + tid];

    __shared__ uint8_t s_perm[PERM_SIZE];
    __shared__ float s_grid[TILE][TILE];
    __syncthreads();

    int sh_as=s_ranges[0], sh_ac=s_ranges[1], sh_bs=s_ranges[2], sh_bc=s_ranges[3];
    int ct_as=s_ranges[4], ct_ac=s_ranges[5], ct_bs=s_ranges[6], ct_bc=s_ranges[7];
    float sh_amp=s_dbl[0], ct_amp=s_dbl[1], sh4=sh_amp*4.0f;

    int center = -(G / 2) * step_2x;
    int tiles_dim = (G + TILE - 1) / TILE;

    int seed_perm_off = seed * MAX_OCTAVES * PERM_SIZE;
    int my_base = seed * MAX_HITS;  // this seed's region in the hit buffer
    __shared__ int s_hit_count;
    if (tid == 0) s_hit_count = 0;
    __syncthreads();

    float my_dx[MAXC], my_dz[MAXC], my_cont[MAXC];
    float my_x[MAXC], my_z[MAXC];
    int my_cx[MAXC], my_cz[MAXC], my_ncells;

    #define LOAD_P(j) { \
        const uint8_t *pb = perm + seed_perm_off + (j)*PERM_SIZE; \
        if (tid < PERM_SIZE) s_perm[tid] = pb[tid]; \
        __syncthreads(); }

    for (int tz = 0; tz < tiles_dim; tz++) {
        for (int tx = 0; tx < tiles_dim; tx++) {

            int tox = center + tx * TILE * step_2x;
            int toz = center + tz * TILE * step_2x;
            int tile_w = (tx == tiles_dim-1) ? G - tx*TILE : TILE;
            int tile_h = (tz == tiles_dim-1) ? G - tz*TILE : TILE;
            int tile_cells = tile_w * tile_h;

            my_ncells = 0;
            for (int i = tid; i < tile_cells && my_ncells < MAXC; i += THREADS) {
                my_cx[my_ncells] = i % tile_w;
                my_cz[my_ncells] = i / tile_w;
                my_x[my_ncells] = (float)(tox + my_cx[my_ncells] * step_2x);
                my_z[my_ncells] = (float)(toz + my_cz[my_ncells] * step_2x);
                my_ncells++;
            }
            for (int c = 0; c < MAXC; c++) my_dx[c]=my_dz[c]=my_cont[c]=0;

            // Shift A, B, Cont A, B (same as sparse_kernel)
            for (int j = sh_as; j < sh_as + sh_ac; j++) {
                LOAD_P(j); float lf=s_lac[j], aj=s_amp[j];
                float joa=s_oa[j], job=s_ob[j], joc=s_oc[j];
                uint8_t jh2=s_h2[j]; float jd2=s_d2[j], jt2=s_t2[j];
                #pragma unroll
                for (int c=0; c<MAXC; c++) { if(c<my_ncells){
                    float X=my_x[c], Z=my_z[c];
                    my_dx[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,X*lf,0,Z*lf);
                    my_dz[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,Z*lf,X*lf,0);
                }}
            }
            for (int j = sh_bs; j < sh_bs + sh_bc; j++) {
                LOAD_P(j); float lf=s_lac[j], aj=s_amp[j];
                float joa=s_oa[j], job=s_ob[j], joc=s_oc[j];
                uint8_t jh2=s_h2[j]; float jd2=s_d2[j], jt2=s_t2[j];
                #pragma unroll
                for (int c=0; c<MAXC; c++) { if(c<my_ncells){
                    float X=my_x[c], Z=my_z[c];
                    my_dx[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,X*lf*fr,0,Z*lf*fr);
                    my_dz[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,Z*lf*fr,X*lf*fr,0);
                }}
            }
            for (int j = ct_as; j < ct_as + ct_ac; j++) {
                LOAD_P(j); float lf=s_lac[j], aj=s_amp[j];
                float joa=s_oa[j], job=s_ob[j], joc=s_oc[j];
                uint8_t jh2=s_h2[j]; float jd2=s_d2[j], jt2=s_t2[j];
                #pragma unroll
                for (int c=0; c<MAXC; c++) { if(c<my_ncells){
                    float px=my_x[c]+my_dx[c]*sh4, pz=my_z[c]+my_dz[c]*sh4;
                    my_cont[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,px*lf,0,pz*lf);
                }}
            }
            for (int j = ct_bs; j < ct_bs + ct_bc; j++) {
                LOAD_P(j); float lf=s_lac[j], aj=s_amp[j];
                float joa=s_oa[j], job=s_ob[j], joc=s_oc[j];
                uint8_t jh2=s_h2[j]; float jd2=s_d2[j], jt2=s_t2[j];
                #pragma unroll
                for (int c=0; c<MAXC; c++) { if(c<my_ncells){
                    float px=my_x[c]+my_dx[c]*sh4, pz=my_z[c]+my_dz[c]*sh4;
                    my_cont[c]+=aj*perlin_tier(s_perm,joa,job,joc,jh2,jd2,jt2,px*lf*fr,0,pz*lf*fr);
                }}
            }

            #pragma unroll
            for (int c = 0; c < MAXC; c++)
                if (c < my_ncells) s_grid[my_cz[c]][my_cx[c]] = my_cont[c] * ct_amp;
            __syncthreads();

            // ---- Collect ALL adjacent pairs ----
            for (int i = tid; i < tile_cells; i += THREADS) {
                int sx = i % tile_w, sz = i / tile_w;
                if (s_grid[sz][sx] >= -1.05f) continue;

                // Count mushroom neighbors (4-directional)
                // Report this mushroom cell if it has >=1 mushroom neighbor
                int has_nb = 0;
                if (sx > 0       && s_grid[sz][sx-1] < -1.05f) has_nb = 1;
                if (!has_nb && sx+1 < tile_w && s_grid[sz][sx+1] < -1.05f) has_nb = 1;
                if (!has_nb && sz > 0       && s_grid[sz-1][sx] < -1.05f) has_nb = 1;
                if (!has_nb && sz+1 < tile_h && s_grid[sz+1][sx] < -1.05f) has_nb = 1;
                if (!has_nb) continue;

                int idx = atomicAdd(&s_hit_count, 1);
                if (idx < MAX_HITS) {
                    hit_gx[my_base + idx] = tox + sx * step_2x;
                    hit_gz[my_base + idx] = toz + sz * step_2x;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) hit_counts[seed] = (s_hit_count < MAX_HITS) ? s_hit_count : MAX_HITS;
    #undef LOAD_P
}

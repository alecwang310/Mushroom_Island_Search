/*
 * sparse_kernel.cu — GPU sparse-grid mushroom scanner.
 *
 * Exports: gpu_scan_seeds(...) → DLL callable from Python ctypes.
 *
 * One block per seed, 256 threads per block.
 * Processes one G×G sparse grid per wavefront (8 grids × 256 threads = 2048 samples).
 * Detection: K×K block of grid cells ALL < -1.05 → candidate.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define THREADS 256
#define MAX_GRIDS 32  // must be >= G

__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ __forceinline__ float perlin(
    const uint8_t *perm, float oa, float oc,
    uint8_t h2, float d2, float t2, float x, float z)
{
    float d1 = x + oa, d3 = z + oc;
    float i1 = floorf(d1), i3 = floorf(d3);
    d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF, h3 = ((int)i3) & 0xFF;
    float t1 = d1*d1*d1 * (d1*(d1*6.0f-15.0f)+10.0f);
    float t3 = d3*d3*d3 * (d3*(d3*6.0f-15.0f)+10.0f);

    int va = perm[h1]+h2, vb = perm[h1+1]+h2;
    int v2a=perm[va&0xFF]+h3, v2b=perm[(va&0xFF)+1]+h3;
    int v3a=perm[vb&0xFF]+h3, v3b=perm[(vb&0xFF)+1]+h3;
    int v4a=perm[v2a&0xFF], v4b=perm[(v2a&0xFF)+1];
    int v5a=perm[v2b&0xFF], v5b=perm[(v2b&0xFF)+1];
    int v6a=perm[v3a&0xFF], v6b=perm[(v3a&0xFF)+1];
    int v7a=perm[v3b&0xFF], v7b=perm[(v3b&0xFF)+1];

    #define L(i,a,b,c) (c_grad[(i)&0xF][0]*(a)+c_grad[(i)&0xF][1]*(b)+c_grad[(i)&0xF][2]*(c))
    float l1=L(v4a,d1,d2,d3);   float l5=L(v4b,d1,d2,d3-1);
    float l2=L(v6a,d1-1,d2,d3); float l6=L(v6b,d1-1,d2,d3-1);
    float l3=L(v5a,d1,d2-1,d3); float l7=L(v5b,d1,d2-1,d3-1);
    float l4=L(v7a,d1-1,d2-1,d3); float l8=L(v7b,d1-1,d2-1,d3-1);
    #undef L

    l1+=t1*(l2-l1); l3+=t1*(l4-l3); l5+=t1*(l6-l5); l7+=t1*(l8-l7);
    l1+=t2*(l3-l1); l5+=t2*(l7-l5);
    return l1+t3*(l5-l1);
}

extern "C" __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    const int *grid_offsets, int num_grids, int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz, int *hit_grid)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f/331.0f;

    __shared__ uint8_t s_perm[MAX_OCTAVES][PERM_SIZE];
    __shared__ float   s_oa[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float   s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float   s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    int s_ranges[8]; float s_dbl[2];

    // Load perlin data into shared memory
    int oct=0, b=tid;
    while (oct<MAX_OCTAVES && b<PERM_SIZE) {
        int src=(seed*MAX_OCTAVES+oct)*PERM_SIZE+b;
        s_perm[oct][b]=perm[src];
        oct+=(b+THREADS)/PERM_SIZE; b=(b+THREADS)%PERM_SIZE;
    }
    if (tid<MAX_OCTAVES) {
        int src=seed*MAX_OCTAVES+tid;
        s_oa[tid]=oa[src]; s_oc[tid]=oc[src];
        s_amp[tid]=amp[src]; s_lac[tid]=lac[src];
        s_h2[tid]=h2[src]; s_d2[tid]=d2[src]; s_t2[tid]=t2[src];
    }
    if (tid<8) s_ranges[tid]=ranges[seed*8+tid];
    if (tid<2) s_dbl[tid]=dbl_amps[seed*2+tid];
    __syncthreads();

    int sh_as=s_ranges[0],sh_ac=s_ranges[1],sh_bs=s_ranges[2],sh_bc=s_ranges[3];
    int ct_as=s_ranges[4],ct_ac=s_ranges[5],ct_bs=s_ranges[6],ct_bc=s_ranges[7];
    float sh_amp=s_dbl[0], ct_amp=s_dbl[1];
    __shared__ float s_grid[MAX_GRIDS * MAX_GRIDS];
    __shared__ int s_found;

    // Process grids one at a time
    for (int gid = 0; gid < num_grids; gid++) {
        if (tid == 0) s_found = 0;
        __syncthreads();

        int ox = grid_offsets[gid*2], oz = grid_offsets[gid*2+1];
        int total = G * G;

        // 1. Sample grid into shared memory
        for (int idx = tid; idx < total; idx += THREADS) {
            int cx = idx % G, cz = idx / G;
            float x = (float)(ox + cx * step);
            float z = (float)(oz + cz * step);

            float dx = 0, dz = 0;
            // shift A
            for (int i=sh_as; i<sh_as+sh_ac; i++) {
                float lf=s_lac[i];
                dx += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],x*lf,z*lf);
                dz += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],z*lf,x*lf);
            }
            // shift B
            for (int i=sh_bs; i<sh_bs+sh_bc; i++) {
                float lf=s_lac[i], fr2=fr;
                dx += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],x*lf*fr2,z*lf*fr2);
                dz += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],z*lf*fr2,x*lf*fr2);
            }
            dx *= sh_amp; dz *= sh_amp;
            float px = x + dx*4.0f, pz = z + dz*4.0f;

            // continentalness
            float cont = 0;
            for (int i=ct_as; i<ct_as+ct_ac; i++) {
                float lf=s_lac[i];
                cont += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf,pz*lf);
            }
            for (int i=ct_bs; i<ct_bs+ct_bc; i++) {
                float lf=s_lac[i];
                cont += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf*fr,pz*lf*fr);
            }
            s_grid[idx] = cont * ct_amp;
        }
        __syncthreads();

        // 2. Check for K×K blocks where ALL cells < -1.05
        // Only need ceil((G-K+1)^2 / THREADS) threads for this
        int blocks_per_axis = G - K + 1;
        int total_blocks = blocks_per_axis * blocks_per_axis;
        for (int idx = tid; idx < total_blocks; idx += THREADS) {
            int bx = idx % blocks_per_axis;
            int bz = idx / blocks_per_axis;
            int all_mushroom = 1;
            for (int dz2 = 0; dz2 < K && all_mushroom; dz2++) {
                for (int dx2 = 0; dx2 < K; dx2++) {
                    if (s_grid[(bz+dz2)*G + (bx+dx2)] >= -1.05f) {
                        all_mushroom = 0;
                        break;
                    }
                }
            }
            if (all_mushroom) {
                // Found! Write hit coordinates (center of the K×K block)
                hit_flags[seed] = 1;
                hit_gx[seed] = ox + (bx + K/2) * step;
                hit_gz[seed] = oz + (bz + K/2) * step;
                hit_grid[seed] = gid;
                s_found = 1;  // signal all threads to break cooperatively
            }
        }
        __syncthreads();
        // All threads break together — no divergence at next __syncthreads
        if (s_found) break;
    }
}

// ---- Host wrapper ----
extern "C" __declspec(dllexport) int gpu_scan_seeds(
    void *d_perm, void *d_oa, void *d_oc,
    void *d_amp, void *d_lac, void *d_h2, void *d_d2, void *d_t2,
    void *d_ranges, void *d_dbl_amps, int num_seeds,
    void *d_grid_offsets, int num_grids, int G, int step, int K,
    void *d_hit_flags, void *d_hit_x, void *d_hit_z, void *d_hit_grid)
{
    sparse_scan<<<num_seeds, THREADS>>>(
        (const uint8_t*)d_perm, (const float*)d_oa, (const float*)d_oc,
        (const float*)d_amp, (const float*)d_lac,
        (const uint8_t*)d_h2, (const float*)d_d2, (const float*)d_t2,
        (const int*)d_ranges, (const float*)d_dbl_amps, num_seeds,
        (const int*)d_grid_offsets, num_grids, G, step, K,
        (int*)d_hit_flags, (int*)d_hit_x, (int*)d_hit_z, (int*)d_hit_grid);
    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : 1;
}

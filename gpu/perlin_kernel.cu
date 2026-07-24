/*
 * perlin_kernel.cu — CUDA kernel for batched MC 1.18+ continentalness sampling.
 *
 * Design:
 *   - One thread block per seed (up to 1024 seeds per launch)
 *   - 256 threads per block, each processing multiple grid points
 *   - Permutation tables (24×257 bytes) loaded into shared memory once
 *   - Branch-free indexed_lerp via constant-memory gradient table
 *   - Float32 throughout for max throughput (RTX 5080)
 *
 * Compile: nvcc -O3 -arch=sm_89 -c perlin_kernel.cu -o perlin_kernel.o
 *          (sm_89 = RTX 5080/5090 Blackwell)
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* ==========================================================================
 * Constants
 * ========================================================================== */
#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define BLOCK_THREADS 256

/* Branch-free gradient table (16 vectors × 3 components).
 * Matching cubiomes indexedLerp exactly. */
__constant__ float c_gradient[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

/* ==========================================================================
 * Device-side perlin sample (y=0 fast path, float32)
 * ========================================================================== */
__device__ __forceinline__ float perlin_sample(
    const uint8_t *perm,
    float offset_a, float offset_c,
    uint8_t h2, float d2, float t2,
    float x, float z)
{
    float d1 = x + offset_a;
    float d3 = z + offset_c;
    float i1 = floorf(d1), i3 = floorf(d3);
    d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF;
    int h3 = ((int)i3) & 0xFF;

    float t1 = d1*d1*d1 * (d1 * (d1*6.0f - 15.0f) + 10.0f);
    float t3 = d3*d3*d3 * (d3 * (d3*6.0f - 15.0f) + 10.0f);

    /* Vec2 lookup chain (matching cubiomes) */
    int v1_a = perm[h1] + h2,        v1_b = perm[h1+1] + h2;
    int v2_a = perm[v1_a & 0xFF] + h3, v2_b = perm[(v1_a & 0xFF)+1] + h3;
    int v3_a = perm[v1_b & 0xFF] + h3, v3_b = perm[(v1_b & 0xFF)+1] + h3;
    int v4_a = perm[v2_a & 0xFF],      v4_b = perm[(v2_a & 0xFF)+1];
    int v5_a = perm[v2_b & 0xFF],      v5_b = perm[(v2_b & 0xFF)+1];
    int v6_a = perm[v3_a & 0xFF],      v6_b = perm[(v3_a & 0xFF)+1];
    int v7_a = perm[v3_b & 0xFF],      v7_b = perm[(v3_b & 0xFF)+1];

    /* Branch-free indexed_lerp */
    #define LERP(idx, a, b, c) \
        (c_gradient[(idx)&0xF][0]*(a) + \
         c_gradient[(idx)&0xF][1]*(b) + \
         c_gradient[(idx)&0xF][2]*(c))

    float l1 = LERP(v4_a, d1, d2, d3);
    float l5 = LERP(v4_b, d1, d2, d3-1);
    float l2 = LERP(v6_a, d1-1, d2, d3);
    float l6 = LERP(v6_b, d1-1, d2, d3-1);
    float l3 = LERP(v5_a, d1, d2-1, d3);
    float l7 = LERP(v5_b, d1, d2-1, d3-1);
    float l4 = LERP(v7_a, d1-1, d2-1, d3);
    float l8 = LERP(v7_b, d1-1, d2-1, d3-1);

    /* Trilinear interpolation */
    l1 += t1 * (l2 - l1); l3 += t1 * (l4 - l3);
    l5 += t1 * (l6 - l5); l7 += t1 * (l8 - l7);
    l1 += t2 * (l3 - l1); l5 += t2 * (l7 - l5);
    return l1 + t3 * (l5 - l1);
    #undef LERP
}

/* ==========================================================================
 * Sample one DoublePerlinNoise (octA + octB at scaled frequency)
 * ========================================================================== */
__device__ __forceinline__ float sample_dbl_perlin(
    const uint8_t (*perm)[PERM_SIZE],
    const float *offset_a, const float *offset_c,
    const float *amplitude, const float *lacunarity,
    const uint8_t *h2, const float *d2, const float *t2,
    int octA_start, int octA_count,
    int octB_start, int octB_count,
    float dbl_amp, float x, float z, float freq_ratio)
{
    float v = 0.0f;

    /* octA */
    for (int i = octA_start; i < octA_start + octA_count; i++) {
        float lf = lacunarity[i];
        v += amplitude[i] * perlin_sample(
            perm[i], offset_a[i], offset_c[i],
            h2[i], d2[i], t2[i], x * lf, z * lf);
    }

    /* octB at scaled frequency */
    for (int i = octB_start; i < octB_start + octB_count; i++) {
        float lf = lacunarity[i];
        v += amplitude[i] * perlin_sample(
            perm[i], offset_a[i], offset_c[i],
            h2[i], d2[i], t2[i],
            x * lf * freq_ratio, z * lf * freq_ratio);
    }

    return v * dbl_amp;
}

/* ==========================================================================
 * Main kernel: one block per seed
 * ========================================================================== */
extern "C" __global__ void continentalness_kernel(
    /* Input: seed data (global memory, one slice per seed) */
    const uint8_t *perm_global,       /* (num_seeds, MAX_OCTAVES, PERM_SIZE) */
    const float   *offset_a_global,   /* (num_seeds, MAX_OCTAVES) */
    const float   *offset_c_global,   /* (num_seeds, MAX_OCTAVES) */
    const float   *amplitude_global,  /* (num_seeds, MAX_OCTAVES) */
    const float   *lacunarity_global, /* (num_seeds, MAX_OCTAVES) */
    const uint8_t *h2_global,         /* (num_seeds, MAX_OCTAVES) */
    const float   *d2_global,         /* (num_seeds, MAX_OCTAVES) */
    const float   *t2_global,         /* (num_seeds, MAX_OCTAVES) */
    const int     *ranges_global,     /* (num_seeds, 8): octA/B start+count */
    const float   *dbl_amps_global,   /* (num_seeds, 2): shift_amp, cont_amp */

    /* Grid parameters */
    int x0, int z0, int step, int grid_w, int grid_h,

    /* Output */
    float  *cont_grid,    /* (num_seeds, grid_h, grid_w) */
    int    *hit_counts,   /* (num_seeds,) */
    float  *min_cont,     /* (num_seeds,) */
    int    *hit_coords     /* (num_seeds, max_hits, 2): x,z of hits, -1 if none */
)
{
    int seed_idx = blockIdx.x;
    int total_points = grid_w * grid_h;
    int tid = threadIdx.x;
    float freq_ratio = 337.0f / 331.0f;

    /* Load seed's ranges and amplitudes */
    int shift_octA_start = ranges_global[seed_idx * 8 + 0];
    int shift_octA_count = ranges_global[seed_idx * 8 + 1];
    int shift_octB_start = ranges_global[seed_idx * 8 + 2];
    int shift_octB_count = ranges_global[seed_idx * 8 + 3];
    int cont_octA_start  = ranges_global[seed_idx * 8 + 4];
    int cont_octA_count  = ranges_global[seed_idx * 8 + 5];
    int cont_octB_start  = ranges_global[seed_idx * 8 + 6];
    int cont_octB_count  = ranges_global[seed_idx * 8 + 7];
    float shift_dbl_amp  = dbl_amps_global[seed_idx * 2 + 0];
    float cont_dbl_amp   = dbl_amps_global[seed_idx * 2 + 1];

    /* Load permutation tables and parameters into shared memory */
    __shared__ uint8_t s_perm[MAX_OCTAVES][PERM_SIZE];
    __shared__ float   s_offset_a[MAX_OCTAVES];
    __shared__ float   s_offset_c[MAX_OCTAVES];
    __shared__ float   s_amplitude[MAX_OCTAVES];
    __shared__ float   s_lacunarity[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float   s_d2[MAX_OCTAVES];
    __shared__ float   s_t2[MAX_OCTAVES];

    /* Cooperative load of permutation tables (256 threads, 24 tables) */
    int oct = 0, byte = tid;
    while (oct < MAX_OCTAVES && byte < PERM_SIZE) {
        int src_idx = (seed_idx * MAX_OCTAVES + oct) * PERM_SIZE + byte;
        s_perm[oct][byte] = perm_global[src_idx];
        oct += (byte + BLOCK_THREADS) / PERM_SIZE;
        byte = (byte + BLOCK_THREADS) % PERM_SIZE;
    }

    /* Load scalar parameters (stride 1 thread per octave) */
    if (tid < MAX_OCTAVES) {
        int src = seed_idx * MAX_OCTAVES + tid;
        s_offset_a[tid]   = offset_a_global[src];
        s_offset_c[tid]   = offset_c_global[src];
        s_amplitude[tid]  = amplitude_global[src];
        s_lacunarity[tid] = lacunarity_global[src];
        s_h2[tid]         = h2_global[src];
        s_d2[tid]         = d2_global[src];
        s_t2[tid]         = t2_global[src];
    }
    __syncthreads();

    /* Each thread processes multiple grid points */
    float local_min = 1e10f;
    int local_hits = 0;

    for (int p = tid; p < total_points; p += BLOCK_THREADS) {
        int gx = p % grid_w;
        int gz = p / grid_w;
        float x = (float)(x0 + gx * step);
        float z = (float)(z0 + gz * step);

        /* Shift distortion (note coordinate swizzle for dz) */
        float dx = sample_dbl_perlin(
            s_perm, s_offset_a, s_offset_c, s_amplitude, s_lacunarity,
            s_h2, s_d2, s_t2,
            shift_octA_start, shift_octA_count,
            shift_octB_start, shift_octB_count,
            shift_dbl_amp, x, z, freq_ratio);
        float dz = sample_dbl_perlin(
            s_perm, s_offset_a, s_offset_c, s_amplitude, s_lacunarity,
            s_h2, s_d2, s_t2,
            shift_octA_start, shift_octA_count,
            shift_octB_start, shift_octB_count,
            shift_dbl_amp, z, x, freq_ratio);  /* swizzle! */

        float px = x + dx * 4.0f;
        float pz = z + dz * 4.0f;

        /* Continentalness at shifted coordinate */
        float cont = sample_dbl_perlin(
            s_perm, s_offset_a, s_offset_c, s_amplitude, s_lacunarity,
            s_h2, s_d2, s_t2,
            cont_octA_start, cont_octA_count,
            cont_octB_start, cont_octB_count,
            cont_dbl_amp, px, pz, freq_ratio);

        /* Store in output grid */
        int out_idx = seed_idx * total_points + p;
        cont_grid[out_idx] = cont;

        if (cont < local_min) local_min = cont;
        if (cont > -1.2f && cont < -1.05f) {
            local_hits++;
            /* Atomic write hit coordinate (if we have room) */
            /* Simplified: just increment counter, coordinate stored via
               a second pass or the host reads cont_grid directly */
        }
    }

    /* Reduction: find block minimum and hit count */
    __shared__ float s_min[BLOCK_THREADS];
    __shared__ int   s_hits[BLOCK_THREADS];
    s_min[tid] = local_min;
    s_hits[tid] = local_hits;
    __syncthreads();

    /* Block reduction (256 → 1) */
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_min[tid + stride] < s_min[tid])
                s_min[tid] = s_min[tid + stride];
            s_hits[tid] += s_hits[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        hit_counts[seed_idx] = s_hits[0];
        min_cont[seed_idx] = s_min[0];
    }
}

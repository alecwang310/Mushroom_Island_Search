/*
 * tiered_kernel.cu — Hex-grid O6+O15 mushroom triple prefilter.
 *
 * Default path:
 *   - Each warp keeps both 256-byte permutation tables in packed registers.
 *   - One lane owns eight consecutive bytes (two uint32 registers) per table.
 *   - Adjacent permutation lookups use three PTX shuffles per pair instead of
 *     fourteen conflict-prone shared-memory loads per Perlin evaluation.
 *   - One cell is evaluated per thread wave to keep register pressure bounded.
 *   - Detection uses one 32-bit mask per row instead of a float shared grid.
 *     A hit is the center plus two true hex neighbors; the two directions
 *     and coarse-row parity are packed into the returned geometry code.
 *
 * Build with -DTIERED_USE_WARP_PERM=0 for the shared-memory reference path.
 * If ptxas reports spills, benchmark TIERED_MIN_BLOCKS_PER_SM=3 before raising
 * register limits manually; the four permutation words are used every wave and
 * should remain resident rather than being spilled to local/L2 memory.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include "../engine/continentalness.h"

#define THREADS  256
#define TILE     32
#define WAVES    4
#define MAX_HITS 512
#define FULL_MASK 0xFFFFFFFFu

#ifndef TIERED_USE_WARP_PERM
#define TIERED_USE_WARP_PERM 1
#endif

#ifndef TIERED_MIN_BLOCKS_PER_SM
#define TIERED_MIN_BLOCKS_PER_SM 4
#endif

static_assert(THREADS % 32 == 0, "tiered_scan requires whole warps");
static_assert(THREADS * WAVES == TILE * TILE,
              "thread waves must cover one complete tile");

__device__ __forceinline__ float fade(float value) {
    return value * value * value
        * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float grad_dot(
    uint32_t hash, float x, float y, float z)
{
    uint32_t h = hash & 15u;
    float u = h < 8u ? x : y;
    float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    uint32_t u_bits = __float_as_uint(u) ^ ((h & 1u) * 0x80000000u);
    uint32_t v_bits = __float_as_uint(v) ^ (((h >> 1) & 1u) * 0x80000000u);
    return __uint_as_float(u_bits) + __uint_as_float(v_bits);
}

#if TIERED_USE_WARP_PERM

__device__ __forceinline__ void keep_perm_words(
    uint32_t &perm6_lo, uint32_t &perm6_hi,
    uint32_t &perm15_lo, uint32_t &perm15_hi)
{
    asm volatile(""
        : "+r"(perm6_lo), "+r"(perm6_hi),
          "+r"(perm15_lo), "+r"(perm15_hi));
}

__device__ __forceinline__ uint32_t perm_pair_warp(
    uint32_t local_lo, uint32_t local_hi, uint32_t index)
{
    uint32_t wrapped = index & 0xFFu;
    uint32_t source_lane = wrapped >> 3;
    uint32_t next_lane = (source_lane + 1u) & 31u;
    uint32_t byte_offset = wrapped & 7u;

    uint32_t source_lo, source_hi, next_lo;
    asm volatile(
        "shfl.sync.idx.b32 %0, %3, %5, 31, 0xFFFFFFFF;\n\t"
        "shfl.sync.idx.b32 %1, %4, %5, 31, 0xFFFFFFFF;\n\t"
        "shfl.sync.idx.b32 %2, %3, %6, 31, 0xFFFFFFFF;"
        : "=&r"(source_lo), "=&r"(source_hi), "=&r"(next_lo)
        : "r"(local_lo), "r"(local_hi),
          "r"(source_lane), "r"(next_lane));

    uint32_t selector = byte_offset | ((byte_offset + 1u) << 4);
    uint32_t adjacent;
    asm volatile("prmt.b32 %0, %1, %2, %3;"
        : "=r"(adjacent)
        : "r"(source_lo), "r"(source_hi), "r"(selector));

    uint32_t crossing = (source_hi >> 24) | ((next_lo & 0xFFu) << 8);
    return (byte_offset == 7u ? crossing : adjacent) & 0xFFFFu;
}

__device__ __forceinline__ float perlin_warp(
    uint32_t perm_lo, uint32_t perm_hi,
    float offset_x, float offset_z,
    uint32_t cached_h2, float cached_d2, float cached_t2,
    float x, float z)
{
    float dx = x + offset_x;
    float dz = z + offset_z;
    int cell_x = __float2int_rd(dx);
    int cell_z = __float2int_rd(dz);
    dx -= (float)cell_x;
    dz -= (float)cell_z;
    uint32_t h1 = (uint32_t)cell_x & 0xFFu;
    uint32_t h3 = (uint32_t)cell_z & 0xFFu;
    float tx = fade(dx);
    float tz = fade(dz);

    uint32_t pair = perm_pair_warp(perm_lo, perm_hi, h1);
    uint32_t va = (pair & 0xFFu) + cached_h2;
    uint32_t vb = (pair >> 8) + cached_h2;

    pair = perm_pair_warp(perm_lo, perm_hi, va);
    uint32_t v2a = (pair & 0xFFu) + h3;
    uint32_t v2b = (pair >> 8) + h3;
    pair = perm_pair_warp(perm_lo, perm_hi, vb);
    uint32_t v3a = (pair & 0xFFu) + h3;
    uint32_t v3b = (pair >> 8) + h3;

    pair = perm_pair_warp(perm_lo, perm_hi, v2a);
    uint32_t v4a = pair & 0xFFu;
    uint32_t v4b = pair >> 8;
    pair = perm_pair_warp(perm_lo, perm_hi, v2b);
    uint32_t v5a = pair & 0xFFu;
    uint32_t v5b = pair >> 8;
    pair = perm_pair_warp(perm_lo, perm_hi, v3a);
    uint32_t v6a = pair & 0xFFu;
    uint32_t v6b = pair >> 8;
    pair = perm_pair_warp(perm_lo, perm_hi, v3b);
    uint32_t v7a = pair & 0xFFu;
    uint32_t v7b = pair >> 8;

    float l1 = grad_dot(v4a, dx, cached_d2, dz);
    float l5 = grad_dot(v4b, dx, cached_d2, dz - 1.0f);
    float l2 = grad_dot(v6a, dx - 1.0f, cached_d2, dz);
    float l6 = grad_dot(v6b, dx - 1.0f, cached_d2, dz - 1.0f);
    float l3 = grad_dot(v5a, dx, cached_d2 - 1.0f, dz);
    float l7 = grad_dot(v5b, dx, cached_d2 - 1.0f, dz - 1.0f);
    float l4 = grad_dot(v7a, dx - 1.0f, cached_d2 - 1.0f, dz);
    float l8 = grad_dot(v7b, dx - 1.0f, cached_d2 - 1.0f, dz - 1.0f);

    l1 = fmaf(tx, l2 - l1, l1);
    l3 = fmaf(tx, l4 - l3, l3);
    l5 = fmaf(tx, l6 - l5, l5);
    l7 = fmaf(tx, l8 - l7, l7);
    l1 = fmaf(cached_t2, l3 - l1, l1);
    l5 = fmaf(cached_t2, l7 - l5, l5);
    return fmaf(tz, l5 - l1, l1);
}

#else

__device__ __forceinline__ uint32_t perm_pair_shared(
    const uint32_t *perm, uint32_t index)
{
    uint32_t wrapped = index & 0xFFu;
    return (perm[wrapped] & 0xFFu)
        | ((perm[(wrapped + 1u) & 0xFFu] & 0xFFu) << 8);
}

__device__ __forceinline__ float perlin_shared(
    const uint32_t *perm,
    float offset_x, float offset_z,
    uint32_t cached_h2, float cached_d2, float cached_t2,
    float x, float z)
{
    float dx = x + offset_x;
    float dz = z + offset_z;
    int cell_x = __float2int_rd(dx);
    int cell_z = __float2int_rd(dz);
    dx -= (float)cell_x;
    dz -= (float)cell_z;
    uint32_t h1 = (uint32_t)cell_x & 0xFFu;
    uint32_t h3 = (uint32_t)cell_z & 0xFFu;
    float tx = fade(dx);
    float tz = fade(dz);

    uint32_t pair = perm_pair_shared(perm, h1);
    uint32_t va = (pair & 0xFFu) + cached_h2;
    uint32_t vb = (pair >> 8) + cached_h2;
    pair = perm_pair_shared(perm, va);
    uint32_t v2a = (pair & 0xFFu) + h3;
    uint32_t v2b = (pair >> 8) + h3;
    pair = perm_pair_shared(perm, vb);
    uint32_t v3a = (pair & 0xFFu) + h3;
    uint32_t v3b = (pair >> 8) + h3;
    pair = perm_pair_shared(perm, v2a);
    uint32_t v4a = pair & 0xFFu;
    uint32_t v4b = pair >> 8;
    pair = perm_pair_shared(perm, v2b);
    uint32_t v5a = pair & 0xFFu;
    uint32_t v5b = pair >> 8;
    pair = perm_pair_shared(perm, v3a);
    uint32_t v6a = pair & 0xFFu;
    uint32_t v6b = pair >> 8;
    pair = perm_pair_shared(perm, v3b);
    uint32_t v7a = pair & 0xFFu;
    uint32_t v7b = pair >> 8;

    float l1 = grad_dot(v4a, dx, cached_d2, dz);
    float l5 = grad_dot(v4b, dx, cached_d2, dz - 1.0f);
    float l2 = grad_dot(v6a, dx - 1.0f, cached_d2, dz);
    float l6 = grad_dot(v6b, dx - 1.0f, cached_d2, dz - 1.0f);
    float l3 = grad_dot(v5a, dx, cached_d2 - 1.0f, dz);
    float l7 = grad_dot(v5b, dx, cached_d2 - 1.0f, dz - 1.0f);
    float l4 = grad_dot(v7a, dx - 1.0f, cached_d2 - 1.0f, dz);
    float l8 = grad_dot(v7b, dx - 1.0f, cached_d2 - 1.0f, dz - 1.0f);

    l1 = fmaf(tx, l2 - l1, l1);
    l3 = fmaf(tx, l4 - l3, l3);
    l5 = fmaf(tx, l6 - l5, l5);
    l7 = fmaf(tx, l8 - l7, l7);
    l1 = fmaf(cached_t2, l3 - l1, l1);
    l5 = fmaf(cached_t2, l7 - l5, l5);
    return fmaf(tz, l5 - l1, l1);
}

#endif

__device__ __forceinline__ void emit_warp_hits(
    uint32_t candidate_mask, int lane, int hit_x, int hit_z, int hit_code,
    int *seed_hit_count, int hit_capacity, int *hit_count,
    int seed, int4 *hits)
{
    if (candidate_mask == 0u) return;

    int warp_count = __popc(candidate_mask);
    int seed_base = 0;
    if (lane == 0)
        seed_base = atomicAdd(seed_hit_count, warp_count);
    seed_base = __shfl_sync(FULL_MASK, seed_base, 0);

    int accepted = MAX_HITS - seed_base;
    if (accepted < 0) accepted = 0;
    if (accepted > warp_count) accepted = warp_count;

    int output_base = 0;
    if (lane == 0 && accepted > 0)
        output_base = atomicAdd(hit_count, accepted);
    output_base = __shfl_sync(FULL_MASK, output_base, 0);

    uint32_t lane_bit = 1u << lane;
    int rank = __popc(candidate_mask & (lane_bit - 1u));
    if ((candidate_mask & lane_bit) && rank < accepted) {
        int output_index = output_base + rank;
        if (output_index < hit_capacity)
            hits[output_index] = make_int4(seed, hit_x, hit_z, hit_code);
    }
}

extern "C" __launch_bounds__(THREADS, TIERED_MIN_BLOCKS_PER_SM)
__global__ void tiered_scan(
    const ContTieredParams *params, int num_seeds,
    int G, int step_2x, float threshold,
    int hit_capacity, int *hit_count,
    int4 *hits)
{
    int seed = blockIdx.x;
    int tid = threadIdx.x;
    if (seed >= num_seeds) return;

    int lane = tid & 31;
    int warp = tid >> 5;
    const ContTieredParams *seed_params = params + seed;

    float oa6 = seed_params->offset_a[0];
    float oc6 = seed_params->offset_c[0];
    float amp6 = seed_params->amplitude[0];
    float lac6 = seed_params->lacunarity[0];
    uint32_t h26 = seed_params->cached_h2[0];
    float d26 = seed_params->cached_d2[0];
    float t26 = seed_params->cached_t2[0];
    float oa15 = seed_params->offset_a[1];
    float oc15 = seed_params->offset_c[1];
    float amp15 = seed_params->amplitude[1];
    float lac15 = seed_params->lacunarity[1];
    uint32_t h215 = seed_params->cached_h2[1];
    float d215 = seed_params->cached_d2[1];
    float t215 = seed_params->cached_t2[1];
    float ct_amp = seed_params->cont_dbl_amp;
    float frequency_ratio = 337.0f / 331.0f;

#if TIERED_USE_WARP_PERM
    const uint2 *perm6_words = reinterpret_cast<const uint2*>(seed_params->perm[0]);
    const uint2 *perm15_words = reinterpret_cast<const uint2*>(seed_params->perm[1]);
    uint2 perm6 = perm6_words[lane];
    uint2 perm15 = perm15_words[lane];
#else
    __shared__ uint32_t shared_perm6[CONT_TIERED_PERM_SIZE];
    __shared__ uint32_t shared_perm15[CONT_TIERED_PERM_SIZE];
    if (tid < CONT_TIERED_PERM_SIZE) {
        shared_perm6[tid] = (uint32_t)seed_params->perm[0][tid];
        shared_perm15[tid] = (uint32_t)seed_params->perm[1][tid];
    }
#endif

    __shared__ uint32_t row_masks[TILE];
    __shared__ int seed_hit_count;
    if (tid == 0) seed_hit_count = 0;
    __syncthreads();

    float spacing_x = (float)step_2x;
    float spacing_z = spacing_x * 0.8660254037844386f;
    float half_spacing_x = spacing_x * 0.5f;
    int center_x = (int)(-(G / 2) * spacing_x);
    int center_z = (int)(-(G / 2) * spacing_z);
    int tiles_dim = (G + TILE - 1) / TILE;

    for (int tile_z = 0; tile_z < tiles_dim; tile_z++) {
        for (int tile_x = 0; tile_x < tiles_dim; tile_x++) {
            float tile_origin_x = center_x + tile_x * TILE * spacing_x;
            float tile_origin_z = center_z + tile_z * TILE * spacing_z;
            int tile_width = tile_x == tiles_dim - 1 ? G - tile_x * TILE : TILE;
            int tile_height = tile_z == tiles_dim - 1 ? G - tile_z * TILE : TILE;
            int tile_cells = tile_width * tile_height;
            bool full_tile = tile_width == TILE && tile_height == TILE;

            if (!full_tile) {
                if (tid < TILE) row_masks[tid] = 0u;
                __syncthreads();
            }

            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                bool valid;
                int cell_x;
                int cell_z;
                if (full_tile) {
                    valid = true;
                    cell_x = lane;
                    cell_z = warp + wave * (THREADS / 32);
                } else {
                    valid = cell_index < tile_cells;
                    cell_x = valid ? cell_index % tile_width : 0;
                    cell_z = valid ? cell_index / tile_width : 0;
                }

                float sample_x = valid
                    ? tile_origin_x + cell_x * spacing_x
                        + (cell_z & 1) * half_spacing_x
                    : 0.0f;
                float sample_z = valid
                    ? tile_origin_z + cell_z * spacing_z
                    : 0.0f;

#if TIERED_USE_WARP_PERM
                keep_perm_words(perm6.x, perm6.y, perm15.x, perm15.y);
                float continentalness = amp6 * perlin_warp(
                    perm6.x, perm6.y, oa6, oc6, h26, d26, t26,
                    sample_x * lac6, sample_z * lac6);
                continentalness += amp15 * perlin_warp(
                    perm15.x, perm15.y, oa15, oc15, h215, d215, t215,
                    sample_x * lac15 * frequency_ratio,
                    sample_z * lac15 * frequency_ratio);
#else
                float continentalness = amp6 * perlin_shared(
                    shared_perm6, oa6, oc6, h26, d26, t26,
                    sample_x * lac6, sample_z * lac6);
                continentalness += amp15 * perlin_shared(
                    shared_perm15, oa15, oc15, h215, d215, t215,
                    sample_x * lac15 * frequency_ratio,
                    sample_z * lac15 * frequency_ratio);
#endif

                bool low = valid && continentalness * ct_amp < threshold;
                if (full_tile) {
                    uint32_t mask = __ballot_sync(FULL_MASK, low);
                    if (lane == 0) row_masks[cell_z] = mask;
                } else if (low) {
                    atomicOr(&row_masks[cell_z], 1u << cell_x);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                bool valid;
                int cell_x;
                int cell_z;
                if (full_tile) {
                    valid = true;
                    cell_x = lane;
                    cell_z = warp + wave * (THREADS / 32);
                } else {
                    valid = cell_index < tile_cells;
                    cell_x = valid ? cell_index % tile_width : 0;
                    cell_z = valid ? cell_index / tile_width : 0;
                }

                uint32_t row = valid ? row_masks[cell_z] : 0u;
                uint32_t bit = 1u << cell_x;
                uint32_t neighbor_mask = 0u;
                neighbor_mask |= ((row << 1) & bit) ? (1u << 0) : 0u;
                neighbor_mask |= ((row >> 1) & bit) ? (1u << 1) : 0u;
                int row_parity = cell_z & 1;
                if (valid && cell_z > 0) {
                    uint32_t above = row_masks[cell_z - 1];
                    neighbor_mask |= (above & bit) ? (1u << 2) : 0u;
                    uint32_t above_diagonal = row_parity
                        ? (above >> 1) : (above << 1);
                    neighbor_mask |= (above_diagonal & bit) ? (1u << 3) : 0u;
                }
                if (valid && cell_z + 1 < tile_height) {
                    uint32_t below = row_masks[cell_z + 1];
                    neighbor_mask |= (below & bit) ? (1u << 4) : 0u;
                    uint32_t below_diagonal = row_parity
                        ? (below >> 1) : (below << 1);
                    neighbor_mask |= (below_diagonal & bit) ? (1u << 5) : 0u;
                }

                bool candidate = valid && (row & bit)
                    && (__popc(neighbor_mask) >= 2);
                uint32_t candidate_mask = __ballot_sync(FULL_MASK, candidate);
                uint32_t first_neighbor = neighbor_mask & (~neighbor_mask + 1u);
                uint32_t remaining_neighbors = neighbor_mask ^ first_neighbor;
                uint32_t second_neighbor = remaining_neighbors
                    & (~remaining_neighbors + 1u);
                int hit_code = (int)(first_neighbor | second_neighbor)
                    | ((cell_z & 1) << 6);
                int hit_x = (int)(tile_origin_x + cell_x * spacing_x
                    + (cell_z & 1) * half_spacing_x);
                int hit_z = (int)(tile_origin_z + cell_z * spacing_z);
                emit_warp_hits(candidate_mask, lane, hit_x, hit_z, hit_code,
                               &seed_hit_count, hit_capacity, hit_count,
                               seed, hits);
            }
            __syncthreads();
            if (seed_hit_count >= MAX_HITS) return;
        }
    }
}

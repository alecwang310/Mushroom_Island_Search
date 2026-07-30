/*
 * tiered_kernel.cu — Hex-grid O6 area prefilter.
 *
 * Default path:
 *   - Each block keeps both permutation tables in packed shared memory.
 *   - Each adjacent permutation pair is stored in one uint32 entry, reducing
 *     the shared-memory lookup count without changing the Perlin sequence.
 *   - One cell is evaluated per thread wave to keep register pressure bounded.
 *   - A shared-memory label propagation pass finds connected O6 components on
 *     the 1x grid.  Seven rounds are sufficient for the default 6M gate,
 *     which is seven 250-block hex cells.
 *
 * Build with -DTIERED_USE_WARP_PERM=1 for the warp-register reference path.
 * The shared path stores each adjacent permutation pair in one 32-bit word,
 * so each permutation-pair lookup needs one shared load instead of two.
 * The default prefix path precomputes the three x-only pair stages once per
 * lane per full tile and reuses the resulting two packed second-level pairs
 * across the four waves; build with -DTIERED_USE_PREFIX_LUT=0 to disable it.
 * Build with -DTIERED_SHARED_PACKED_PAIRS=0 to restore the byte-table path.
 * Build with -DTIERED_USE_TRANSPOSED_SHARED_PERM=1 to use replicated lane
 * slots; TIERED_SHARED_REPLICAS=16 caps lookup conflicts at roughly 2-way while
 * fitting under the target GPU's per-block shared-memory limit.
 * If the warp-register control is enabled and ptxas reports spills, benchmark
 * TIERED_MIN_BLOCKS_PER_SM=3 before raising register limits manually; the four
 * permutation words are used every wave and should remain resident rather than
 * being spilled to local/L2 memory.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include "../engine/continentalness.h"
#include "tiered_hit.h"

#define THREADS  256
#define TILE     32
#define TILE_INTERIOR 26
#define TILE_HALO 3
#define WAVES    4
#define MAX_HITS 512
#define FULL_MASK 0xFFFFFFFFu
#define INVALID_LABEL -1
#define VECTOR_CELL_SIDE 256
#define VECTOR_WORDS_PER_ROW (VECTOR_CELL_SIDE / 32)
#define VECTOR_CELL_WORDS \
    (VECTOR_CELL_SIDE * VECTOR_WORDS_PER_ROW)

#ifndef TIERED_USE_WARP_PERM
#define TIERED_USE_WARP_PERM 0
#endif

#ifndef TIERED_USE_TRANSPOSED_SHARED_PERM
#define TIERED_USE_TRANSPOSED_SHARED_PERM 0
#endif

#ifndef TIERED_SHARED_PACKED_PAIRS
#define TIERED_SHARED_PACKED_PAIRS 1
#endif

#ifndef TIERED_USE_PREFIX_LUT
#define TIERED_USE_PREFIX_LUT 1
#endif

#ifndef TIERED_SHARED_REPLICAS
#define TIERED_SHARED_REPLICAS 16
#endif

#ifndef TIERED_MIN_BLOCKS_PER_SM
#define TIERED_MIN_BLOCKS_PER_SM 3
#endif

static_assert(THREADS % 32 == 0, "tiered_scan requires whole warps");
static_assert(THREADS * WAVES == TILE * TILE,
              "thread waves must cover one complete tile");
static_assert(TIERED_COMPONENT_CAP <= 8,
              "component extraction assumes at most eight cells");
#if TIERED_USE_TRANSPOSED_SHARED_PERM
static_assert(TIERED_SHARED_REPLICAS >= 1
              && TIERED_SHARED_REPLICAS <= 32
              && (TIERED_SHARED_REPLICAS
                  & (TIERED_SHARED_REPLICAS - 1)) == 0,
              "shared replicas must be a power of two from 1 to 32");
#endif

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
    const uint32_t *perm, int lane, uint32_t index)
{
    uint32_t wrapped = index & 0xFFu;
#if TIERED_USE_TRANSPOSED_SHARED_PERM
    uint32_t replica = (uint32_t)lane
        & (TIERED_SHARED_REPLICAS - 1u);
#if TIERED_SHARED_PACKED_PAIRS
    return perm[wrapped * TIERED_SHARED_REPLICAS + replica];
#else
    uint32_t first = perm[wrapped * TIERED_SHARED_REPLICAS + replica]
        & 0xFFu;
    uint32_t second = perm[((wrapped + 1u) & 0xFFu)
        * TIERED_SHARED_REPLICAS + replica] & 0xFFu;
    return first | (second << 8);
#endif
#elif TIERED_SHARED_PACKED_PAIRS
    return perm[wrapped];
#else
    return (perm[wrapped] & 0xFFu)
        | ((perm[(wrapped + 1u) & 0xFFu] & 0xFFu) << 8);
#endif
}

#if TIERED_USE_PREFIX_LUT

__device__ __forceinline__ uint32_t build_shared_prefix(
    const uint32_t *perm, int lane, uint32_t h1, uint32_t cached_h2)
{
    uint32_t pair0 = perm_pair_shared(perm, lane, h1);
    uint32_t va = (pair0 & 0xFFu) + cached_h2;
    uint32_t vb = (pair0 >> 8) + cached_h2;
    uint32_t pair1 = perm_pair_shared(perm, lane, va);
    uint32_t pair2 = perm_pair_shared(perm, lane, vb);
    return (pair1 & 0xFFFFu) | ((pair2 & 0xFFFFu) << 16);
}

#endif

__device__ __forceinline__ float perlin_shared(
    const uint32_t *perm,
    int lane,
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

    uint32_t pair = perm_pair_shared(perm, lane, h1);
    uint32_t va = (pair & 0xFFu) + cached_h2;
    uint32_t vb = (pair >> 8) + cached_h2;
    pair = perm_pair_shared(perm, lane, va);
    uint32_t v2a = (pair & 0xFFu) + h3;
    uint32_t v2b = (pair >> 8) + h3;
    pair = perm_pair_shared(perm, lane, vb);
    uint32_t v3a = (pair & 0xFFu) + h3;
    uint32_t v3b = (pair >> 8) + h3;
    pair = perm_pair_shared(perm, lane, v2a);
    uint32_t v4a = pair & 0xFFu;
    uint32_t v4b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v2b);
    uint32_t v5a = pair & 0xFFu;
    uint32_t v5b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v3a);
    uint32_t v6a = pair & 0xFFu;
    uint32_t v6b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v3b);
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

#if TIERED_USE_PREFIX_LUT

__device__ __forceinline__ float perlin_shared_prefix(
    const uint32_t *perm,
    int lane,
    uint32_t prefix,
    float offset_x, float offset_z,
    float cached_d2, float cached_t2,
    float x, float z)
{
    float dx = x + offset_x;
    float dz = z + offset_z;
    int cell_x = __float2int_rd(dx);
    int cell_z = __float2int_rd(dz);
    dx -= (float)cell_x;
    dz -= (float)cell_z;
    uint32_t h3 = (uint32_t)cell_z & 0xFFu;
    float tx = fade(dx);
    float tz = fade(dz);

    uint32_t pair1 = prefix & 0xFFFFu;
    uint32_t pair2 = prefix >> 16;
    uint32_t v2a = (pair1 & 0xFFu) + h3;
    uint32_t v2b = (pair1 >> 8) + h3;
    uint32_t v3a = (pair2 & 0xFFu) + h3;
    uint32_t v3b = (pair2 >> 8) + h3;

    uint32_t pair = perm_pair_shared(perm, lane, v2a);
    uint32_t v4a = pair & 0xFFu;
    uint32_t v4b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v2b);
    uint32_t v5a = pair & 0xFFu;
    uint32_t v5b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v3a);
    uint32_t v6a = pair & 0xFFu;
    uint32_t v6b = pair >> 8;
    pair = perm_pair_shared(perm, lane, v3b);
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

#endif

__device__ __forceinline__ bool vector_cell_enabled(
    const uint32_t *refine_cells, int cell_x, int cell_z)
{
    uint32_t wrapped_x = (uint32_t)cell_x & 0xFFu;
    uint32_t wrapped_z = (uint32_t)cell_z & 0xFFu;
    int word_index = (int)wrapped_z * VECTOR_WORDS_PER_ROW
        + (int)(wrapped_x >> 5);
    return (refine_cells[word_index] & (1u << (wrapped_x & 31u))) != 0;
}

__device__ __forceinline__ uint32_t dilate_vector_word(
    const uint32_t *candidate_cells, int row, int word)
{
    int wrapped_row = row & (VECTOR_CELL_SIDE - 1);
    int left_word = (word - 1) & (VECTOR_WORDS_PER_ROW - 1);
    int right_word = (word + 1) & (VECTOR_WORDS_PER_ROW - 1);
    int base = wrapped_row * VECTOR_WORDS_PER_ROW;
    uint32_t center = candidate_cells[base + word];
    return center | (center << 1) | (center >> 1)
        | (candidate_cells[base + left_word] >> 31)
        | (candidate_cells[base + right_word] << 31);
}

__device__ __forceinline__ int reserve_warp_hit(
    uint32_t candidate_mask, int lane,
    int *seed_hit_count, int hit_capacity, int *hit_count)
{
    if (candidate_mask == 0u) return -1;

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
    if ((candidate_mask & lane_bit) == 0 || rank >= accepted)
        return -1;
    int output_index = output_base + rank;
    return output_index < hit_capacity ? output_index : -1;
}

__device__ __forceinline__ void write_component_hit(
    int output_index, int seed, int label, int component_size,
    int tile_base_x, int tile_base_z,
    int center_x, int center_z,
    float spacing_x, float spacing_z, float half_spacing_x,
    const int *component_labels, TieredHit *hits)
{
    int anchor_cell_x = label & (TILE - 1);
    int anchor_cell_z = label / TILE;
    int anchor_global_x = tile_base_x + anchor_cell_x;
    int anchor_global_z = tile_base_z + anchor_cell_z;

    TieredHit hit = {};
    hit.seed = seed;
    hit.x = (int)(center_x + anchor_global_x * spacing_x
        + (anchor_global_z & 1) * half_spacing_x);
    hit.z = (int)(center_z + anchor_global_z * spacing_z);

    int component_count = 0;
    for (int cell_index = label;
         cell_index < TILE * TILE
             && component_count < TIERED_COMPONENT_CAP;
         cell_index++) {
        if (component_labels[cell_index] != label)
            continue;
        int cell_x = cell_index & (TILE - 1);
        int cell_z = cell_index / TILE;
        tiered_store_component_offset(
            &hit, component_count,
            cell_x - anchor_cell_x, cell_z - anchor_cell_z);
        component_count++;
    }
    if (component_count == 0) {
        tiered_store_component_offset(&hit, 0, 0, 0);
        component_count = 1;
    }
    hit.meta = tiered_hit_meta(
        anchor_global_z & 1, component_count, component_size);
    hits[output_index] = hit;
}

extern "C" __launch_bounds__(THREADS, TIERED_MIN_BLOCKS_PER_SM)
__global__ void tiered_scan(
    const ContTieredParams *params, int num_seeds,
    int G, int step_1x, float threshold, int o6_only,
    int minimum_connected_cells, int vector_screen,
    float vector_margin,
    int hit_capacity, int *hit_count,
    TieredHit *hits,
    unsigned long long *vector_stats)
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
    float oa15 = 0.0f;
    float oc15 = 0.0f;
    float amp15 = 0.0f;
    float lac15 = 0.0f;
    uint32_t h215 = 0u;
    float d215 = 0.0f;
    float t215 = 0.0f;
    if (!o6_only) {
        oa15 = seed_params->offset_a[1];
        oc15 = seed_params->offset_c[1];
        amp15 = seed_params->amplitude[1];
        lac15 = seed_params->lacunarity[1];
        h215 = seed_params->cached_h2[1];
        d215 = seed_params->cached_d2[1];
        t215 = seed_params->cached_t2[1];
    }
    float ct_amp = seed_params->cont_dbl_amp;
    float frequency_ratio = 337.0f / 331.0f;

#if TIERED_USE_WARP_PERM
    const uint2 *perm6_words = reinterpret_cast<const uint2*>(seed_params->perm[0]);
    uint2 perm6 = perm6_words[lane];
    uint2 perm15 = make_uint2(0u, 0u);
    if (!o6_only) {
        const uint2 *perm15_words = reinterpret_cast<const uint2*>(
            seed_params->perm[1]);
        perm15 = perm15_words[lane];
    }
#else
#if TIERED_USE_TRANSPOSED_SHARED_PERM
    __shared__ uint32_t shared_perm6[
        CONT_TIERED_PERM_SIZE * TIERED_SHARED_REPLICAS];
    __shared__ uint32_t shared_perm15[
        CONT_TIERED_PERM_SIZE * TIERED_SHARED_REPLICAS];
    if (tid < CONT_TIERED_PERM_SIZE) {
        uint32_t perm6_value = (uint32_t)seed_params->perm[0][tid];
        uint32_t perm15_value = 0u;
#if TIERED_SHARED_PACKED_PAIRS
        perm6_value |= (uint32_t)seed_params->perm[0]
            [(tid + 1) & (CONT_TIERED_PERM_SIZE - 1)] << 8;
        if (!o6_only) {
            perm15_value = (uint32_t)seed_params->perm[1][tid]
                | ((uint32_t)seed_params->perm[1]
                    [(tid + 1) & (CONT_TIERED_PERM_SIZE - 1)] << 8);
        }
#else
        if (!o6_only)
            perm15_value = (uint32_t)seed_params->perm[1][tid];
#endif
        #pragma unroll
        for (int copy_lane = 0;
             copy_lane < TIERED_SHARED_REPLICAS; copy_lane++) {
            int destination_replica = (lane + copy_lane)
                & (TIERED_SHARED_REPLICAS - 1);
            shared_perm6[tid * TIERED_SHARED_REPLICAS
                + destination_replica] = perm6_value;
            shared_perm15[tid * TIERED_SHARED_REPLICAS
                + destination_replica] = perm15_value;
        }
    }
#else
    __shared__ uint32_t shared_perm6[CONT_TIERED_PERM_SIZE];
    __shared__ uint32_t shared_perm15[CONT_TIERED_PERM_SIZE];
    if (tid < CONT_TIERED_PERM_SIZE) {
#if TIERED_SHARED_PACKED_PAIRS
        shared_perm6[tid] = (uint32_t)seed_params->perm[0][tid]
            | ((uint32_t)seed_params->perm[0]
                [(tid + 1) & (CONT_TIERED_PERM_SIZE - 1)] << 8);
        if (!o6_only) {
            shared_perm15[tid] = (uint32_t)seed_params->perm[1][tid]
                | ((uint32_t)seed_params->perm[1]
                    [(tid + 1) & (CONT_TIERED_PERM_SIZE - 1)] << 8);
        }
#else
        shared_perm6[tid] = (uint32_t)seed_params->perm[0][tid];
        if (!o6_only)
            shared_perm15[tid] = (uint32_t)seed_params->perm[1][tid];
#endif
    }
#endif
#endif

    __shared__ uint32_t vector_candidates[VECTOR_CELL_WORDS];
    __shared__ uint32_t vector_refine[VECTOR_CELL_WORDS];
    __shared__ int component_labels_a[TILE * TILE];
    __shared__ int component_labels_b[TILE * TILE];
    __shared__ int seed_hit_count;
    __shared__ int tile_has_refine;
    __shared__ unsigned int vector_candidate_count;
    __shared__ unsigned int vector_refine_count;
    if (tid == 0) seed_hit_count = 0;
    __syncthreads();

    bool use_vector_screen = o6_only && vector_screen;
    float vector_threshold = threshold + vector_margin;
    int warp_count = THREADS / 32;
    for (int word_index = warp;
         word_index < VECTOR_CELL_WORDS;
         word_index += warp_count) {
        int vector_cell = word_index * 32 + lane;
        int vector_x = vector_cell & (VECTOR_CELL_SIDE - 1);
        int vector_z = vector_cell >> 8;
        bool promising = true;
        if (use_vector_screen) {
#if TIERED_USE_WARP_PERM
            keep_perm_words(perm6.x, perm6.y, perm15.x, perm15.y);
            float center_value = amp6 * perlin_warp(
                perm6.x, perm6.y, 0.0f, 0.0f, h26, d26, t26,
                (float)vector_x + 0.5f,
                (float)vector_z + 0.5f) * ct_amp;
#else
            float center_value = amp6 * perlin_shared(
                shared_perm6, lane, 0.0f, 0.0f, h26, d26, t26,
                (float)vector_x + 0.5f,
                (float)vector_z + 0.5f) * ct_amp;
#endif
            promising = center_value < vector_threshold;
        }
        uint32_t promising_mask = __ballot_sync(FULL_MASK, promising);
        if (lane == 0)
            vector_candidates[word_index] = promising_mask;
    }
    __syncthreads();

    for (int word_index = tid;
         word_index < VECTOR_CELL_WORDS;
         word_index += THREADS) {
        int row = word_index / VECTOR_WORDS_PER_ROW;
        int word = word_index & (VECTOR_WORDS_PER_ROW - 1);
        uint32_t refined;
        if (use_vector_screen) {
            refined = dilate_vector_word(vector_candidates, row - 1, word)
                | dilate_vector_word(vector_candidates, row, word)
                | dilate_vector_word(vector_candidates, row + 1, word);
        } else {
            refined = FULL_MASK;
        }
        vector_refine[word_index] = refined;
    }
    __syncthreads();

    if (vector_stats) {
        if (tid == 0) {
            vector_candidate_count = 0;
            vector_refine_count = 0;
        }
        __syncthreads();
        unsigned int local_candidates = 0;
        unsigned int local_refine = 0;
        for (int word_index = tid;
             word_index < VECTOR_CELL_WORDS;
             word_index += THREADS) {
            local_candidates += __popc(vector_candidates[word_index]);
            local_refine += __popc(vector_refine[word_index]);
        }
        atomicAdd(&vector_candidate_count, local_candidates);
        atomicAdd(&vector_refine_count, local_refine);
        __syncthreads();
        if (tid == 0) {
            atomicAdd(vector_stats + 0,
                      (unsigned long long)VECTOR_CELL_SIDE * VECTOR_CELL_SIDE);
            atomicAdd(vector_stats + 1,
                      (unsigned long long)vector_candidate_count);
            atomicAdd(vector_stats + 2,
                      (unsigned long long)vector_refine_count);
        }
    }
    __syncthreads();

    float spacing_x = (float)step_1x;
    float spacing_z = spacing_x * 0.8660254037844386f;
    float half_spacing_x = spacing_x * 0.5f;
    int center_x = (int)(-(G / 2) * spacing_x);
    int center_z = (int)(-(G / 2) * spacing_z);
    int tiles_dim = (G + TILE_INTERIOR - 1) / TILE_INTERIOR;
    if (minimum_connected_cells < 1)
        minimum_connected_cells = 1;
    if (minimum_connected_cells > TILE * TILE)
        return;

    for (int tile_z = 0; tile_z < tiles_dim; tile_z++) {
        for (int tile_x = 0; tile_x < tiles_dim; tile_x++) {
            int tile_base_x = tile_x * TILE_INTERIOR - TILE_HALO;
            int tile_base_z = tile_z * TILE_INTERIOR - TILE_HALO;
            int interior_begin_x = tile_x * TILE_INTERIOR;
            int interior_begin_z = tile_z * TILE_INTERIOR;
            int interior_end_x = interior_begin_x + TILE_INTERIOR;
            int interior_end_z = interior_begin_z + TILE_INTERIOR;
            if (interior_end_x > G) interior_end_x = G;
            if (interior_end_z > G) interior_end_z = G;

            bool lane_has_refine = false;
            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                int cell_x = cell_index & (TILE - 1);
                int cell_z = cell_index / TILE;
                int global_x = tile_base_x + cell_x;
                int global_z = tile_base_z + cell_z;
                bool valid = global_x >= 0 && global_x < G
                    && global_z >= 0 && global_z < G;
                bool refine = valid;
                if (valid && use_vector_screen) {
                    float sample_x = center_x + global_x * spacing_x
                        + (global_z & 1) * half_spacing_x;
                    float sample_z = center_z + global_z * spacing_z;
                    int vector_x = __float2int_rd(sample_x * lac6 + oa6);
                    int vector_z = __float2int_rd(sample_z * lac6 + oc6);
                    refine = vector_cell_enabled(
                        vector_refine, vector_x, vector_z);
                }
                lane_has_refine |= refine;
            }
            if (tid == 0) tile_has_refine = 0;
            __syncthreads();
            bool warp_has_refine = __any_sync(FULL_MASK, lane_has_refine);
            if (lane == 0 && warp_has_refine)
                atomicExch(&tile_has_refine, 1);
            __syncthreads();
            if (!tile_has_refine)
                continue;

#if !TIERED_USE_WARP_PERM && TIERED_USE_PREFIX_LUT
            uint32_t prefix6 = 0u;
            uint32_t prefix15 = 0u;
            if (lane_has_refine) {
                int prefix_global_z = tile_base_z + warp;
                float prefix_sample_x = center_x
                    + (tile_base_x + lane) * spacing_x
                    + (prefix_global_z & 1) * half_spacing_x;
                uint32_t prefix_h16 = (uint32_t)__float2int_rd(
                    prefix_sample_x * lac6 + oa6);
                prefix6 = build_shared_prefix(
                    shared_perm6, lane, prefix_h16, h26);
                if (!o6_only) {
                    uint32_t prefix_h115 = (uint32_t)__float2int_rd(
                        prefix_sample_x * lac15 * frequency_ratio + oa15);
                    prefix15 = build_shared_prefix(
                        shared_perm15, lane, prefix_h115, h215);
                }
            }
#endif

            int *labels_current = component_labels_a;
            int *labels_next = component_labels_b;
            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                int cell_x;
                int cell_z;
                cell_x = cell_index & (TILE - 1);
                cell_z = cell_index / TILE;
                int global_x = tile_base_x + cell_x;
                int global_z = tile_base_z + cell_z;
                bool valid = global_x >= 0 && global_x < G
                    && global_z >= 0 && global_z < G;

                float sample_x = valid
                    ? center_x + global_x * spacing_x
                        + (global_z & 1) * half_spacing_x
                    : 0.0f;
                float sample_z = valid
                    ? center_z + global_z * spacing_z
                    : 0.0f;

                bool refine = valid;
                if (valid && use_vector_screen) {
                    int vector_x = __float2int_rd(sample_x * lac6 + oa6);
                    int vector_z = __float2int_rd(sample_z * lac6 + oc6);
                    refine = vector_cell_enabled(
                        vector_refine, vector_x, vector_z);
                }

                float continentalness = 0.0f;
                if (refine) {
#if TIERED_USE_WARP_PERM
                    keep_perm_words(perm6.x, perm6.y, perm15.x, perm15.y);
                    continentalness = amp6 * perlin_warp(
                        perm6.x, perm6.y, oa6, oc6, h26, d26, t26,
                        sample_x * lac6, sample_z * lac6);
                    if (!o6_only) {
                        continentalness += amp15 * perlin_warp(
                            perm15.x, perm15.y, oa15, oc15,
                            h215, d215, t215,
                            sample_x * lac15 * frequency_ratio,
                            sample_z * lac15 * frequency_ratio);
                    }
#else
#if TIERED_USE_PREFIX_LUT
                    continentalness = amp6 * perlin_shared_prefix(
                        shared_perm6, lane, prefix6, oa6, oc6, d26, t26,
                        sample_x * lac6, sample_z * lac6);
                    if (!o6_only) {
                        continentalness += amp15 * perlin_shared_prefix(
                            shared_perm15, lane, prefix15, oa15, oc15,
                            d215, t215,
                            sample_x * lac15 * frequency_ratio,
                            sample_z * lac15 * frequency_ratio);
                    }
#else
                    continentalness = amp6 * perlin_shared(
                        shared_perm6, lane, oa6, oc6, h26, d26, t26,
                        sample_x * lac6, sample_z * lac6);
                    if (!o6_only) {
                        continentalness += amp15 * perlin_shared(
                            shared_perm15, lane, oa15, oc15,
                            h215, d215, t215,
                            sample_x * lac15 * frequency_ratio,
                            sample_z * lac15 * frequency_ratio);
                    }
#endif
#endif
                }

                bool low = refine && continentalness * ct_amp < threshold;
                labels_current[cell_index] = low ? cell_index : INVALID_LABEL;
            }
            __syncthreads();

            for (int round = 0; round < minimum_connected_cells; round++) {
                #pragma unroll
                for (int wave = 0; wave < WAVES; wave++) {
                    int cell_index = tid + wave * THREADS;
                    int cell_x = cell_index & (TILE - 1);
                    int cell_z = cell_index / TILE;
                    int global_z = tile_base_z + cell_z;
                    int current_label = labels_current[cell_index];
                    int minimum_label = current_label;
                    if (current_label != INVALID_LABEL) {
                        if (cell_x > 0) {
                            int neighbor = labels_current[cell_index - 1];
                            if (neighbor >= 0 && neighbor < minimum_label)
                                minimum_label = neighbor;
                        }
                        if (cell_x + 1 < TILE) {
                            int neighbor = labels_current[cell_index + 1];
                            if (neighbor >= 0 && neighbor < minimum_label)
                                minimum_label = neighbor;
                        }
                        if (cell_z > 0) {
                            int neighbor = labels_current[cell_index - TILE];
                            if (neighbor >= 0 && neighbor < minimum_label)
                                minimum_label = neighbor;
                            int diagonal_x = cell_x
                                + ((global_z & 1) ? 1 : -1);
                            if (diagonal_x >= 0 && diagonal_x < TILE) {
                                neighbor = labels_current[
                                    cell_index - TILE + diagonal_x - cell_x];
                                if (neighbor >= 0 && neighbor < minimum_label)
                                    minimum_label = neighbor;
                            }
                        }
                        if (cell_z + 1 < TILE) {
                            int neighbor = labels_current[cell_index + TILE];
                            if (neighbor >= 0 && neighbor < minimum_label)
                                minimum_label = neighbor;
                            int diagonal_x = cell_x
                                + ((global_z & 1) ? 1 : -1);
                            if (diagonal_x >= 0 && diagonal_x < TILE) {
                                neighbor = labels_current[
                                    cell_index + TILE + diagonal_x - cell_x];
                                if (neighbor >= 0 && neighbor < minimum_label)
                                    minimum_label = neighbor;
                            }
                        }
                    }
                    labels_next[cell_index] = minimum_label;
                }
                __syncthreads();
                int *swap = labels_current;
                labels_current = labels_next;
                labels_next = swap;
            }

            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                labels_next[cell_index] = 0;
            }
            __syncthreads();

            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                int label = labels_current[cell_index];
                if (label >= 0)
                    atomicAdd(&labels_next[label], 1);
            }
            __syncthreads();

            #pragma unroll
            for (int wave = 0; wave < WAVES; wave++) {
                int cell_index = tid + wave * THREADS;
                int cell_x;
                int cell_z;
                cell_x = cell_index & (TILE - 1);
                cell_z = cell_index / TILE;
                int global_x = tile_base_x + cell_x;
                int global_z = tile_base_z + cell_z;
                bool valid = global_x >= 0 && global_x < G
                    && global_z >= 0 && global_z < G;
                bool interior = global_x >= interior_begin_x
                    && global_x < interior_end_x
                    && global_z >= interior_begin_z
                    && global_z < interior_end_z;
                int label = labels_current[cell_index];
                int component_size = label >= 0
                    ? labels_next[label] & 0x7FFFFFFF : 0;
                bool candidate = valid && interior && label >= 0
                    && component_size >= minimum_connected_cells;
                bool emit = false;
                if (candidate) {
                    unsigned int old = atomicOr(
                        reinterpret_cast<unsigned int*>(&labels_next[label]),
                        0x80000000u);
                    emit = (old & 0x80000000u) == 0;
                }
                uint32_t candidate_mask = __ballot_sync(FULL_MASK, emit);
                int output_index = reserve_warp_hit(
                    candidate_mask, lane, &seed_hit_count,
                    hit_capacity, hit_count);
                if (output_index >= 0) {
                    write_component_hit(
                        output_index, seed, label, component_size,
                        tile_base_x, tile_base_z,
                        center_x, center_z,
                        spacing_x, spacing_z, half_spacing_x,
                        labels_current, hits);
                }
            }
            __syncthreads();
            if (seed_hit_count >= MAX_HITS) return;
        }
    }
}

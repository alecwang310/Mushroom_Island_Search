/* coarse_verify_kernel.cu — fixed R=2 six-octave GPU area screen. */
#include <cuda_runtime.h>
#include <stdint.h>
#include "../engine/continentalness.h"

#define VERIFY_THREADS 128
#define VERIFY_RADIUS 2
#define VERIFY_ROOT_RADIUS 2
#define VERIFY_GRID_RADIUS 4
#define VERIFY_GRID_SIDE (VERIFY_GRID_RADIUS * 2 + 1)
#define VERIFY_GRID_CELLS (VERIFY_GRID_SIDE * VERIFY_GRID_SIDE)
#define VERIFY_GROUPED_MAX_THREADS 256
#define VERIFY_GROUPED_MAX_WARPS (VERIFY_GROUPED_MAX_THREADS / 32)
#define VERIFY_OCTAVE_FREQUENCY_RATIO 1.0181268882172204f
#define VERIFY_THRESHOLD -1.05f
#define VERIFY_PERIODIC_O6_FALLBACK_BAND 0.01f

static_assert(VERIFY_ROOT_RADIUS + VERIFY_RADIUS <= VERIFY_GRID_RADIUS,
              "R=2 verifier grid is too small for the root spacing");

__device__ __forceinline__ float verify_fade(float value) {
    return value * value * value
        * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float verify_grad_dot(
    uint32_t hash, float x, float y, float z)
{
    uint32_t gradient = hash & 15u;
    float first = gradient < 8u ? x : y;
    float second = gradient < 4u
        ? y
        : ((gradient == 12u || gradient == 14u) ? x : z);
    uint32_t first_bits = __float_as_uint(first)
        ^ ((gradient & 1u) * 0x80000000u);
    uint32_t second_bits = __float_as_uint(second)
        ^ (((gradient >> 1) & 1u) * 0x80000000u);
    return __uint_as_float(first_bits) + __uint_as_float(second_bits);
}

__device__ __forceinline__ uint32_t verify_perm_pair(
    const uint32_t *perm, uint32_t index)
{
    return perm[index & 0xFFu];
}

__device__ __forceinline__ float verify_perlin(
    const uint32_t *perm,
    float offset_x, float offset_z,
    uint32_t cached_h2, float cached_d2, float cached_t2,
    float sample_x, float sample_z)
{
    float shifted_x = sample_x + offset_x;
    float shifted_z = sample_z + offset_z;
    int cell_x = __float2int_rd(shifted_x);
    int cell_z = __float2int_rd(shifted_z);
    float local_x = shifted_x - (float)cell_x;
    float local_z = shifted_z - (float)cell_z;
    uint32_t hash_x = (uint32_t)cell_x & 0xFFu;
    uint32_t hash_z = (uint32_t)cell_z & 0xFFu;
    float fade_x = verify_fade(local_x);
    float fade_z = verify_fade(local_z);

    uint32_t pair = verify_perm_pair(perm, hash_x);
    uint32_t value_a = (pair & 0xFFu) + cached_h2;
    uint32_t value_b = (pair >> 8) + cached_h2;
    pair = verify_perm_pair(perm, value_a);
    uint32_t value_2a = (pair & 0xFFu) + hash_z;
    uint32_t value_2b = (pair >> 8) + hash_z;
    pair = verify_perm_pair(perm, value_b);
    uint32_t value_3a = (pair & 0xFFu) + hash_z;
    uint32_t value_3b = (pair >> 8) + hash_z;
    pair = verify_perm_pair(perm, value_2a);
    uint32_t value_4a = pair & 0xFFu;
    uint32_t value_4b = pair >> 8;
    pair = verify_perm_pair(perm, value_2b);
    uint32_t value_5a = pair & 0xFFu;
    uint32_t value_5b = pair >> 8;
    pair = verify_perm_pair(perm, value_3a);
    uint32_t value_6a = pair & 0xFFu;
    uint32_t value_6b = pair >> 8;
    pair = verify_perm_pair(perm, value_3b);
    uint32_t value_7a = pair & 0xFFu;
    uint32_t value_7b = pair >> 8;

    float level_1 = verify_grad_dot(value_4a,
        local_x, cached_d2, local_z);
    float level_5 = verify_grad_dot(value_4b,
        local_x, cached_d2, local_z - 1.0f);
    float level_2 = verify_grad_dot(value_6a,
        local_x - 1.0f, cached_d2, local_z);
    float level_6 = verify_grad_dot(value_6b,
        local_x - 1.0f, cached_d2, local_z - 1.0f);
    float level_3 = verify_grad_dot(value_5a,
        local_x, cached_d2 - 1.0f, local_z);
    float level_7 = verify_grad_dot(value_5b,
        local_x, cached_d2 - 1.0f, local_z - 1.0f);
    float level_4 = verify_grad_dot(value_7a,
        local_x - 1.0f, cached_d2 - 1.0f, local_z);
    float level_8 = verify_grad_dot(value_7b,
        local_x - 1.0f, cached_d2 - 1.0f, local_z - 1.0f);

    level_1 = fmaf(fade_x, level_2 - level_1, level_1);
    level_3 = fmaf(fade_x, level_4 - level_3, level_3);
    level_5 = fmaf(fade_x, level_6 - level_5, level_5);
    level_7 = fmaf(fade_x, level_8 - level_7, level_7);
    level_1 = fmaf(cached_t2, level_3 - level_1, level_1);
    level_5 = fmaf(cached_t2, level_7 - level_5, level_5);
    return fmaf(fade_z, level_5 - level_1, level_1);
}

__device__ __forceinline__ int verify_row_bit(int row_delta) {
    int bit = row_delta % 2;
    return bit < 0 ? bit + 2 : bit;
}

__device__ __forceinline__ int verify_lattice_x(
    int step, int row_parity, int axial_x, int row_delta)
{
    int half_step = step / 2;
    int center_stagger = row_parity ? half_step : 0;
    int target_parity = row_parity ^ verify_row_bit(row_delta);
    int target_stagger = target_parity ? half_step : 0;
    return axial_x * step + target_stagger - center_stagger;
}

__device__ __forceinline__ int verify_hex_distance(
    int axial_x, int row_delta)
{
    int third = -axial_x - row_delta;
    int distance = abs(axial_x);
    if (abs(row_delta) > distance) distance = abs(row_delta);
    if (abs(third) > distance) distance = abs(third);
    return distance;
}

__device__ __forceinline__ int verify_round_even(float value) {
    return __float2int_rn(value);
}

__device__ __forceinline__ void verify_root(
    int direction, int row_parity, int step_1x, int step_2x,
    int &root_x, int &root_row)
{
    int root_radius = step_2x / step_1x;
    int half_step_2x = step_2x / 2;
    int signed_stagger = row_parity == 0
        ? half_step_2x
        : -half_step_2x;

    if (direction < 2) {
        root_x = direction == 0 ? -root_radius : root_radius;
        root_row = 0;
        return;
    }

    root_row = direction < 4 ? -root_radius : root_radius;
    int point_x = (direction == 2 || direction == 4)
        ? signed_stagger
        : -signed_stagger;
    int row_shift = verify_lattice_x(step_1x, row_parity, 0, root_row);
    root_x = verify_round_even(
        (float)(point_x - row_shift) / (float)step_1x);
}

__device__ __forceinline__ int verify_grid_index(int axial_x, int row_delta) {
    return (axial_x + VERIFY_GRID_RADIUS) * VERIFY_GRID_SIDE
        + (row_delta + VERIFY_GRID_RADIUS);
}

__device__ __forceinline__ bool verify_in_bounds(
    int axial_x, int row_delta)
{
    return axial_x >= -VERIFY_GRID_RADIUS
        && axial_x <= VERIFY_GRID_RADIUS
        && row_delta >= -VERIFY_GRID_RADIUS
        && row_delta <= VERIFY_GRID_RADIUS;
}

__device__ __forceinline__ float verify_six_octaves(
    const ContVerifyParams *params,
    const uint32_t shared_perm[CONT_VERIFY_OCTAVES][CONT_VERIFY_PERM_SIZE],
    float sample_x, float sample_z)
{
    float value = 0.0f;
    for (int octave = 0; octave < 3; octave++) {
        value += params->amplitude[octave] * verify_perlin(
            shared_perm[octave], params->offset_a[octave],
            params->offset_c[octave], params->cached_h2[octave],
            params->cached_d2[octave], params->cached_t2[octave],
            sample_x * params->lacunarity[octave],
            sample_z * params->lacunarity[octave]);
    }
    for (int octave = 3; octave < CONT_VERIFY_OCTAVES; octave++) {
        value += params->amplitude[octave] * verify_perlin(
            shared_perm[octave], params->offset_a[octave],
            params->offset_c[octave], params->cached_h2[octave],
            params->cached_d2[octave], params->cached_t2[octave],
            sample_x * params->lacunarity[octave]
                * VERIFY_OCTAVE_FREQUENCY_RATIO,
            sample_z * params->lacunarity[octave]
                * VERIFY_OCTAVE_FREQUENCY_RATIO);
    }
    return value * params->cont_dbl_amp;
}

__device__ __forceinline__ float verify_two_octaves(
    const ContTieredParams *params,
    const uint32_t shared_perm[CONT_TIERED_OCTAVES][CONT_TIERED_PERM_SIZE],
    float sample_x, float sample_z)
{
    float value = params->amplitude[0] * verify_perlin(
        shared_perm[0], params->offset_a[0], params->offset_c[0],
        params->cached_h2[0], params->cached_d2[0], params->cached_t2[0],
        sample_x * params->lacunarity[0],
        sample_z * params->lacunarity[0]);
    value += params->amplitude[1] * verify_perlin(
        shared_perm[1], params->offset_a[1], params->offset_c[1],
        params->cached_h2[1], params->cached_d2[1], params->cached_t2[1],
        sample_x * params->lacunarity[1] * VERIFY_OCTAVE_FREQUENCY_RATIO,
        sample_z * params->lacunarity[1] * VERIFY_OCTAVE_FREQUENCY_RATIO);
    return value * params->cont_dbl_amp;
}

__device__ __forceinline__ long long verify_floor_division(
    long long numerator, long long denominator)
{
    long long quotient = numerator / denominator;
    long long remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0)))
        quotient--;
    return quotient;
}

__device__ __forceinline__ long long verify_ceiling_division(
    long long numerator, long long denominator)
{
    return -verify_floor_division(-numerator, denominator);
}

extern "C" __global__ void coarse_verify_r2(
    const ContVerifyParams *params,
    const int4 *hits,
    int hit_count,
    int step_1x,
    int step_2x,
    int4 *results)
{
    int hit_index = blockIdx.x;
    if (hit_index >= hit_count) return;

    const int4 hit = hits[hit_index];
    if (step_1x <= 0 || step_2x != step_1x * 2) {
        if (threadIdx.x == 0)
            results[hit_index] = make_int4(hit.x, 0, 1, 0);
        return;
    }
    const ContVerifyParams *seed_params = params + hit.x;
    __shared__ uint32_t shared_perm[
        CONT_VERIFY_OCTAVES][CONT_VERIFY_PERM_SIZE];
    __shared__ uint8_t region[VERIFY_GRID_CELLS];
    __shared__ uint8_t low[VERIFY_GRID_CELLS];
    __shared__ uint8_t visited[VERIFY_GRID_CELLS];
    __shared__ int queue[VERIFY_GRID_CELLS];
    __shared__ int root_x[3];
    __shared__ int root_row[3];
    __shared__ int direction_count;

    int thread_id = threadIdx.x;
    for (int perm_index = thread_id;
         perm_index < CONT_VERIFY_PERM_SIZE;
         perm_index += blockDim.x) {
        for (int octave = 0; octave < CONT_VERIFY_OCTAVES; octave++) {
            uint8_t first = seed_params->perm[octave][perm_index];
            uint8_t second = seed_params->perm[octave]
                [(perm_index + 1) & 0xFF];
            shared_perm[octave][perm_index] = (uint32_t)first
                | ((uint32_t)second << 8);
        }
    }

    if (thread_id == 0) {
        int pair_mask = hit.w & 0x3F;
        direction_count = 0;
        root_x[0] = 0;
        root_row[0] = 0;
        for (int direction = 0; direction < 6; direction++) {
            if ((pair_mask & (1 << direction)) == 0) continue;
            if (direction_count >= 2) continue;
            verify_root(direction, (hit.w >> 6) & 1,
                        step_1x, step_2x,
                        root_x[direction_count + 1],
                        root_row[direction_count + 1]);
            direction_count++;
        }
    }
    __syncthreads();

    if (thread_id < VERIFY_GRID_CELLS) {
        int axial_x = thread_id / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        int row_delta = thread_id % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        bool in_region = false;
        for (int root = 0; root < 3; root++) {
            if (root < direction_count + 1
                    && verify_hex_distance(
                        axial_x - root_x[root],
                        row_delta - root_row[root]) <= VERIFY_RADIUS) {
                in_region = true;
            }
        }
        region[thread_id] = in_region ? 1 : 0;
        if (!in_region) {
            low[thread_id] = 0;
        } else {
            int row_parity = (hit.w >> 6) & 1;
            int sample_x = hit.y + verify_lattice_x(
                step_1x, row_parity, axial_x, row_delta);
            int sample_z = hit.z + __float2int_rz(
                (float)row_delta * 0.8660254037844386f * step_1x);
            float value = verify_six_octaves(
                seed_params, shared_perm,
                (float)sample_x, (float)sample_z);
            low[thread_id] = value < VERIFY_THRESHOLD ? 1 : 0;
        }
        visited[thread_id] = 0;
    }
    __syncthreads();

    if (thread_id == 0) {
        int queue_head = 0;
        int queue_tail = 0;
        int connected_count = 0;
        int region_count = 0;
        for (int index = 0; index < VERIFY_GRID_CELLS; index++) {
            region_count += region[index] != 0;
        }

        for (int root = 0; root < 3; root++) {
            if (root >= direction_count + 1) continue;
            int root_index = verify_grid_index(root_x[root], root_row[root]);
            if (!verify_in_bounds(root_x[root], root_row[root])) continue;
            if (region[root_index] && low[root_index]
                    && !visited[root_index]) {
                visited[root_index] = 1;
                queue[queue_tail++] = root_index;
            }
        }

        const int neighbor_x[6] = {-1, 1, 0, 0, -1, 1};
        const int neighbor_row[6] = {0, 0, -1, 1, 1, -1};
        while (queue_head < queue_tail) {
            int index = queue[queue_head++];
            int axial_x = index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            connected_count++;
            for (int direction = 0; direction < 6; direction++) {
                int next_x = axial_x + neighbor_x[direction];
                int next_row = row_delta + neighbor_row[direction];
                if (!verify_in_bounds(next_x, next_row)) continue;
                int next_index = verify_grid_index(next_x, next_row);
                if (region[next_index] && low[next_index]
                        && !visited[next_index]) {
                    visited[next_index] = 1;
                    queue[queue_tail++] = next_index;
                }
            }
        }

        int clipped = 0;
        for (int index = 0; index < VERIFY_GRID_CELLS; index++) {
            if (!visited[index]) continue;
            int axial_x = index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int nearest_root_distance = VERIFY_GRID_RADIUS + 1;
            for (int root = 0; root < 3; root++) {
                if (root >= direction_count + 1) continue;
                int distance = verify_hex_distance(
                    axial_x - root_x[root], row_delta - root_row[root]);
                if (distance < nearest_root_distance)
                    nearest_root_distance = distance;
            }
            if (nearest_root_distance == VERIFY_RADIUS) {
                clipped = 1;
            }
        }

        results[hit_index] = make_int4(
            hit.x, connected_count, clipped, region_count);
    }
}

extern "C" __global__ void coarse_verify_r2_2oct(
    const ContTieredParams *params,
    const int4 *hits,
    int hit_count,
    int step_1x,
    int step_2x,
    float threshold,
    int4 *results)
{
    int hit_index = blockIdx.x;
    if (hit_index >= hit_count) return;

    const int4 hit = hits[hit_index];
    if (step_1x <= 0 || step_2x != step_1x * 2) {
        if (threadIdx.x == 0)
            results[hit_index] = make_int4(hit.x, 0, 1, 0);
        return;
    }
    const ContTieredParams *seed_params = params + hit.x;
    __shared__ uint32_t shared_perm[
        CONT_TIERED_OCTAVES][CONT_TIERED_PERM_SIZE];
    __shared__ uint8_t region[VERIFY_GRID_CELLS];
    __shared__ uint8_t low[VERIFY_GRID_CELLS];
    __shared__ uint8_t visited[VERIFY_GRID_CELLS];
    __shared__ int queue[VERIFY_GRID_CELLS];
    __shared__ int root_x[3];
    __shared__ int root_row[3];
    __shared__ int direction_count;

    int thread_id = threadIdx.x;
    for (int perm_index = thread_id;
         perm_index < CONT_TIERED_PERM_SIZE;
         perm_index += blockDim.x) {
        for (int octave = 0; octave < CONT_TIERED_OCTAVES; octave++) {
            uint8_t first = seed_params->perm[octave][perm_index];
            uint8_t second = seed_params->perm[octave]
                [(perm_index + 1) & 0xFF];
            shared_perm[octave][perm_index] = (uint32_t)first
                | ((uint32_t)second << 8);
        }
    }

    if (thread_id == 0) {
        int pair_mask = hit.w & 0x3F;
        direction_count = 0;
        root_x[0] = 0;
        root_row[0] = 0;
        for (int direction = 0; direction < 6; direction++) {
            if ((pair_mask & (1 << direction)) == 0) continue;
            if (direction_count >= 2) continue;
            verify_root(direction, (hit.w >> 6) & 1,
                        step_1x, step_2x,
                        root_x[direction_count + 1],
                        root_row[direction_count + 1]);
            direction_count++;
        }
    }
    __syncthreads();

    if (thread_id < VERIFY_GRID_CELLS) {
        int axial_x = thread_id / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        int row_delta = thread_id % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        bool in_region = false;
        for (int root = 0; root < 3; root++) {
            if (root < direction_count + 1
                    && verify_hex_distance(
                        axial_x - root_x[root],
                        row_delta - root_row[root]) <= VERIFY_RADIUS) {
                in_region = true;
            }
        }
        region[thread_id] = in_region ? 1 : 0;
        if (!in_region) {
            low[thread_id] = 0;
        } else {
            int row_parity = (hit.w >> 6) & 1;
            int sample_x = hit.y + verify_lattice_x(
                step_1x, row_parity, axial_x, row_delta);
            int sample_z = hit.z + __float2int_rz(
                (float)row_delta * 0.8660254037844386f * step_1x);
            float value = verify_two_octaves(
                seed_params, shared_perm,
                (float)sample_x, (float)sample_z);
            low[thread_id] = value < threshold ? 1 : 0;
        }
        visited[thread_id] = 0;
    }
    __syncthreads();

    if (thread_id == 0) {
        int queue_head = 0;
        int queue_tail = 0;
        int connected_count = 0;
        int region_count = 0;
        for (int index = 0; index < VERIFY_GRID_CELLS; index++)
            region_count += region[index] != 0;

        for (int root = 0; root < 3; root++) {
            if (root >= direction_count + 1) continue;
            int root_index = verify_grid_index(root_x[root], root_row[root]);
            if (!verify_in_bounds(root_x[root], root_row[root])) continue;
            if (region[root_index] && low[root_index]
                    && !visited[root_index]) {
                visited[root_index] = 1;
                queue[queue_tail++] = root_index;
            }
        }

        const int neighbor_x[6] = {-1, 1, 0, 0, -1, 1};
        const int neighbor_row[6] = {0, 0, -1, 1, 1, -1};
        while (queue_head < queue_tail) {
            int index = queue[queue_head++];
            int axial_x = index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            connected_count++;
            for (int direction = 0; direction < 6; direction++) {
                int next_x = axial_x + neighbor_x[direction];
                int next_row = row_delta + neighbor_row[direction];
                if (!verify_in_bounds(next_x, next_row)) continue;
                int next_index = verify_grid_index(next_x, next_row);
                if (region[next_index] && low[next_index]
                        && !visited[next_index]) {
                    visited[next_index] = 1;
                    queue[queue_tail++] = next_index;
                }
            }
        }

        int clipped = 0;
        for (int index = 0; index < VERIFY_GRID_CELLS; index++) {
            if (!visited[index]) continue;
            int axial_x = index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int nearest_root_distance = VERIFY_GRID_RADIUS + 1;
            for (int root = 0; root < 3; root++) {
                if (root >= direction_count + 1) continue;
                int distance = verify_hex_distance(
                    axial_x - root_x[root], row_delta - root_row[root]);
                if (distance < nearest_root_distance)
                    nearest_root_distance = distance;
            }
            if (nearest_root_distance == VERIFY_RADIUS)
                clipped = 1;
        }

        results[hit_index] = make_int4(
            hit.x, connected_count, clipped, region_count);
    }
}

extern "C" __global__ void coarse_verify_r2_2oct_grouped(
    const ContTieredParams *params,
    const int4 *raw_hits,
    int raw_hit_count,
    int translation_period,
    int world_border,
    int step_1x,
    int step_2x,
    float threshold,
    int minimum_connected_cells,
    int output_capacity,
    int *output_count,
    int4 *output_hits,
    unsigned long long *debug_stats)
{
    int raw_hit_index = blockIdx.x;
    if (raw_hit_index >= raw_hit_count) return;
    if (step_1x <= 0 || step_2x != step_1x * 2) return;

    const int4 hit = raw_hits[raw_hit_index];
    const ContTieredParams *seed_params = params + hit.x;
    int thread_id = threadIdx.x;
    int lane = thread_id & 31;
    int warp = thread_id >> 5;
    int warp_count = blockDim.x >> 5;

    __shared__ uint32_t shared_perm[
        CONT_TIERED_OCTAVES][CONT_TIERED_PERM_SIZE];
    __shared__ uint8_t region_cells[VERIFY_GRID_CELLS];
    __shared__ int compact_index[VERIFY_GRID_CELLS];
    __shared__ unsigned long long neighbor_mask[VERIFY_GRID_CELLS];
    __shared__ float o6_value[VERIFY_GRID_CELLS];
    __shared__ int root_x[3];
    __shared__ int root_row[3];
    __shared__ int direction_count;
    __shared__ int region_count;
    __shared__ unsigned long long root_mask;
    __shared__ unsigned long long boundary_mask;
    __shared__ unsigned int warp_clipped[VERIFY_GROUPED_MAX_WARPS];
    __shared__ unsigned int warp_max_connected[VERIFY_GROUPED_MAX_WARPS];

    for (int perm_index = thread_id;
         perm_index < CONT_TIERED_PERM_SIZE;
         perm_index += blockDim.x) {
        for (int octave = 0; octave < CONT_TIERED_OCTAVES; octave++) {
            uint8_t first = seed_params->perm[octave][perm_index];
            uint8_t second = seed_params->perm[octave]
                [(perm_index + 1) & 0xFF];
            shared_perm[octave][perm_index] = (uint32_t)first
                | ((uint32_t)second << 8);
        }
    }

    if (thread_id == 0) {
        int pair_mask = hit.w & 0x3F;
        direction_count = 0;
        root_x[0] = 0;
        root_row[0] = 0;
        for (int direction = 0; direction < 6; direction++) {
            if ((pair_mask & (1 << direction)) == 0) continue;
            if (direction_count >= 2) continue;
            verify_root(direction, (hit.w >> 6) & 1,
                        step_1x, step_2x,
                        root_x[direction_count + 1],
                        root_row[direction_count + 1]);
            direction_count++;
        }
        region_count = 0;
        root_mask = 0;
        boundary_mask = 0;
        for (int cell_index = 0;
             cell_index < VERIFY_GRID_CELLS; cell_index++)
            compact_index[cell_index] = -1;
        for (int cell_index = 0;
             cell_index < VERIFY_GRID_CELLS; cell_index++) {
            int axial_x = cell_index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = cell_index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            bool in_region = false;
            for (int root = 0; root < direction_count + 1; root++) {
                if (verify_hex_distance(
                        axial_x - root_x[root],
                        row_delta - root_row[root]) <= VERIFY_RADIUS) {
                    in_region = true;
                }
            }
            if (in_region) {
                compact_index[cell_index] = region_count;
                region_cells[region_count++] = (uint8_t)cell_index;
            }
        }
        const int neighbor_x[6] = {-1, 1, 0, 0, -1, 1};
        const int neighbor_row[6] = {0, 0, -1, 1, 1, -1};
        for (int region_slot = 0;
             region_slot < region_count; region_slot++) {
            int cell_index = region_cells[region_slot];
            int axial_x = cell_index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = cell_index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            unsigned long long adjacent = 0;
            for (int direction = 0; direction < 6; direction++) {
                int next_x = axial_x + neighbor_x[direction];
                int next_row = row_delta + neighbor_row[direction];
                if (!verify_in_bounds(next_x, next_row)) continue;
                int next_index = verify_grid_index(next_x, next_row);
                int next_slot = compact_index[next_index];
                if (next_slot >= 0)
                    adjacent |= 1ull << next_slot;
            }
            neighbor_mask[region_slot] = adjacent;

            int nearest_root_distance = VERIFY_GRID_RADIUS + 1;
            for (int root = 0; root < direction_count + 1; root++) {
                int distance = verify_hex_distance(
                    axial_x - root_x[root], row_delta - root_row[root]);
                if (distance < nearest_root_distance)
                    nearest_root_distance = distance;
            }
            if (nearest_root_distance == VERIFY_RADIUS)
                boundary_mask |= 1ull << region_slot;
        }
        for (int root = 0; root < direction_count + 1; root++) {
            int index = verify_grid_index(root_x[root], root_row[root]);
            int slot = compact_index[index];
            if (slot >= 0)
                root_mask |= 1ull << slot;
        }
    }
    __syncthreads();

    if (thread_id < region_count) {
        int cell_index = region_cells[thread_id];
        int axial_x = cell_index / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        int row_delta = cell_index % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
        int row_parity = (hit.w >> 6) & 1;
        int sample_x = hit.y + verify_lattice_x(
            step_1x, row_parity, axial_x, row_delta);
        int sample_z = hit.z + __float2int_rz(
            (float)row_delta * 0.8660254037844386f * step_1x);
        o6_value[thread_id] = seed_params->amplitude[0] * verify_perlin(
            shared_perm[0], seed_params->offset_a[0],
            seed_params->offset_c[0], seed_params->cached_h2[0],
            seed_params->cached_d2[0], seed_params->cached_t2[0],
            (float)sample_x * seed_params->lacunarity[0],
            (float)sample_z * seed_params->lacunarity[0]);
    }
    __syncthreads();

    int first_slot = lane;
    int second_slot = lane + 32;
    int first_cell = first_slot < region_count
        ? region_cells[first_slot] : 0;
    int second_cell = second_slot < region_count
        ? region_cells[second_slot] : 0;
    float first_o6 = first_slot < region_count
        ? o6_value[first_slot] : 0.0f;
    float second_o6 = second_slot < region_count
        ? o6_value[second_slot] : 0.0f;
    float o15_amplitude = seed_params->amplitude[1];
    float o15_scale = seed_params->lacunarity[1]
        * VERIFY_OCTAVE_FREQUENCY_RATIO;
    float continentalness_scale = seed_params->cont_dbl_amp;

    long long min_dx = verify_ceiling_division(
        -(long long)world_border - hit.y, translation_period);
    long long max_dx = verify_floor_division(
        (long long)world_border - hit.y, translation_period);
    long long min_dz = verify_ceiling_division(
        -(long long)world_border - hit.z, translation_period);
    long long max_dz = verify_floor_division(
        (long long)world_border - hit.z, translation_period);
    int count_x = (int)(max_dx - min_dx + 1);
    int count_z = (int)(max_dz - min_dz + 1);
    int translation_count = count_x > 0 && count_z > 0
        ? count_x * count_z : 0;

    unsigned int local_clipped = 0;
    unsigned int local_max_connected = 0;
    for (int translation_index = warp;
         translation_index < translation_count;
         translation_index += warp_count) {
        int dx_index = translation_index / count_z;
        int dz_index = translation_index - dx_index * count_z;
        int translated_x = (int)((long long)hit.y
            + (min_dx + dx_index) * translation_period);
        int translated_z = (int)((long long)hit.z
            + (min_dz + dz_index) * translation_period);

        bool first_low = false;
        if (first_slot < region_count) {
            int axial_x = first_cell / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = first_cell % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_parity = (hit.w >> 6) & 1;
            int sample_x = translated_x + verify_lattice_x(
                step_1x, row_parity, axial_x, row_delta);
            int sample_z = translated_z + __float2int_rz(
                (float)row_delta * 0.8660254037844386f * step_1x);
            float o15 = o15_amplitude * verify_perlin(
                shared_perm[1], seed_params->offset_a[1],
                seed_params->offset_c[1], seed_params->cached_h2[1],
                seed_params->cached_d2[1], seed_params->cached_t2[1],
                (float)sample_x * o15_scale,
                (float)sample_z * o15_scale);
            float value = (first_o6 + o15) * continentalness_scale;
            if (fabsf(value - threshold)
                    < VERIFY_PERIODIC_O6_FALLBACK_BAND) {
                float exact_o6 = seed_params->amplitude[0] * verify_perlin(
                    shared_perm[0], seed_params->offset_a[0],
                    seed_params->offset_c[0], seed_params->cached_h2[0],
                    seed_params->cached_d2[0], seed_params->cached_t2[0],
                    (float)sample_x * seed_params->lacunarity[0],
                    (float)sample_z * seed_params->lacunarity[0]);
                value = (exact_o6 + o15) * continentalness_scale;
            }
            first_low = value < threshold;
        }
        uint32_t first_mask = __ballot_sync(0xFFFFFFFFu, first_low);

        bool second_low = false;
        if (second_slot < region_count) {
            int axial_x = second_cell / VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_delta = second_cell % VERIFY_GRID_SIDE - VERIFY_GRID_RADIUS;
            int row_parity = (hit.w >> 6) & 1;
            int sample_x = translated_x + verify_lattice_x(
                step_1x, row_parity, axial_x, row_delta);
            int sample_z = translated_z + __float2int_rz(
                (float)row_delta * 0.8660254037844386f * step_1x);
            float o15 = o15_amplitude * verify_perlin(
                shared_perm[1], seed_params->offset_a[1],
                seed_params->offset_c[1], seed_params->cached_h2[1],
                seed_params->cached_d2[1], seed_params->cached_t2[1],
                (float)sample_x * o15_scale,
                (float)sample_z * o15_scale);
            float value = (second_o6 + o15) * continentalness_scale;
            if (fabsf(value - threshold)
                    < VERIFY_PERIODIC_O6_FALLBACK_BAND) {
                float exact_o6 = seed_params->amplitude[0] * verify_perlin(
                    shared_perm[0], seed_params->offset_a[0],
                    seed_params->offset_c[0], seed_params->cached_h2[0],
                    seed_params->cached_d2[0], seed_params->cached_t2[0],
                    (float)sample_x * seed_params->lacunarity[0],
                    (float)sample_z * seed_params->lacunarity[0]);
                value = (exact_o6 + o15) * continentalness_scale;
            }
            second_low = value < threshold;
        }
        uint32_t second_mask = __ballot_sync(0xFFFFFFFFu, second_low);
        unsigned long long low_mask = (unsigned long long)first_mask
            | ((unsigned long long)second_mask << 32);

        unsigned long long visited = 0;
        unsigned long long frontier = low_mask & root_mask;
        while (frontier != 0) {
            visited |= frontier;
            bool first_connected = first_slot < region_count
                && (visited & (1ull << first_slot)) == 0
                && (low_mask & (1ull << first_slot)) != 0
                && (neighbor_mask[first_slot] & frontier) != 0;
            first_mask = __ballot_sync(
                0xFFFFFFFFu, first_connected);
            bool second_connected = second_slot < region_count
                && (visited & (1ull << second_slot)) == 0
                && (low_mask & (1ull << second_slot)) != 0
                && (neighbor_mask[second_slot] & frontier) != 0;
            second_mask = __ballot_sync(
                0xFFFFFFFFu, second_connected);
            frontier = (unsigned long long)first_mask
                | ((unsigned long long)second_mask << 32);
        }

        if (lane == 0) {
            int connected_count = __popcll(visited);
            int clipped = (visited & boundary_mask) != 0;
            local_clipped += clipped;
            if ((unsigned int)connected_count > local_max_connected)
                local_max_connected = connected_count;

            if (connected_count >= minimum_connected_cells) {
                int output_index = atomicAdd(output_count, 1);
                if (output_index < output_capacity) {
                    output_hits[output_index] = make_int4(
                        hit.x, translated_x, translated_z, hit.w);
                }
            }
        }
    }

    if (lane == 0) {
        warp_clipped[warp] = local_clipped;
        warp_max_connected[warp] = local_max_connected;
    }
    __syncthreads();

    if (thread_id == 0 && debug_stats) {
        unsigned int clipped_count = 0;
        unsigned int max_connected = 0;
        for (int warp_index = 0; warp_index < warp_count; warp_index++) {
            clipped_count += warp_clipped[warp_index];
            if (warp_max_connected[warp_index] > max_connected)
                max_connected = warp_max_connected[warp_index];
        }
        atomicAdd(debug_stats + 0, (unsigned long long)translation_count);
        atomicAdd(debug_stats + 1,
                  (unsigned long long)translation_count * region_count);
        atomicAdd(debug_stats + 2, (unsigned long long)clipped_count);
        atomicMax(debug_stats + 3, (unsigned long long)max_connected);
    }
}

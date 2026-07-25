/*
 * hunt_engine.cu — GPU hex-grid tiered hunt + seed-range prefilter DLL.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll
 *        hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu
 *        ../engine/continentalness.c -I../engine -lcudart
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "../engine/continentalness.h"
}

#define THREADS             256
#define TIER_CHUNK          8192
#define INITIAL_HIT_CAP     65536

extern "C" __global__ void tiered_scan(
    const ContTieredParams *params, int num_seeds,
    int G, int step_2x,
    int hit_capacity, int *hit_count,
    int3 *hits);

extern "C" __global__ void prefilter_seeds(
    uint64_t start_seed, int n,
    uint64_t *survivors, int survivor_capacity, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes);

struct TierBuffers {
    ContTieredParams *h_params;
    ContTieredParams *d_params;
    int *d_hit_count;
    int3 *d_hits;
    int3 *h_hits;
    int seed_cap;
    int hit_cap;
} g_tier = {0};

struct PrefilterBuffers {
    uint64_t *d_survivors;
    int *d_pass_count;
    int survivor_cap;
} g_prefilter = {0};

static void ensure_seed_capacity(int n) {
    if (g_tier.seed_cap >= n) return;
    if (g_tier.h_params) free(g_tier.h_params);
    if (g_tier.d_params) cudaFree(g_tier.d_params);
    g_tier.h_params = (ContTieredParams*)malloc((size_t)n * sizeof(ContTieredParams));
    cudaMalloc(&g_tier.d_params, (size_t)n * sizeof(ContTieredParams));
    g_tier.seed_cap = n;
}

static void ensure_hit_capacity(int capacity) {
    if (g_tier.hit_cap >= capacity) return;
    if (g_tier.d_hits) cudaFree(g_tier.d_hits);
    if (g_tier.h_hits) free(g_tier.h_hits);

    cudaMalloc(&g_tier.d_hits, (size_t)capacity * sizeof(int3));
    g_tier.h_hits = (int3*)malloc((size_t)capacity * sizeof(int3));
    g_tier.hit_cap = capacity;
}

static void ensure_prefilter_capacity(int capacity) {
    if (!g_prefilter.d_pass_count)
        cudaMalloc(&g_prefilter.d_pass_count, sizeof(int));
    if (g_prefilter.survivor_cap >= capacity) return;
    if (g_prefilter.d_survivors) cudaFree(g_prefilter.d_survivors);
    cudaMalloc(&g_prefilter.d_survivors, (size_t)capacity * sizeof(uint64_t));
    g_prefilter.survivor_cap = capacity;
}

extern "C" __declspec(dllexport) int prefilter_range(
    uint64_t start_seed, int n, float lo, float hi,
    uint64_t *h_survivors, int survivor_capacity)
{
    if (n <= 0) return 0;
    if (survivor_capacity <= 0) return -n;
    ensure_prefilter_capacity(survivor_capacity);

    cudaMemset(g_prefilter.d_pass_count, 0, sizeof(int));
    prefilter_seeds<<<(n + THREADS - 1) / THREADS, THREADS>>>(
        start_seed, n, g_prefilter.d_survivors, survivor_capacity,
        g_prefilter.d_pass_count, lo, hi, 0);

    int pass_count = 0;
    cudaMemcpy(&pass_count, g_prefilter.d_pass_count,
               sizeof(int), cudaMemcpyDeviceToHost);
    if (pass_count > survivor_capacity)
        return -pass_count;
    if (pass_count > 0)
        cudaMemcpy(h_survivors, g_prefilter.d_survivors,
                   (size_t)pass_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return pass_count;
}

static int tiered_chunk(const uint64_t *seeds, int n, int step_2x, int G,
                        int hit_capacity, int64_t *hit_results)
{
    if (n <= 0) return 0;
    ensure_seed_capacity(n);
    ensure_hit_capacity(hit_capacity);
    if (!g_tier.d_hit_count)
        cudaMalloc(&g_tier.d_hit_count, sizeof(int));

    cont_batch_init_tiered(seeds, n, 0, g_tier.h_params);
    cudaMemcpy(g_tier.d_params, g_tier.h_params,
               (size_t)n * sizeof(ContTieredParams), cudaMemcpyHostToDevice);
    cudaMemset(g_tier.d_hit_count, 0, sizeof(int));

    tiered_scan<<<n, THREADS>>>(
        g_tier.d_params, n, G, step_2x,
        hit_capacity, g_tier.d_hit_count,
        g_tier.d_hits);

    int total = 0;
    cudaMemcpy(&total, g_tier.d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (total > hit_capacity)
        return -total;
    if (total == 0)
        return 0;

    cudaMemcpy(g_tier.h_hits, g_tier.d_hits,
               (size_t)total * sizeof(int3), cudaMemcpyDeviceToHost);

    for (int i = 0; i < total; i++) {
        int seed_idx = g_tier.h_hits[i].x;
        hit_results[i * 3] = (int64_t)seeds[seed_idx];
        hit_results[i * 3 + 1] = (int64_t)g_tier.h_hits[i].y;
        hit_results[i * 3 + 2] = (int64_t)g_tier.h_hits[i].z;
    }
    return total;
}

extern "C" __declspec(dllexport) int hunt_batch_tiered(
    uint64_t start_seed, int n, int step_2x, int G,
    int hit_capacity, int64_t *hit_results)
{
    uint64_t *seeds = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds[i] = start_seed + (uint64_t)i;
    int total = tiered_chunk(seeds, n, step_2x, G, hit_capacity, hit_results);
    free(seeds);
    return total;
}

extern "C" __declspec(dllexport) int tiered_scan_mem(
    const uint64_t *seeds, int n, int step_2x, int G,
    int hit_capacity, int64_t *hit_results)
{
    if (n > TIER_CHUNK) return -1;
    return tiered_chunk(seeds, n, step_2x, G, hit_capacity, hit_results);
}

extern "C" __declspec(dllexport) int hunt_batch_from_file(
    const char *seed_file, int step_2x, int G, const char *hits_file)
{
    FILE *input = fopen(seed_file, "rb");
    if (!input) return -1;
    fseek(input, 0, SEEK_END);
    long file_size = ftell(input);
    rewind(input);
    int n = (int)(file_size / 8);
    if (n <= 0 || file_size % 8 != 0) { fclose(input); return -1; }

    uint64_t *seeds = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    if (fread(seeds, 8, n, input) != (size_t)n) {
        fclose(input);
        free(seeds);
        return -1;
    }
    fclose(input);

    FILE *output = fopen(hits_file, "wb");
    if (!output) { free(seeds); return -1; }

    int hit_capacity = INITIAL_HIT_CAP;
    int64_t *chunk_hits = (int64_t*)malloc((size_t)hit_capacity * 3 * sizeof(int64_t));
    int total = 0;
    for (int offset = 0; offset < n; offset += TIER_CHUNK) {
        int size = (n - offset < TIER_CHUNK) ? (n - offset) : TIER_CHUNK;
        int hit_count = tiered_chunk(seeds + offset, size, step_2x, G,
                                     hit_capacity, chunk_hits);
        if (hit_count < 0) {
            hit_capacity = -hit_count;
            chunk_hits = (int64_t*)realloc(
                chunk_hits, (size_t)hit_capacity * 3 * sizeof(int64_t));
            hit_count = tiered_chunk(seeds + offset, size, step_2x, G,
                                     hit_capacity, chunk_hits);
        }
        if (hit_count < 0) break;
        fwrite(chunk_hits, 3 * sizeof(int64_t), hit_count, output);
        total += hit_count;
    }

    free(chunk_hits);
    free(seeds);
    fclose(output);
    return total;
}

extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g_tier.d_params) cudaFree(g_tier.d_params);
    if (g_tier.d_hit_count) cudaFree(g_tier.d_hit_count);
    if (g_tier.d_hits) cudaFree(g_tier.d_hits);
    free(g_tier.h_params);
    free(g_tier.h_hits);
    g_tier = {};

    if (g_prefilter.d_survivors) cudaFree(g_prefilter.d_survivors);
    if (g_prefilter.d_pass_count) cudaFree(g_prefilter.d_pass_count);
    g_prefilter = {};
}

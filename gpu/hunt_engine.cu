/*
 * hunt_engine.cu — GPU hex-grid tiered hunt + seed-range prefilter DLL.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll
 *        hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu
 *        ../engine/continentalness.c -I../engine -lcudart
 */
#include <cuda_runtime.h>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "../engine/continentalness.h"
}

#define THREADS             256
#define TIER_CHUNK          8192
#define INITIAL_HIT_CAP     65536
#define TIER_TIMING_COUNT   8

extern "C" __global__ void tiered_scan(
    const ContTieredParams *params, int num_seeds,
    int G, int step_2x, float threshold,
    int hit_capacity, int *hit_count,
    int4 *hits);

extern "C" __global__ void prefilter_seeds(
    uint64_t start_seed, int n,
    uint64_t *survivors, int survivor_capacity, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes);

struct TierBuffers {
    ContTieredParams *h_params;
    ContTieredParams *d_params;
    int *d_hit_count;
    int4 *d_hits;
    int4 *h_hits;
    int seed_cap;
    int hit_cap;
    cudaEvent_t phase_start;
    cudaEvent_t memset_done;
    cudaEvent_t kernel_done;
    int timing_events_ready;
} g_tier = {0};

struct TierTimings {
    double total_ms;
    double init_ms;
    double h2d_ms;
    double memset_ms;
    double kernel_ms;
    double count_d2h_ms;
    double hits_d2h_ms;
    double pack_ms;
};

using TierClock = std::chrono::steady_clock;

static double elapsed_ms(TierClock::time_point start,
                         TierClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void ensure_timing_events() {
    if (g_tier.timing_events_ready) return;
    cudaEventCreate(&g_tier.phase_start);
    cudaEventCreate(&g_tier.memset_done);
    cudaEventCreate(&g_tier.kernel_done);
    g_tier.timing_events_ready = 1;
}

static void write_timings(const TierTimings *timings,
                          double *output, int output_count) {
    if (!timings || !output || output_count < TIER_TIMING_COUNT) return;
    output[0] = timings->total_ms;
    output[1] = timings->init_ms;
    output[2] = timings->h2d_ms;
    output[3] = timings->memset_ms;
    output[4] = timings->kernel_ms;
    output[5] = timings->count_d2h_ms;
    output[6] = timings->hits_d2h_ms;
    output[7] = timings->pack_ms;
}

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

    cudaMalloc(&g_tier.d_hits, (size_t)capacity * sizeof(int4));
    g_tier.h_hits = (int4*)malloc((size_t)capacity * sizeof(int4));
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
                        float threshold, int hit_capacity, int64_t *hit_results,
                        TierTimings *timings)
{
    if (n <= 0) return 0;
    TierClock::time_point total_started;
    if (timings) {
        *timings = {};
        total_started = TierClock::now();
        ensure_timing_events();
    }
    ensure_seed_capacity(n);
    ensure_hit_capacity(hit_capacity);
    if (!g_tier.d_hit_count)
        cudaMalloc(&g_tier.d_hit_count, sizeof(int));

    auto stage_started = TierClock::now();
    cont_batch_init_tiered(seeds, n, 0, g_tier.h_params);
    if (timings)
        timings->init_ms = elapsed_ms(stage_started, TierClock::now());

    stage_started = TierClock::now();
    cudaMemcpy(g_tier.d_params, g_tier.h_params,
               (size_t)n * sizeof(ContTieredParams), cudaMemcpyHostToDevice);
    if (timings)
        timings->h2d_ms = elapsed_ms(stage_started, TierClock::now());

    if (timings)
        cudaEventRecord(g_tier.phase_start);
    cudaMemset(g_tier.d_hit_count, 0, sizeof(int));
    if (timings)
        cudaEventRecord(g_tier.memset_done);

    tiered_scan<<<n, THREADS>>>(
        g_tier.d_params, n, G, step_2x, threshold,
        hit_capacity, g_tier.d_hit_count,
        g_tier.d_hits);
    if (timings) {
        cudaEventRecord(g_tier.kernel_done);
        cudaEventSynchronize(g_tier.kernel_done);
        float memset_ms = 0.0f;
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(
            &memset_ms, g_tier.phase_start, g_tier.memset_done);
        cudaEventElapsedTime(
            &kernel_ms, g_tier.memset_done, g_tier.kernel_done);
        timings->memset_ms = memset_ms;
        timings->kernel_ms = kernel_ms;
    }

    int total = 0;
    stage_started = TierClock::now();
    cudaMemcpy(&total, g_tier.d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (timings)
        timings->count_d2h_ms = elapsed_ms(stage_started, TierClock::now());
    if (total > hit_capacity) {
        if (timings)
            timings->total_ms = elapsed_ms(total_started, TierClock::now());
        return -total;
    }
    if (total == 0) {
        if (timings)
            timings->total_ms = elapsed_ms(total_started, TierClock::now());
        return 0;
    }

    stage_started = TierClock::now();
    cudaMemcpy(g_tier.h_hits, g_tier.d_hits,
               (size_t)total * sizeof(int4), cudaMemcpyDeviceToHost);
    if (timings)
        timings->hits_d2h_ms = elapsed_ms(stage_started, TierClock::now());

    stage_started = TierClock::now();
    for (int i = 0; i < total; i++) {
        int seed_idx = g_tier.h_hits[i].x;
        hit_results[i * 4] = (int64_t)seeds[seed_idx];
        hit_results[i * 4 + 1] = (int64_t)g_tier.h_hits[i].y;
        hit_results[i * 4 + 2] = (int64_t)g_tier.h_hits[i].z;
        hit_results[i * 4 + 3] = (int64_t)g_tier.h_hits[i].w;
    }
    if (timings) {
        timings->pack_ms = elapsed_ms(stage_started, TierClock::now());
        timings->total_ms = elapsed_ms(total_started, TierClock::now());
    }
    return total;
}

extern "C" __declspec(dllexport) int hunt_batch_tiered(
    uint64_t start_seed, int n, int step_2x, int G, float threshold,
    int hit_capacity, int64_t *hit_results)
{
    uint64_t *seeds = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds[i] = start_seed + (uint64_t)i;
    int total = tiered_chunk(
        seeds, n, step_2x, G, threshold, hit_capacity, hit_results, nullptr);
    free(seeds);
    return total;
}

extern "C" __declspec(dllexport) int tiered_scan_mem(
    const uint64_t *seeds, int n, int step_2x, int G, float threshold,
    int hit_capacity, int64_t *hit_results)
{
    if (n > TIER_CHUNK) return -1;
    return tiered_chunk(
        seeds, n, step_2x, G, threshold, hit_capacity, hit_results, nullptr);
}

extern "C" __declspec(dllexport) int tiered_scan_mem_profiled(
    const uint64_t *seeds, int n, int step_2x, int G, float threshold,
    int hit_capacity, int64_t *hit_results,
    double *timing_output, int timing_count)
{
    if (n > TIER_CHUNK) return -1;
    TierTimings timings = {};
    int total = tiered_chunk(
        seeds, n, step_2x, G, threshold, hit_capacity, hit_results, &timings);
    write_timings(&timings, timing_output, timing_count);
    return total;
}

extern "C" __declspec(dllexport) int hunt_batch_from_file(
    const char *seed_file, int step_2x, int G, float threshold,
    const char *hits_file)
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
    int64_t *chunk_hits = (int64_t*)malloc((size_t)hit_capacity * 4 * sizeof(int64_t));
    int total = 0;
    for (int offset = 0; offset < n; offset += TIER_CHUNK) {
        int size = (n - offset < TIER_CHUNK) ? (n - offset) : TIER_CHUNK;
        int hit_count = tiered_chunk(seeds + offset, size, step_2x, G,
                                     threshold, hit_capacity, chunk_hits,
                                     nullptr);
        if (hit_count < 0) {
            hit_capacity = -hit_count;
            chunk_hits = (int64_t*)realloc(
                chunk_hits, (size_t)hit_capacity * 4 * sizeof(int64_t));
            hit_count = tiered_chunk(seeds + offset, size, step_2x, G,
                                     threshold, hit_capacity, chunk_hits,
                                     nullptr);
        }
        if (hit_count < 0) break;
        fwrite(chunk_hits, 4 * sizeof(int64_t), hit_count, output);
        total += hit_count;
    }

    free(chunk_hits);
    free(seeds);
    fclose(output);
    return total;
}

extern "C" __declspec(dllexport) void hunt_cleanup() {
    if (g_tier.timing_events_ready) {
        cudaEventDestroy(g_tier.phase_start);
        cudaEventDestroy(g_tier.memset_done);
        cudaEventDestroy(g_tier.kernel_done);
    }
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

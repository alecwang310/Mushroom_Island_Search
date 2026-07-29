/*
 * hunt_engine.cu — GPU hex-grid tiered hunt + seed-range prefilter DLL.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll
 *        hunt_engine.cu tiered_kernel.cu coarse_verify_kernel.cu
 *        prefilter_kernel.cu
 *        ../engine/continentalness.c -I../engine -lcudart
 */
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern "C" {
#include "../engine/continentalness.h"
}

#define THREADS             256
#define VERIFY_THREADS      128
#define TIER_CHUNK          8192
#define INITIAL_HIT_CAP     65536
#define TIER_TIMING_COUNT   9
#define COARSE_MIN_AREA     6000000LL
#define TRANSLATION_ESTIMATE_THRESHOLD -0.88f
#define TRANSLATION_GROUPED_THREADS 256

extern "C" __global__ void tiered_scan(
    const ContTieredParams *params, int num_seeds,
    int G, int step_1x, float threshold, int o6_only,
    int minimum_connected_cells,
    int hit_capacity, int *hit_count,
    int4 *hits);

extern "C" __global__ void coarse_verify_r2(
    const ContVerifyParams *params,
    const int4 *hits,
    int hit_count,
    int step_1x,
    int step_2x,
    int4 *results);

extern "C" __global__ void coarse_verify_r2_2oct(
    const ContTieredParams *params,
    const int4 *hits,
    int hit_count,
    int step_1x,
    int step_2x,
    float threshold,
    int4 *results);

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
    unsigned long long *debug_stats);

extern "C" __global__ void prefilter_seeds(
    uint64_t start_seed, int n,
    uint64_t *survivors, int survivor_capacity, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes);

struct TierBuffers {
    ContTieredParams *h_params;
    ContTieredParams *d_params;
    ContVerifyParams *h_verify_params;
    ContVerifyParams *d_verify_params;
    int *d_hit_count;
    unsigned long long *d_translation_stats;
    int4 *d_hits;
    int4 *h_hits;
    int4 *d_coarse_results;
    int4 *h_coarse_results;
    int seed_cap;
    int hit_cap;
    cudaEvent_t phase_start;
    cudaEvent_t memset_done;
    cudaEvent_t kernel_done;
    cudaEvent_t coarse_done;
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
    double coarse_kernel_ms;
};

using TierClock = std::chrono::steady_clock;

static double elapsed_ms(TierClock::time_point start,
                         TierClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static int64_t coarse_min_area() {
    const char *value = getenv("HUNT_GPU_COARSE_MIN_AREA");
    if (!value || !*value) return COARSE_MIN_AREA;
    char *end = nullptr;
    long long parsed = strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0)
        return COARSE_MIN_AREA;
    return (int64_t)parsed;
}

static int debug_translation_stats() {
    const char *value = getenv("HUNT_DEBUG_TRANSLATION_STATS");
    return value && *value && atoi(value) != 0;
}

static float translation_estimate_threshold() {
    const char *value = getenv("HUNT_TRANSLATION_ESTIMATE_THRESHOLD");
    if (!value || !*value) return TRANSLATION_ESTIMATE_THRESHOLD;
    char *end = nullptr;
    float parsed = strtof(value, &end);
    if (end == value || *end != '\0')
        return TRANSLATION_ESTIMATE_THRESHOLD;
    return parsed;
}

static int translation_grouped() {
    const char *value = getenv("HUNT_TRANSLATION_GROUPED");
    return !value || !*value || atoi(value) != 0;
}

static int translation_grouped_threads() {
    const char *value = getenv("HUNT_TRANSLATION_GROUPED_THREADS");
    int threads = value && *value ? atoi(value) : TRANSLATION_GROUPED_THREADS;
    if (threads <= 128) return 128;
    if (threads <= 256) return 256;
    return 512;
}

static int cap_o6_grid_size(int grid_size, int step_2x,
                            int o6_period)
{
    if (grid_size <= 0) return 1;
    if (step_2x <= 0 || o6_period <= 0) return grid_size;

    /* Keep the sampled span strictly inside one open O6 period. */
    int max_grid = ((o6_period - 1) / step_2x) + 1;
    if (max_grid < 1) max_grid = 1;
    return grid_size < max_grid ? grid_size : max_grid;
}

static void ensure_timing_events() {
    if (g_tier.timing_events_ready) return;
    cudaEventCreate(&g_tier.phase_start);
    cudaEventCreate(&g_tier.memset_done);
    cudaEventCreate(&g_tier.kernel_done);
    cudaEventCreate(&g_tier.coarse_done);
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
    output[8] = timings->coarse_kernel_ms;
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
    if (g_tier.h_verify_params) free(g_tier.h_verify_params);
    if (g_tier.d_verify_params) cudaFree(g_tier.d_verify_params);
    g_tier.h_params = (ContTieredParams*)malloc((size_t)n * sizeof(ContTieredParams));
    cudaMalloc(&g_tier.d_params, (size_t)n * sizeof(ContTieredParams));
    g_tier.h_verify_params = (ContVerifyParams*)malloc(
        (size_t)n * sizeof(ContVerifyParams));
    cudaMalloc(&g_tier.d_verify_params,
               (size_t)n * sizeof(ContVerifyParams));
    g_tier.seed_cap = n;
}

static void ensure_hit_capacity(int capacity) {
    if (g_tier.hit_cap >= capacity) return;
    if (g_tier.d_hits) cudaFree(g_tier.d_hits);
    if (g_tier.h_hits) free(g_tier.h_hits);
    if (g_tier.d_coarse_results) cudaFree(g_tier.d_coarse_results);
    if (g_tier.h_coarse_results) free(g_tier.h_coarse_results);

    cudaMalloc(&g_tier.d_hits, (size_t)capacity * sizeof(int4));
    g_tier.h_hits = (int4*)malloc((size_t)capacity * sizeof(int4));
    cudaMalloc(&g_tier.d_coarse_results, (size_t)capacity * sizeof(int4));
    g_tier.h_coarse_results = (int4*)malloc(
        (size_t)capacity * sizeof(int4));
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

static int64_t floor_division(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0)))
        --quotient;
    return quotient;
}

static int64_t ceiling_division(int64_t numerator, int64_t denominator) {
    return -floor_division(-numerator, denominator);
}

static int translation_min_index(int coordinate, int period, int border) {
    return (int)ceiling_division(
        -static_cast<int64_t>(border) - coordinate, period);
}

static int translation_max_index(int coordinate, int period, int border) {
    return (int)floor_division(
        static_cast<int64_t>(border) - coordinate, period);
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
    if (step_2x <= 0 || (step_2x & 1) != 0) return -1;
    const int step_1x = step_2x / 2;
    const double o6_cell_area = (double)step_1x * step_1x
        * 0.8660254037844386 * 16.0;
    int o6_minimum_connected_cells = (int)ceil(
        (double)coarse_min_area() / o6_cell_area);
    if (o6_minimum_connected_cells < 1)
        o6_minimum_connected_cells = 1;
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
    cont_batch_init_6oct(seeds, n, 0, g_tier.h_verify_params);
    if (timings)
        timings->init_ms = elapsed_ms(stage_started, TierClock::now());

    stage_started = TierClock::now();
    cudaMemcpy(g_tier.d_params, g_tier.h_params,
               (size_t)n * sizeof(ContTieredParams), cudaMemcpyHostToDevice);
    cudaMemcpy(g_tier.d_verify_params, g_tier.h_verify_params,
               (size_t)n * sizeof(ContVerifyParams), cudaMemcpyHostToDevice);
    if (timings)
        timings->h2d_ms = elapsed_ms(stage_started, TierClock::now());

    if (timings)
        cudaEventRecord(g_tier.phase_start);
    cudaMemset(g_tier.d_hit_count, 0, sizeof(int));
    if (timings)
        cudaEventRecord(g_tier.memset_done);

    tiered_scan<<<n, THREADS>>>(
        g_tier.d_params, n, G, step_1x, threshold,
        1, o6_minimum_connected_cells,
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

    coarse_verify_r2<<<total, VERIFY_THREADS>>>(
        g_tier.d_verify_params, g_tier.d_hits, total,
        step_1x, step_2x, g_tier.d_coarse_results);
    if (timings) {
        cudaEventRecord(g_tier.coarse_done);
        cudaEventSynchronize(g_tier.coarse_done);
        float coarse_ms = 0.0f;
        cudaEventElapsedTime(
            &coarse_ms, g_tier.kernel_done, g_tier.coarse_done);
        timings->coarse_kernel_ms = coarse_ms;
    }

    stage_started = TierClock::now();
    cudaMemcpy(g_tier.h_hits, g_tier.d_hits,
               (size_t)total * sizeof(int4), cudaMemcpyDeviceToHost);
    cudaMemcpy(g_tier.h_coarse_results, g_tier.d_coarse_results,
               (size_t)total * sizeof(int4), cudaMemcpyDeviceToHost);
    if (timings)
        timings->hits_d2h_ms = elapsed_ms(stage_started, TierClock::now());

    stage_started = TierClock::now();
    int output_count = 0;
    const double coarse_cell_area = (double)step_1x * step_1x
        * 0.8660254037844386 * 16.0;
    const int64_t coarse_target = coarse_min_area();
    for (int i = 0; i < total; i++) {
        const int4 coarse = g_tier.h_coarse_results[i];
        int64_t coarse_area = (int64_t)(
            (double)coarse.y * coarse_cell_area);
        if (coarse.z == 0 && coarse_area < coarse_target)
            continue;

        int seed_idx = g_tier.h_hits[i].x;
        hit_results[output_count * 4] = (int64_t)seeds[seed_idx];
        hit_results[output_count * 4 + 1] = (int64_t)g_tier.h_hits[i].y;
        hit_results[output_count * 4 + 2] = (int64_t)g_tier.h_hits[i].z;
        hit_results[output_count * 4 + 3] = (int64_t)g_tier.h_hits[i].w;
        output_count++;
    }
    if (timings) {
        timings->pack_ms = elapsed_ms(stage_started, TierClock::now());
        timings->total_ms = elapsed_ms(total_started, TierClock::now());
    }
    return output_count;
}

static int tiered_translation_chunk(
    const uint64_t *seeds, int n, int scan_step_2x, int G,
    float o6_threshold, int translation_period, int world_border,
    int estimate_step_1x, int estimate_step_2x,
    int64_t estimate_min_area, int hit_capacity, int64_t *hit_results)
{
    if (n <= 0) return 0;
    if (hit_capacity <= 0 || translation_period <= 0
            || scan_step_2x <= 0 || (scan_step_2x & 1) != 0
            || world_border < 0 || estimate_step_1x <= 0
            || estimate_step_2x != estimate_step_1x * 2)
        return -1;
    const int scan_step_1x = scan_step_2x / 2;
    const int debug_stats = debug_translation_stats();
    const int grouped = translation_grouped();
    const int grouped_threads = translation_grouped_threads();
    const float estimate_threshold = translation_estimate_threshold();
    const double cell_area = (double)estimate_step_1x * estimate_step_1x
        * 0.8660254037844386 * 16.0;
    int minimum_connected_cells = (int)ceil(
        (double)estimate_min_area / cell_area);
    if (minimum_connected_cells < 1)
        minimum_connected_cells = 1;
    const double o6_cell_area = (double)scan_step_1x * scan_step_1x
        * 0.8660254037844386 * 16.0;
    int o6_minimum_connected_cells = (int)ceil(
        (double)coarse_min_area() / o6_cell_area);
    if (o6_minimum_connected_cells < 1)
        o6_minimum_connected_cells = 1;
    float scan_kernel_ms = 0.0f;
    float estimate_kernel_ms = 0.0f;
    double init_ms = 0.0;
    double params_h2d_ms = 0.0;
    double scan_setup_ms = 0.0;
    double raw_count_d2h_ms = 0.0;
    double raw_hits_d2h_ms = 0.0;
    double translation_count_ms = 0.0;
    double translation_fill_ms = 0.0;
    double translation_h2d_ms = 0.0;
    double estimate_setup_ms = 0.0;
    double result_count_d2h_ms = 0.0;
    double result_d2h_ms = 0.0;
    double pack_ms = 0.0;
    auto total_started = TierClock::now();
    auto stage_started = total_started;
    G = cap_o6_grid_size(G, scan_step_1x, translation_period);
    ensure_seed_capacity(n);
    ensure_hit_capacity(hit_capacity);
    if (!g_tier.d_hit_count)
        cudaMalloc(&g_tier.d_hit_count, sizeof(int));
    if (debug_stats && !g_tier.d_translation_stats)
        cudaMalloc(&g_tier.d_translation_stats,
                   6 * sizeof(unsigned long long));
    if (debug_stats)
        ensure_timing_events();

    stage_started = TierClock::now();
    cont_batch_init_tiered(seeds, n, 0, g_tier.h_params);
    init_ms = elapsed_ms(stage_started, TierClock::now());
    stage_started = TierClock::now();
    cudaMemcpy(g_tier.d_params, g_tier.h_params,
               (size_t)n * sizeof(ContTieredParams), cudaMemcpyHostToDevice);
    params_h2d_ms = elapsed_ms(stage_started, TierClock::now());

    stage_started = TierClock::now();
    cudaMemset(g_tier.d_hit_count, 0, sizeof(int));
    scan_setup_ms = elapsed_ms(stage_started, TierClock::now());
    if (debug_stats)
        cudaEventRecord(g_tier.phase_start);
    tiered_scan<<<n, THREADS>>>(
        g_tier.d_params, n, G, scan_step_1x, o6_threshold,
        1, o6_minimum_connected_cells,
        hit_capacity, g_tier.d_hit_count, g_tier.d_hits);
    if (debug_stats) {
        cudaEventRecord(g_tier.kernel_done);
        cudaEventSynchronize(g_tier.kernel_done);
        cudaEventElapsedTime(
            &scan_kernel_ms, g_tier.phase_start, g_tier.kernel_done);
    }

    int raw_count = 0;
    stage_started = TierClock::now();
    cudaMemcpy(&raw_count, g_tier.d_hit_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    raw_count_d2h_ms = elapsed_ms(stage_started, TierClock::now());
    if (raw_count > hit_capacity)
        return -raw_count;
    if (raw_count == 0)
        return 0;

    int output_count = 0;
    int clipped_count = 0;
    int64_t max_estimated_area = 0;
    int64_t region_points = 0;
    int translated_count = 0;
    unsigned long long base_o6_calls = 0;
    unsigned long long fallback_o6_calls = 0;

    if (grouped) {
        stage_started = TierClock::now();
        cudaMemset(g_tier.d_hit_count, 0, sizeof(int));
        if (debug_stats)
            cudaMemset(g_tier.d_translation_stats, 0,
                       6 * sizeof(unsigned long long));
        estimate_setup_ms = elapsed_ms(stage_started, TierClock::now());
        if (debug_stats)
            cudaEventRecord(g_tier.phase_start);
        coarse_verify_r2_2oct_grouped<<<raw_count, grouped_threads>>>(
            g_tier.d_params, g_tier.d_hits, raw_count,
            translation_period, world_border,
            estimate_step_1x, estimate_step_2x, estimate_threshold,
            minimum_connected_cells, hit_capacity,
            g_tier.d_hit_count, g_tier.d_coarse_results,
            debug_stats ? g_tier.d_translation_stats : nullptr);
        if (debug_stats) {
            cudaEventRecord(g_tier.coarse_done);
            cudaEventSynchronize(g_tier.coarse_done);
            cudaEventElapsedTime(
                &estimate_kernel_ms, g_tier.phase_start, g_tier.coarse_done);
        }

        stage_started = TierClock::now();
        cudaMemcpy(&output_count, g_tier.d_hit_count, sizeof(int),
                   cudaMemcpyDeviceToHost);
        result_count_d2h_ms = elapsed_ms(stage_started, TierClock::now());
        if (debug_stats) {
            unsigned long long stats[6] = {};
            cudaMemcpy(stats, g_tier.d_translation_stats, sizeof(stats),
                       cudaMemcpyDeviceToHost);
            translated_count = stats[0] > 0x7FFFFFFFULL
                ? 0x7FFFFFFF : (int)stats[0];
            region_points = stats[1] > 0x7FFFFFFFFFFFFFFFULL
                ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)stats[1];
            clipped_count = stats[2] > 0x7FFFFFFFULL
                ? 0x7FFFFFFF : (int)stats[2];
            max_estimated_area = (int64_t)(stats[3] * cell_area);
            base_o6_calls = stats[4];
            fallback_o6_calls = stats[5];
        }
        if (output_count > hit_capacity)
            return -output_count;
        if (output_count > 0) {
            stage_started = TierClock::now();
            cudaMemcpy(g_tier.h_coarse_results, g_tier.d_coarse_results,
                       (size_t)output_count * sizeof(int4),
                       cudaMemcpyDeviceToHost);
            result_d2h_ms = elapsed_ms(stage_started, TierClock::now());
        }

        stage_started = TierClock::now();
        for (int i = 0; i < output_count; i++) {
            const int4 hit = g_tier.h_coarse_results[i];
            hit_results[i * 4] = (int64_t)seeds[hit.x];
            hit_results[i * 4 + 1] = (int64_t)hit.y;
            hit_results[i * 4 + 2] = (int64_t)hit.z;
            hit_results[i * 4 + 3] = (int64_t)hit.w;
        }
        pack_ms = elapsed_ms(stage_started, TierClock::now());
    } else {
        stage_started = TierClock::now();
        cudaMemcpy(g_tier.h_hits, g_tier.d_hits,
                   (size_t)raw_count * sizeof(int4), cudaMemcpyDeviceToHost);
        raw_hits_d2h_ms = elapsed_ms(stage_started, TierClock::now());

        stage_started = TierClock::now();
        size_t translated_size = 0;
        for (int i = 0; i < raw_count; i++) {
            const int4 hit = g_tier.h_hits[i];
            int min_dx = translation_min_index(
                hit.y, translation_period, world_border);
            int max_dx = translation_max_index(
                hit.y, translation_period, world_border);
            int min_dz = translation_min_index(
                hit.z, translation_period, world_border);
            int max_dz = translation_max_index(
                hit.z, translation_period, world_border);
            int64_t count_x = (int64_t)max_dx - min_dx + 1;
            int64_t count_z = (int64_t)max_dz - min_dz + 1;
            if (count_x <= 0 || count_z <= 0)
                continue;
            size_t hit_translations = (size_t)(count_x * count_z);
            if (hit_translations > 0x7FFFFFFFu
                    || translated_size > 0x7FFFFFFFu - hit_translations)
                return -hit_capacity;
            translated_size += hit_translations;
        }
        translation_count_ms = elapsed_ms(stage_started, TierClock::now());
        if (translated_size == 0)
            return 0;

        stage_started = TierClock::now();
        std::vector<int4> translated(translated_size);
        size_t translated_index = 0;
        for (int i = 0; i < raw_count; i++) {
            const int4 hit = g_tier.h_hits[i];
            int min_dx = translation_min_index(
                hit.y, translation_period, world_border);
            int max_dx = translation_max_index(
                hit.y, translation_period, world_border);
            int min_dz = translation_min_index(
                hit.z, translation_period, world_border);
            int max_dz = translation_max_index(
                hit.z, translation_period, world_border);
            for (int dx = min_dx; dx <= max_dx; dx++) {
                for (int dz = min_dz; dz <= max_dz; dz++) {
                    translated[translated_index++] = make_int4(
                        hit.x,
                        (int)((int64_t)hit.y
                            + (int64_t)dx * translation_period),
                        (int)((int64_t)hit.z
                            + (int64_t)dz * translation_period),
                        hit.w);
                }
            }
        }
        translation_fill_ms = elapsed_ms(stage_started, TierClock::now());

        translated_count = (int)translated.size();
        ensure_hit_capacity(translated_count);
        stage_started = TierClock::now();
        cudaMemcpy(g_tier.d_hits, translated.data(),
                   translated.size() * sizeof(int4), cudaMemcpyHostToDevice);
        translation_h2d_ms = elapsed_ms(stage_started, TierClock::now());
        if (debug_stats)
            cudaEventRecord(g_tier.phase_start);
        coarse_verify_r2_2oct<<<translated_count, VERIFY_THREADS>>>(
            g_tier.d_params, g_tier.d_hits, translated_count,
            estimate_step_1x, estimate_step_2x, estimate_threshold,
            g_tier.d_coarse_results);
        if (debug_stats) {
            cudaEventRecord(g_tier.coarse_done);
            cudaEventSynchronize(g_tier.coarse_done);
            cudaEventElapsedTime(
                &estimate_kernel_ms, g_tier.phase_start, g_tier.coarse_done);
        }
        stage_started = TierClock::now();
        cudaMemcpy(g_tier.h_coarse_results, g_tier.d_coarse_results,
                   translated.size() * sizeof(int4), cudaMemcpyDeviceToHost);
        result_d2h_ms = elapsed_ms(stage_started, TierClock::now());

        stage_started = TierClock::now();
        for (int i = 0; i < translated_count; i++) {
            const int4 estimate = g_tier.h_coarse_results[i];
            int64_t estimated_area = (int64_t)(estimate.y * cell_area);
            region_points += estimate.w;
            if (estimate.z != 0)
                clipped_count++;
            if (estimated_area > max_estimated_area)
                max_estimated_area = estimated_area;
            if (estimate.y < minimum_connected_cells)
                continue;
            if (output_count < hit_capacity) {
                const int4 hit = translated[i];
                hit_results[output_count * 4] = (int64_t)seeds[hit.x];
                hit_results[output_count * 4 + 1] = (int64_t)hit.y;
                hit_results[output_count * 4 + 2] = (int64_t)hit.z;
                hit_results[output_count * 4 + 3] = (int64_t)hit.w;
            }
            output_count++;
        }
        pack_ms = elapsed_ms(stage_started, TierClock::now());
        if (output_count > hit_capacity)
            return -output_count;
    }

    if (debug_stats) {
        double estimate_seconds = estimate_kernel_ms / 1000.0;
        double point_rate = estimate_seconds > 0.0
            ? region_points / estimate_seconds : 0.0;
        if (!grouped)
            base_o6_calls = (unsigned long long)region_points;
        unsigned long long o15_calls = region_points;
        double physical_perlin_calls = (double)base_o6_calls
            + (double)o15_calls + (double)fallback_o6_calls;
        double perlin_rate = estimate_seconds > 0.0
            ? physical_perlin_calls / estimate_seconds : 0.0;
        double total_ms = elapsed_ms(total_started, TierClock::now());
        fprintf(stderr,
                "translation_stats mode=%s threads=%d threshold=%.3f "
                "raw=%d expanded=%d clipped=%d area_pass=%d "
                "min_cells=%d max_area=%lld region_points=%lld "
                "scan_ms=%.3f estimate_ms=%.3f logical_points_s=%.3e "
                "physical_perlin_s=%.3e fallback_o6=%llu\n",
                grouped ? "grouped" : "expanded",
                grouped ? grouped_threads : VERIFY_THREADS,
                estimate_threshold, raw_count, translated_count,
                clipped_count, output_count, minimum_connected_cells,
                (long long)max_estimated_area, (long long)region_points,
                scan_kernel_ms, estimate_kernel_ms, point_rate, perlin_rate,
                fallback_o6_calls);
        fprintf(stderr,
                "translation_phases total_ms=%.3f init_ms=%.3f "
                "params_h2d_ms=%.3f scan_setup_ms=%.3f "
                "scan_kernel_ms=%.3f raw_count_d2h_ms=%.3f "
                "raw_hits_d2h_ms=%.3f translation_count_ms=%.3f "
                "translation_fill_ms=%.3f translation_h2d_ms=%.3f "
                "estimate_setup_ms=%.3f estimate_kernel_ms=%.3f "
                "result_count_d2h_ms=%.3f result_d2h_ms=%.3f "
                "pack_ms=%.3f\n",
                total_ms, init_ms, params_h2d_ms, scan_setup_ms,
                scan_kernel_ms, raw_count_d2h_ms, raw_hits_d2h_ms,
                translation_count_ms, translation_fill_ms,
                translation_h2d_ms, estimate_setup_ms, estimate_kernel_ms,
                result_count_d2h_ms, result_d2h_ms, pack_ms);
        fflush(stderr);
    }
    return output_count;
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

extern "C" __declspec(dllexport) int tiered_scan_mem_translated(
    const uint64_t *seeds, int n, int scan_step_2x, int G,
    float o6_threshold, int translation_period, int world_border,
    int estimate_step_1x, int estimate_step_2x,
    int64_t estimate_min_area, int hit_capacity, int64_t *hit_results)
{
    if (n > TIER_CHUNK) return -1;
    return tiered_translation_chunk(
        seeds, n, scan_step_2x, G, o6_threshold,
        translation_period, world_border,
        estimate_step_1x, estimate_step_2x, estimate_min_area,
        hit_capacity, hit_results);
}

extern "C" __declspec(dllexport) int hunt_batch_tiered_translated(
    uint64_t start_seed, int n, int scan_step_2x, int G,
    float o6_threshold, int translation_period, int world_border,
    int estimate_step_1x, int estimate_step_2x,
    int64_t estimate_min_area, int hit_capacity, int64_t *hit_results)
{
    uint64_t *seeds = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) seeds[i] = start_seed + (uint64_t)i;
    int total = tiered_translation_chunk(
        seeds, n, scan_step_2x, G, o6_threshold,
        translation_period, world_border,
        estimate_step_1x, estimate_step_2x, estimate_min_area,
        hit_capacity, hit_results);
    free(seeds);
    return total;
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
        cudaEventDestroy(g_tier.coarse_done);
    }
    if (g_tier.d_params) cudaFree(g_tier.d_params);
    if (g_tier.d_verify_params) cudaFree(g_tier.d_verify_params);
    if (g_tier.d_hit_count) cudaFree(g_tier.d_hit_count);
    if (g_tier.d_translation_stats) cudaFree(g_tier.d_translation_stats);
    if (g_tier.d_hits) cudaFree(g_tier.d_hits);
    if (g_tier.d_coarse_results) cudaFree(g_tier.d_coarse_results);
    free(g_tier.h_params);
    free(g_tier.h_verify_params);
    free(g_tier.h_hits);
    free(g_tier.h_coarse_results);
    g_tier = {};

    if (g_prefilter.d_survivors) cudaFree(g_prefilter.d_survivors);
    if (g_prefilter.d_pass_count) cudaFree(g_prefilter.d_pass_count);
    g_prefilter = {};
}

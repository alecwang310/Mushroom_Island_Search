/*
 * continentalness.h — Fast MC 1.18+ continentalness sampler.
 * Self-contained single-header C library. No cubiomes dependency.
 *
 * Memory: ~10 KB per ContEngine instance (24 PerlinNoise structs flattened).
 * Thread-safe after init (all sampling is read-only).
 *
 * Mushroom fields: -1.2 < continentalness < -1.05 at 1:4 scale.
 */
#ifndef CONTINENTALNESS_H
#define CONTINENTALNESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum octaves: 3 (shift octA) + 3 (shift octB) + 9 (cont octA) + 9 (cont octB) */
#define CONT_MAX_OCTAVES 24
#define CONT_PERM_SIZE 257
#define CONT_TIERED_OCTAVES 2
#define CONT_TIERED_PERM_SIZE 256
#define CONT_VERIFY_OCTAVES 6
#define CONT_VERIFY_PERM_SIZE 256

typedef struct {
    /* Perlin noise parameters — one flat array per field (SoA layout for SIMD) */
    uint8_t perm[CONT_MAX_OCTAVES][CONT_PERM_SIZE];
    double  offset_a[CONT_MAX_OCTAVES];
    double  offset_b[CONT_MAX_OCTAVES];
    double  offset_c[CONT_MAX_OCTAVES];
    double  amplitude[CONT_MAX_OCTAVES];
    double  lacunarity[CONT_MAX_OCTAVES];
    uint8_t cached_h2[CONT_MAX_OCTAVES];
    double  cached_d2[CONT_MAX_OCTAVES];
    double  cached_t2[CONT_MAX_OCTAVES];

    /* Ranges into the perlin arrays */
    int shift_octA_start, shift_octA_count;
    int shift_octB_start, shift_octB_count;
    int cont_octA_start,  cont_octA_count;
    int cont_octB_start,  cont_octB_count;

    /* DoublePerlin amplitude scaling */
    double shift_dbl_amp;
    double cont_dbl_amp;

    /* Frequency ratio for DoublePerlin octB (337/331) */
    double freq_ratio;
} ContEngine;

/* Compact state for the O6+O15 tiered GPU scan. */
typedef struct {
    uint8_t perm[CONT_TIERED_OCTAVES][CONT_TIERED_PERM_SIZE];
    float offset_a[CONT_TIERED_OCTAVES];
    float offset_b[CONT_TIERED_OCTAVES];
    float offset_c[CONT_TIERED_OCTAVES];
    float amplitude[CONT_TIERED_OCTAVES];
    float lacunarity[CONT_TIERED_OCTAVES];
    uint8_t cached_h2[CONT_TIERED_OCTAVES];
    uint8_t padding[2];
    float cached_d2[CONT_TIERED_OCTAVES];
    float cached_t2[CONT_TIERED_OCTAVES];
    float cont_dbl_amp;
} ContTieredParams;

/* Compact six-octave state for the GPU coarse verifier.  The octave order is
 * O6, O7, O8, O15, O16, O17, matching cont_engine_init_6oct(). */
typedef struct {
    uint8_t perm[CONT_VERIFY_OCTAVES][CONT_VERIFY_PERM_SIZE];
    float offset_a[CONT_VERIFY_OCTAVES];
    float offset_b[CONT_VERIFY_OCTAVES];
    float offset_c[CONT_VERIFY_OCTAVES];
    float amplitude[CONT_VERIFY_OCTAVES];
    float lacunarity[CONT_VERIFY_OCTAVES];
    uint8_t cached_h2[CONT_VERIFY_OCTAVES];
    uint8_t padding[2];
    float cached_d2[CONT_VERIFY_OCTAVES];
    float cached_t2[CONT_VERIFY_OCTAVES];
    float cont_dbl_amp;
} ContVerifyParams;

/**
 * Initialize the engine for a world seed.
 * @param seed         64-bit Minecraft world seed
 * @param large_biomes 0 = normal, 1 = large biomes
 */
void cont_engine_init(ContEngine *e, uint64_t seed, int large_biomes);

/**
 * Disable shift distortion after initialization while preserving continentalness.
 */
void cont_engine_disable_shift(ContEngine *e);

/**
 * Initialize only the six octaves used by cont_flood_fill_6oct:
 * continentalness octaves 6, 7, 8, 15, 16, and 17.
 */
void cont_engine_init_6oct(ContEngine *e, uint64_t seed, int large_biomes);

/**
 * Sample continentalness at a single block coordinate.
 * Returns value in ~[-2, 2]; mushroom fields when in (-1.2, -1.05).
 * Thread-safe (read-only after init). ~30ns per call.
 */
double cont_sample(const ContEngine *e, int x, int z);

/**
 * Sample a grid of continentalness values.
 * @param out     Output array (float[rows*cols]), row-major
 * @param x0, z0  Top-left corner (block coordinates)
 * @param cols, rows  Grid dimensions
 * @param step    Spacing between samples (e.g., 4 for 1:4 scale)
 */
void cont_sample_grid(const ContEngine *e, float *out,
                      int x0, int z0, int cols, int rows, int step);

/**
 * Estimate connected mushroom area around a GPU component hit.
 *
 * Samples the 0.5x hex lattice and performs the connected-component walk in
 * C so callers do not pay a Python/ctypes boundary for every sample.
 * @param geometry_code low 6 bits = zero, one, or two GPU neighbor
 *                      directions; bit 6 = coarse-grid row parity
 * @param step_05x      0.5x sample spacing, normally 125 blocks
 * @param step_2x       GPU coarse spacing, normally 500 blocks
 * @return estimated area in blocks^2 at 1:1 scale
 */
double cont_estimate_triple_area(uint64_t seed, int gx, int gz,
                                 int geometry_code, int step_05x,
                                 int step_2x);

/**
 * Six-octave version of cont_estimate_triple_area for the CPU validation gate.
 * Samples a 1x hex lattice, normally at 250-block spacing around a 500-block
 * GPU hit, and uses only O6, O7, O8, O15, O16, and O17.
 * A sample is considered mushroom when its continentalness is below threshold.
 */
double cont_estimate_triple_area_6oct(uint64_t seed, int gx, int gz,
                                      int geometry_code, int step_1x,
                                      int step_2x, double threshold);

/**
 * Batch-init N seeds and write perlin data directly into flat SoA arrays.
 * One call instead of N — eliminates Python loop overhead for GPU uploads.
 *
 * Layout (each array has N*MAX elements, row-major: seed first, then octave):
 *   perms:  uint8[N * 24 * 257]
 *   oa, oc: float[N * 24]
 *   amp, lac: float[N * 24]
 *   h2: uint8[N * 24]
 *   d2, t2: float[N * 24]
 *   ranges: int32[N * 8]
 *   dbl_amps: float[N * 2]
 */
void cont_batch_init(const uint64_t *seeds, int n, int large_biomes,
                     uint8_t *perms, float *oa, float *ob, float *oc,
                     float *amp, float *lac, uint8_t *h2,
                     float *d2, float *t2, int32_t *ranges, float *dbl_amps);

/**
 * Initialize only the two continentalness octaves used by tiered_scan:
 * the first octave of continentalness A and B (O6 and O15 for normal worlds).
 */
void cont_batch_init_tiered(const uint64_t *seeds, int n, int large_biomes,
                            ContTieredParams *out);

/**
 * Initialize the six essential continentalness octaves used by the GPU
 * coarse verifier: O6, O7, O8, O15, O16, and O17.
 */
void cont_batch_init_6oct(const uint64_t *seeds, int n, int large_biomes,
                          ContVerifyParams *out);

/**
 * Flood fill to measure mushroom island area.
 * @param seed       World seed
 * @param cx, cz     Start cell coordinates at 1:4 scale (int)
 * @param max_cells  Maximum BFS cells to visit (safety cap)
 * @return           Area in blocks^2 at 1:1 scale (cells * 16), or 0
 *                   if the start cell is not mushroom.
 */
int64_t cont_flood_fill(uint64_t seed, int cx, int cz, int max_cells);

/**
 * Same as cont_flood_fill but only uses 6 essential octaves (6,7,8,15,16,17).
 * Much faster for pre-filtering candidates before full flood fill.
 */
int64_t cont_flood_fill_6oct(uint64_t seed, int cx, int cz, int max_cells);

/**
 * Flood fill using only O6 and O15 with a configurable continentalness
 * threshold. Intended for calibrating cheap GPU area screens.
 */
int64_t cont_flood_fill_2oct(uint64_t seed, int cx, int cz,
                             double threshold, int max_cells);

#ifdef __cplusplus
}
#endif
#endif /* CONTINENTALNESS_H */

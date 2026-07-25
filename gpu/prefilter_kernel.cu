/*
 * prefilter_kernel.cu — GPU-side seed pre-filter via O6+O15 variance LUT.
 *
 * Simulates Xoroshiro128++ RNG per seed to extract ob6 and ob15,
 * then scores: score = amp6²·LUT[frac(ob6)] + amp15²·LUT[frac(ob15)]
 *
 * One thread per seed. ~520 RNG calls/seed. Rejection sampling skipped
 * for perm generation (bias < 1e-7, negligible for statistical filter).
 *
 * Output: scores[i], and pass_idx[] (compacted indices of seeds above threshold).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include "variance_lut.h"

#define THREADS_PER_BLOCK 256

// ═══════════════════════════════════════════════════════════════════════════
// Xoroshiro128++ — exact match with Minecraft/continentalness_pipeline.py
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__ void xNextLong(
    uint64_t &lo, uint64_t &hi, uint64_t &out)
{
    uint64_t n = rotl64(lo + hi, 17) + lo;
    hi ^= lo;
    lo = rotl64(lo, 49) ^ hi ^ (hi << 21);
    hi = rotl64(hi, 28);
    out = n;
}

__device__ __forceinline__ double xNextDouble(uint64_t &lo, uint64_t &hi) {
    uint64_t n;
    xNextLong(lo, hi, n);
    return (double)(n >> 11) * (1.0 / (double)(1ULL << 53));
}

// Simplified xNextInt: skip rejection sampling. Bias < 1/2^32 per call.
// For n=256, 2^32/256 = 16,777,216 exactly → zero bias.
__device__ __forceinline__ int xNextIntFast(uint64_t &lo, uint64_t &hi, int n) {
    uint64_t rng;
    xNextLong(lo, hi, rng);
    return (int)(((rng & 0xFFFFFFFFULL) * (uint64_t)n) >> 32);
}

__device__ void xSetSeed(uint64_t seed, uint64_t &lo, uint64_t &hi) {
    uint64_t XL = 0x9e3779b97f4a7c15ULL;
    uint64_t XH = 0x6a09e667f3bcc909ULL;
    uint64_t A  = 0xbf58476d1ce4e5b9ULL;
    uint64_t B  = 0x94d049bb133111ebULL;
    lo = (seed ^ XH);
    hi = (lo + XL);
    lo = ((lo ^ (lo >> 30)) * A);
    hi = ((hi ^ (hi >> 30)) * A);
    lo = ((lo ^ (lo >> 27)) * B);
    hi = ((hi ^ (hi >> 27)) * B);
    lo = (lo ^ (lo >> 31));
    hi = (hi ^ (hi >> 31));
}

// ═══════════════════════════════════════════════════════════════════════════
// ob extraction: PerlinNoise.from_xoroshiro draws oa, ob, oc, THEN 256 perm.
// Each octave has independent RNG state. We only need ob: draw oa, then ob.
// ═══════════════════════════════════════════════════════════════════════════

__device__ float extract_ob(uint64_t ixlo, uint64_t ixhi, int octave_md5_idx) {
    uint64_t lo = ixlo ^ c_md5_octave[octave_md5_idx][0];
    uint64_t hi = ixhi ^ c_md5_octave[octave_md5_idx][1];
    xNextDouble(lo, hi);                              // oa (discard)
    return (float)(xNextDouble(lo, hi) * 256.0);      // ob = raw * 256 (Minecraft scaling)
}

// ═══════════════════════════════════════════════════════════════════════════
// LUT lookup: linear interpolation between nearest c_lut entries
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float lut_lookup(float dy) {
    // dy in [0, 1). Clamp and scale to LUT index.
    float fi = dy * (float)(LUT_SIZE - 1);
    int i = (int)fi;
    float frac = fi - (float)i;
    // Clamp
    if (i < 0) { i = 0; frac = 0.0f; }
    if (i >= LUT_SIZE - 1) { i = LUT_SIZE - 2; frac = 1.0f; }
    return c_lut[i] * (1.0f - frac) + c_lut[i + 1] * frac;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main pre-filter kernel
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void prefilter_seeds(
    const uint64_t *seeds, int n,
    float *scores, int *pass_idx, int *pass_count,
    float lo_thresh, float hi_thresh, int large_biomes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint64_t seed = seeds[tid];

    // ── Step 1: xSetSeed ─────────────────────────────────────────────
    uint64_t lo, hi;
    xSetSeed(seed, lo, hi);

    // ── Step 2: xlo, xhi ─────────────────────────────────────────────
    uint64_t xlo, xhi;
    xNextLong(lo, hi, xlo);  // consume 1 call
    xNextLong(lo, hi, xhi);  // consume 1 call

    // ── Step 3: start continentalness RNG ────────────────────────────
    uint64_t cont_lo = xlo ^ (large_biomes ? MD5_CONT_LARGE_LO : MD5_CONT_LO);
    uint64_t cont_hi = xhi ^ (large_biomes ? MD5_CONT_LARGE_HI : MD5_CONT_HI);

    // ── Step 4: cont A and cont B ixlo/ixhi ─────────────────────────
    uint64_t ixlo_A, ixhi_A, ixlo_B, ixhi_B;
    xNextLong(cont_lo, cont_hi, ixlo_A);   // cont A ixlo
    xNextLong(cont_lo, cont_hi, ixhi_A);   // cont A ixhi
    xNextLong(cont_lo, cont_hi, ixlo_B);   // cont B ixlo
    xNextLong(cont_lo, cont_hi, ixhi_B);   // cont B ixhi

    // ── Step 5: extract ob6 (cont A octave 0, omin=-9 → md5_idx=3) ─
    float ob6 = extract_ob(ixlo_A, ixhi_A, 3);

    // ── Step 6: extract ob15 (cont B octave 0, omin=-9 → md5_idx=3) ─
    float ob15 = extract_ob(ixlo_B, ixhi_B, 3);

    // ── Step 7: LUT lookup + score ───────────────────────────────────
    float dy6 = ob6 - floorf(ob6);
    float dy15 = ob15 - floorf(ob15);

    float score = CONT_AMP_SQ * lut_lookup(dy6)
                + CONT_AMP_SQ * lut_lookup(dy15);

    scores[tid] = score;

    // ── Step 8: stream compaction (band pass: lo ≤ score < hi) ─────
    if (score >= lo_thresh && score < hi_thresh) {
        int idx = atomicAdd(pass_count, 1);
        pass_idx[idx] = tid;
    }
}

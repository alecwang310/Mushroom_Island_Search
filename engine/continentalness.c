/*
 * continentalness.c — Self-contained MC 1.18+ continentalness sampler.
 *
 * Verbatim from cubiomes (rng.h, noise.c, biomenoise.c) with optimizations:
 *   - y=0 always → d2==0 fast path inlined
 *   - SoA layout for cache-friendly grid sampling
 *   - Branch-free indexed_lerp via lookup table
 *
 * Compile: gcc -O3 -march=native -shared -fPIC -o continentalness.so continentalness.c -lm
 *      or: cl /O2 /arch:AVX512 /LD continentalness.c /Fecontinentalness.dll
 */

#include "continentalness.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ==========================================================================
 * 64-bit helpers
 * ========================================================================== */
#define M64 0xFFFFFFFFFFFFFFFFULL

static inline uint64_t rotl64(uint64_t x, int k) {
    return ((x << k) | (x >> (64 - k))) & M64;
}

/* ==========================================================================
 * Xoroshiro128++ (verbatim from cubiomes rng.h)
 * ========================================================================== */
typedef struct { uint64_t lo, hi; } Xoroshiro;

static void xSetSeed(Xoroshiro *xr, uint64_t value) {
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A  = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B  = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l ^= (l >> 31);
    h ^= (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

static inline uint64_t xNextLong(Xoroshiro *xr) {
    uint64_t l = xr->lo, h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

static inline double xNextDouble(Xoroshiro *xr) {
    return (xNextLong(xr) >> 11) * (1.0 / (1ULL << 53));
}

static int xNextInt(Xoroshiro *xr, uint32_t n) {
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n) {
        uint32_t t = (~n + 1) % n;
        while ((uint32_t)r < t)
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    }
    return (int)(r >> 32);
}

/* ==========================================================================
 * MD5 octave hashes (from cubiomes noise.c)
 * ========================================================================== */
static const uint64_t md5_octave[13][2] = {
    {0xb198de63a8012672ULL, 0x7b84cad43ef7b5a8ULL}, /* -12 */
    {0x0fd787bfbc403ec3ULL, 0x74a4a31ca21b48b8ULL}, /* -11 */
    {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL}, /* -10 */
    {0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, /* -9  */
    {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, /* -8  */
    {0xf11268128982754fULL, 0x257a1d670430b0aaULL}, /* -7  */
    {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}, /* -6  */
    {0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL}, /* -5  */
    {0xbd90d5377ba1b762ULL, 0xc07317d419a7548dULL}, /* -4  */
    {0x53d39c6752dac858ULL, 0xbcd1c5a80ab65b3eULL}, /* -3  */
    {0xb4a24d7a84e7677bULL, 0x023ff9668e89b5c4ULL}, /* -2  */
    {0xdffa22b534c5f608ULL, 0xb9b67517d3665ca9ULL}, /* -1  */
    {0xd50708086cef4d7cULL, 0x6e1651ecc7f43309ULL}, /* 0   */
};

/* ==========================================================================
 * Lookup tables (from cubiomes noise.c)
 * ========================================================================== */
static const double lacuna_ini[] = {
    1.0, 0.5, 0.25, 1./8, 1./16, 1./32, 1./64,
    1./128, 1./256, 1./512, 1./1024, 1./2048, 1./4096
};
static const double persist_ini[] = {
    0.0, 1.0, 2./3, 4./7, 8./15, 16./31, 32./63, 64./127, 128./255, 256./511
};
static const double amp_ini[] = {
    0.0, 5./6, 10./9, 15./12, 20./15, 25./18, 30./21, 35./24, 40./27, 45./30
};

/* Branch-free indexed-lerp: precomputed 16×3 gradient vectors */
static const float grad_table[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

/* ==========================================================================
 * Perlin noise init (xPerlinInit from cubiomes noise.c)
 * ========================================================================== */
static void perlin_init(uint8_t perm[257], double *a, double *b, double *c,
                        uint8_t *h2, double *d2, double *t2,
                        uint64_t lo, uint64_t hi) {
    Xoroshiro xr = {lo, hi};
    *a = xNextDouble(&xr) * 256.0;
    *b = xNextDouble(&xr) * 256.0;
    *c = xNextDouble(&xr) * 256.0;

    /* Fisher-Yates shuffle */
    int i;
    for (i = 0; i < 256; i++) perm[i] = (uint8_t)i;
    for (i = 0; i < 256; i++) {
        int j = xNextInt(&xr, 256 - i) + i;
        uint8_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    perm[256] = perm[0];

    /* Cache d2/h2/t2 for y=0 fast path */
    double ib = floor(*b), db = *b - ib;
    *h2 = (uint8_t)((int)ib & 0xFF);
    *d2 = db;
    *t2 = db*db*db * (db * (db*6.0 - 15.0) + 10.0);
}

/* ==========================================================================
 * Perlin noise sample — y=0 fast path (samplePerlin from cubiomes noise.c)
 * ========================================================================== */
static inline double perlin_sample(
    const uint8_t perm[257], double a, double b, double c,
    uint8_t cached_h2, double cached_d2, double cached_t2,
    double x, double y, double z)
{
    double d1 = x + a;
    double d2 = y + b;
    double d3 = z + c;

    uint8_t h2; double t2;
    if (y == 0.0) {
        /* Fast path: use cached values from noise->b */
        h2 = cached_h2;
        d2 = cached_d2;
        t2 = cached_t2;
    } else {
        double i2 = floor(d2);
        d2 -= i2;
        h2 = (uint8_t)((int)i2 & 0xFF);
        t2 = d2*d2*d2 * (d2 * (d2*6.0 - 15.0) + 10.0);
    }

    double i1 = floor(d1), i3 = floor(d3);
    d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF, h3 = ((int)i3) & 0xFF;

    double t1 = d1*d1*d1 * (d1 * (d1*6.0 - 15.0) + 10.0);
    double t3 = d3*d3*d3 * (d3 * (d3*6.0 - 15.0) + 10.0);

    /* vec2 lookup chain (matching cubiomes exactly) */
    const uint8_t *idx = perm;
    int v1_a = idx[h1] + h2, v1_b = idx[h1+1] + h2;
    int v2_a = idx[v1_a & 0xFF] + h3, v2_b = idx[(v1_a & 0xFF)+1] + h3;
    int v3_a = idx[v1_b & 0xFF] + h3, v3_b = idx[(v1_b & 0xFF)+1] + h3;
    int v4_a = idx[v2_a & 0xFF],     v4_b = idx[(v2_a & 0xFF)+1];
    int v5_a = idx[v2_b & 0xFF],     v5_b = idx[(v2_b & 0xFF)+1];
    int v6_a = idx[v3_a & 0xFF],     v6_b = idx[(v3_a & 0xFF)+1];
    int v7_a = idx[v3_b & 0xFF],     v7_b = idx[(v3_b & 0xFF)+1];

    /* Branch-free indexed_lerp via gradient table */
    #define LERP(idx,x,y,z) (grad_table[(idx)&0xF][0]*(x) + \
                             grad_table[(idx)&0xF][1]*(y) + \
                             grad_table[(idx)&0xF][2]*(z))

    double l1 = LERP(v4_a, d1, d2, d3);
    double l5 = LERP(v4_b, d1, d2, d3-1);
    double l2 = LERP(v6_a, d1-1, d2, d3);
    double l6 = LERP(v6_b, d1-1, d2, d3-1);
    double l3 = LERP(v5_a, d1, d2-1, d3);
    double l7 = LERP(v5_b, d1, d2-1, d3-1);
    double l4 = LERP(v7_a, d1-1, d2-1, d3);
    double l8 = LERP(v7_b, d1-1, d2-1, d3-1);

    /* Trilinear interpolation */
    l1 += t1 * (l2 - l1); l3 += t1 * (l4 - l3);
    l5 += t1 * (l6 - l5); l7 += t1 * (l8 - l7);
    l1 += t2 * (l3 - l1); l5 += t2 * (l7 - l5);
    return l1 + t3 * (l5 - l1);
    #undef LERP
}

/* ==========================================================================
 * Octave init (xOctaveInit from cubiomes noise.c)
 * Returns number of octaves created.
 * ========================================================================== */
static int octave_init(
    ContEngine *e, int idx_start,
    uint64_t xlo, uint64_t xhi,
    const double *amplitudes, int omin, int length, int nmax)
{
    /* Each octave init uses xlo/xhi XOR octave MD5 hash */
    double lacuna = lacuna_ini[-omin];
    double persist = persist_ini[length];
    int n = 0;

    for (int i = 0; i < length; i++) {
        if (n == nmax) break;
        if (amplitudes[i] == 0.0) {
            lacuna *= 2.0; persist *= 0.5;
            continue;
        }
        int slot = idx_start + n;
        uint64_t slo = xlo ^ md5_octave[12 + omin + i][0];
        uint64_t shi = xhi ^ md5_octave[12 + omin + i][1];
        perlin_init(e->perm[slot],
                    &e->offset_a[slot], &e->offset_b[slot], &e->offset_c[slot],
                    &e->cached_h2[slot], &e->cached_d2[slot], &e->cached_t2[slot],
                    slo, shi);
        e->amplitude[slot] = amplitudes[i] * persist;
        e->lacunarity[slot] = lacuna;
        n++;
        lacuna *= 2.0; persist *= 0.5;
    }
    return n;
}

/* ==========================================================================
 * DoublePerlin init (xDoublePerlinInit from cubiomes noise.c)
 * ========================================================================== */
static int double_perlin_init(
    ContEngine *e, int *octA_start, int *octA_count,
    int *octB_start, int *octB_count, double *dbl_amp,
    Xoroshiro *xr,
    const double *amplitudes, int omin, int length, int nmax)
{
    int na = -1, nb = -1;
    if (nmax > 0) { na = (nmax + 1) >> 1; nb = nmax - na; }

    /* octA consumes 2 longs from shared xr */
    uint64_t axlo = xNextLong(xr), axhi = xNextLong(xr);
    int start_a = *octA_start;
    int cnt_a = octave_init(e, start_a, axlo, axhi, amplitudes, omin, length, na);
    *octA_start = start_a;
    *octA_count = cnt_a;

    /* octB consumes next 2 longs from shared xr (DIFFERENT values!) */
    uint64_t bxlo = xNextLong(xr), bxhi = xNextLong(xr);
    int start_b = start_a + cnt_a;
    int cnt_b = octave_init(e, start_b, bxlo, bxhi, amplitudes, omin, length, nb);
    *octB_start = start_b;
    *octB_count = cnt_b;

    /* Effective length after trimming zero amplitudes */
    int eff = length;
    while (eff > 0 && amplitudes[eff-1] == 0.0) eff--;
    for (int i = 0; i < eff && amplitudes[i] == 0.0; i++) eff--;
    *dbl_amp = amp_ini[eff];

    return cnt_a + cnt_b;
}

/* ==========================================================================
 * Climate parameter init (init_climate_seed from cubiomes biomenoise.c)
 * ========================================================================== */
static int climate_init(
    ContEngine *e, int *octA_start, int *octA_count,
    int *octB_start, int *octB_count, double *dbl_amp,
    uint64_t xlo, uint64_t xhi, int large, int nptype, int nmax,
    int idx_next)
{
    Xoroshiro xr;
    int n = 0;

    switch (nptype) {
    case 4: { /* NP_SHIFT = 4 */
        static const double amp[] = {1, 1, 1, 0};
        /* md5 "minecraft:offset" */
        xr.lo = xlo ^ 0x080518cf6af25384ULL;
        xr.hi = xhi ^ 0x3f3dfb40a54febd5ULL;
        *octA_start = idx_next;
        n = double_perlin_init(e, octA_start, octA_count, octB_start, octB_count,
                               dbl_amp, &xr, amp, -3, 4, nmax);
        break;
    }
    case 2: { /* NP_CONTINENTALNESS = 2 */
        static const double amp[] = {1, 1, 2, 2, 2, 1, 1, 1, 1};
        /* md5 "minecraft:continentalness" or "minecraft:continentalness_large" */
        xr.lo = xlo ^ (large ? 0x9a3f51a113fce8dcULL : 0x83886c9d0ae3a662ULL);
        xr.hi = xhi ^ (large ? 0xee2dbd157e5dcdadULL : 0xafa638a61b42e8adULL);
        *octA_start = idx_next;
        n = double_perlin_init(e, octA_start, octA_count, octB_start, octB_count,
                               dbl_amp, &xr, amp, large ? -11 : -9, 9, nmax);
        break;
    }
    default:
        fprintf(stderr, "climate_init: unsupported parameter %d\n", nptype);
        exit(1);
    }
    return n;
}

/* ==========================================================================
 * Public API
 * ========================================================================== */

void cont_engine_init(ContEngine *e, uint64_t seed, int large_biomes) {
    memset(e, 0, sizeof(*e));
    e->freq_ratio = 337.0 / 331.0;

    /* Seed Xoroshiro and extract global xlo/xhi */
    Xoroshiro xr;
    xSetSeed(&xr, seed);
    uint64_t xlo = xNextLong(&xr);
    uint64_t xhi = xNextLong(&xr);

    /* Init shift */
    int idx = 0;
    climate_init(e,
        &e->shift_octA_start, &e->shift_octA_count,
        &e->shift_octB_start, &e->shift_octB_count,
        &e->shift_dbl_amp,
        xlo, xhi, large_biomes, 4/*NP_SHIFT*/, -1, idx);
    idx = e->shift_octB_start + e->shift_octB_count;

    /* Init continentalness */
    climate_init(e,
        &e->cont_octA_start, &e->cont_octA_count,
        &e->cont_octB_start, &e->cont_octB_count,
        &e->cont_dbl_amp,
        xlo, xhi, large_biomes, 2/*NP_CONTINENTALNESS*/, -1, idx);
}

/* Sample one DoublePerlinNoise at (x, y, z) */
static inline double sample_double_perlin(
    const ContEngine *e,
    int octA_start, int octA_count,
    int octB_start, int octB_count,
    double dbl_amp, double x, double y, double z)
{
    double v = 0.0;
    int i;

    /* Sample octA */
    for (i = octA_start; i < octA_start + octA_count; i++) {
        double lf = e->lacunarity[i];
        v += e->amplitude[i] * perlin_sample(
            e->perm[i], e->offset_a[i], e->offset_b[i], e->offset_c[i],
            e->cached_h2[i], e->cached_d2[i], e->cached_t2[i],
            x * lf, y * lf, z * lf);
    }

    /* Sample octB at scaled frequency */
    double fr = e->freq_ratio;
    for (i = octB_start; i < octB_start + octB_count; i++) {
        double lf = e->lacunarity[i];
        v += e->amplitude[i] * perlin_sample(
            e->perm[i], e->offset_a[i], e->offset_b[i], e->offset_c[i],
            e->cached_h2[i], e->cached_d2[i], e->cached_t2[i],
            x * lf * fr, y * lf * fr, z * lf * fr);
    }

    return v * dbl_amp;
}

double cont_sample(const ContEngine *e, int x, int z) {
    /* Shift distortion — matching cubiomes exactly:
     *   dx = sampleDoublePerlin(shift, x, 0, z)   → y=0
     *   dz = sampleDoublePerlin(shift, z, x, 0)   → y=x, NOT y=0! */
    double dx = sample_double_perlin(e,
        e->shift_octA_start, e->shift_octA_count,
        e->shift_octB_start, e->shift_octB_count,
        e->shift_dbl_amp, (double)x, 0.0, (double)z);
    double dz = sample_double_perlin(e,
        e->shift_octA_start, e->shift_octA_count,
        e->shift_octB_start, e->shift_octB_count,
        e->shift_dbl_amp, (double)z, (double)x, 0.0);  /* y=x! */

    double px = x + dx * 4.0;
    double pz = z + dz * 4.0;

    /* Continentalness at shifted position, y=0 */
    return sample_double_perlin(e,
        e->cont_octA_start, e->cont_octA_count,
        e->cont_octB_start, e->cont_octB_count,
        e->cont_dbl_amp, px, 0.0, pz);
}

void cont_sample_grid(const ContEngine *e, float *out,
                      int x0, int z0, int cols, int rows, int step) {
    for (int r = 0; r < rows; r++) {
        int z = z0 + r * step;
        float *row_out = out + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            int x = x0 + c * step;
            row_out[c] = (float)cont_sample(e, x, z);
        }
    }
}

void cont_batch_init(const uint64_t *seeds, int n, int large_biomes,
                     uint8_t *perms, float *oa, float *ob, float *oc,
                     float *amp, float *lac, uint8_t *h2,
                     float *d2, float *t2, int32_t *ranges, float *dbl_amps)
{
    #define MAX_OCT 24
    #define PSZ 257
    ContEngine e;
    for (int s = 0; s < n; s++) {
        cont_engine_init(&e, seeds[s], large_biomes);

        int p_off = s * MAX_OCT * PSZ;
        int v_off = s * MAX_OCT;
        memcpy(perms + p_off, e.perm, MAX_OCT * PSZ);
        for (int i = 0; i < MAX_OCT; i++) {
            oa[v_off + i]  = (float)e.offset_a[i];
            ob[v_off + i]  = (float)e.offset_b[i];
            oc[v_off + i]  = (float)e.offset_c[i];
            amp[v_off + i] = (float)e.amplitude[i];
            lac[v_off + i] = (float)e.lacunarity[i];
            h2[v_off + i]  = e.cached_h2[i];
            d2[v_off + i]  = (float)e.cached_d2[i];
            t2[v_off + i]  = (float)e.cached_t2[i];
        }
        memcpy(ranges + s * 8, &e.shift_octA_start, 8 * sizeof(int32_t));
        dbl_amps[s * 2 + 0] = (float)e.shift_dbl_amp;
        dbl_amps[s * 2 + 1] = (float)e.cont_dbl_amp;
    }
}

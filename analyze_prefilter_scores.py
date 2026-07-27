"""analyze_prefilter_scores.py — Analyze prefilter score distribution by island size.

Reproduces the GPU prefilter scoring algorithm in Python and reports
how the variance LUT score distributes across island size buckets.
"""

import json
import math
from collections import defaultdict

# ── LUT from variance_lut.h ─────────────────────────────────────────────────
LUT = [
    0.05816394, 0.05817095, 0.05819171, 0.05822581, 0.05827285, 0.05833243, 0.05840419, 0.05848772,
    0.05858267, 0.05868867, 0.05880535, 0.05893235, 0.05906934, 0.05921596, 0.05937188, 0.05953675,
    0.05971026, 0.05989208, 0.06008190, 0.06027940, 0.06048428, 0.06069625, 0.06091502, 0.06114029,
    0.06137180, 0.06160927, 0.06185244, 0.06210105, 0.06235486, 0.06261362, 0.06287709, 0.06314506,
    0.06341729, 0.06369357, 0.06397369, 0.06425746, 0.06454467, 0.06483515, 0.06512870, 0.06542516,
    0.06572434, 0.06602609, 0.06633025, 0.06663666, 0.06694516, 0.06725563, 0.06756791, 0.06788186,
    0.06819736, 0.06851426, 0.06883245, 0.06915180, 0.06947218, 0.06979347, 0.07011555, 0.07043831,
    0.07076162, 0.07108537, 0.07140945, 0.07173373, 0.07205810, 0.07238244, 0.07270664, 0.07303057,
    0.07335412, 0.07367717, 0.07399959, 0.07432126, 0.07464205, 0.07496183, 0.07528048, 0.07559787,
    0.07591385, 0.07622829, 0.07654105, 0.07685199, 0.07716097, 0.07746784, 0.07777245, 0.07807465,
    0.07837428, 0.07867119, 0.07896522, 0.07925622, 0.07954401, 0.07982845, 0.08010935, 0.08038656,
    0.08065991, 0.08092923, 0.08119434, 0.08145510, 0.08171131, 0.08196281, 0.08220944, 0.08245102,
    0.08268738, 0.08291837, 0.08314380, 0.08336353, 0.08357737, 0.08378519, 0.08398681, 0.08418209,
    0.08437087, 0.08455300, 0.08472834, 0.08489675, 0.08505809, 0.08521223, 0.08535905, 0.08549842,
    0.08563022, 0.08575435, 0.08587069, 0.08597915, 0.08607964, 0.08617207, 0.08625635, 0.08633242,
    0.08640021, 0.08645966, 0.08651071, 0.08655332, 0.08658746, 0.08661309, 0.08663018, 0.08663874,
    0.08663874, 0.08663018, 0.08661309, 0.08658746, 0.08655332, 0.08651071, 0.08645966, 0.08640021,
    0.08633242, 0.08625635, 0.08617207, 0.08607964, 0.08597915, 0.08587069, 0.08575435, 0.08563022,
    0.08549842, 0.08535905, 0.08521223, 0.08505809, 0.08489675, 0.08472834, 0.08455300, 0.08437087,
    0.08418209, 0.08398681, 0.08378519, 0.08357737, 0.08336353, 0.08314380, 0.08291837, 0.08268738,
    0.08245102, 0.08220944, 0.08196281, 0.08171131, 0.08145510, 0.08119434, 0.08092923, 0.08065991,
    0.08038656, 0.08010935, 0.07982845, 0.07954401, 0.07925622, 0.07896522, 0.07867119, 0.07837428,
    0.07807465, 0.07777245, 0.07746784, 0.07716097, 0.07685199, 0.07654105, 0.07622829, 0.07591385,
    0.07559787, 0.07528048, 0.07496183, 0.07464205, 0.07432126, 0.07399959, 0.07367717, 0.07335412,
    0.07303057, 0.07270664, 0.07238244, 0.07205810, 0.07173373, 0.07140945, 0.07108537, 0.07076162,
    0.07043831, 0.07011555, 0.06979347, 0.06947218, 0.06915180, 0.06883245, 0.06851426, 0.06819736,
    0.06788186, 0.06756791, 0.06725563, 0.06694516, 0.06663666, 0.06633025, 0.06602609, 0.06572434,
    0.06542516, 0.06512870, 0.06483515, 0.06454467, 0.06425746, 0.06397369, 0.06369357, 0.06341729,
    0.06314506, 0.06287709, 0.06261362, 0.06235486, 0.06210105, 0.06185244, 0.06160927, 0.06137180,
    0.06114029, 0.06091502, 0.06069625, 0.06048428, 0.06027940, 0.06008190, 0.05989208, 0.05971026,
    0.05953675, 0.05937188, 0.05921596, 0.05906934, 0.05893235, 0.05880535, 0.05868867, 0.05858267,
    0.05848772, 0.05840419, 0.05833243, 0.05827285, 0.05822581, 0.05819171, 0.05817095, 0.05816394,
]

CONT_AMP_SQ = 0.25097943

# ── Xoroshiro128++ (matches prefilter_kernel.cu) ────────────────────────────
MASK64 = 0xFFFFFFFFFFFFFFFF

def rotl64(x, k):
    return ((x << k) | (x >> (64 - k))) & MASK64

def xNextLong(lo, hi):
    n = (rotl64((lo + hi) & MASK64, 17) + lo) & MASK64
    hi ^= lo
    lo = rotl64(lo, 49) ^ hi ^ ((hi << 21) & MASK64)
    hi = rotl64(hi, 28)
    return lo, hi, n

def xNextDouble(lo, hi):
    lo, hi, n = xNextLong(lo, hi)
    return lo, hi, (n >> 11) * (1.0 / (1 << 53))

def xSetSeed(seed):
    XL = 0x9e3779b97f4a7c15
    XH = 0x6a09e667f3bcc909
    A  = 0xbf58476d1ce4e5b9
    B  = 0x94d049bb133111eb
    lo = (seed ^ XH) & MASK64
    hi = (lo + XL) & MASK64
    lo = ((lo ^ (lo >> 30)) * A) & MASK64
    hi = ((hi ^ (hi >> 30)) * A) & MASK64
    lo = ((lo ^ (lo >> 27)) * B) & MASK64
    hi = ((hi ^ (hi >> 27)) * B) & MASK64
    lo = (lo ^ (lo >> 31)) & MASK64
    hi = (hi ^ (hi >> 31)) & MASK64
    return lo, hi

# ── MD5 constants from variance_lut.h ────────────────────────────────────────
MD5_CONT_LO = 0x83886c9d0ae3a662
MD5_CONT_HI = 0xafa638a61b42e8ad
MD5_OCTAVE_IDX3 = [
    [0x082fe255f8be6631, 0x4e96119e22dedc81],
]

def lut_lookup(dy):
    fi = dy * (len(LUT) - 1)
    i = int(fi)
    frac = fi - i
    if i < 0:
        i, frac = 0, 0.0
    if i >= len(LUT) - 1:
        i, frac = len(LUT) - 2, 1.0
    return LUT[i] * (1.0 - frac) + LUT[i + 1] * frac

def compute_prefilter_score(seed):
    seed = seed & MASK64
    lo, hi = xSetSeed(seed)
    lo, hi, xlo = xNextLong(lo, hi)
    lo, hi, xhi = xNextLong(lo, hi)
    cont_lo = xlo ^ MD5_CONT_LO
    cont_hi = xhi ^ MD5_CONT_HI
    cont_lo, cont_hi, ixlo_A = xNextLong(cont_lo, cont_hi)
    cont_lo, cont_hi, ixhi_A = xNextLong(cont_lo, cont_hi)
    cont_lo, cont_hi, ixlo_B = xNextLong(cont_lo, cont_hi)
    cont_lo, cont_hi, ixhi_B = xNextLong(cont_lo, cont_hi)

    lo6 = ixlo_A ^ MD5_OCTAVE_IDX3[0][0]
    hi6 = ixhi_A ^ MD5_OCTAVE_IDX3[0][1]
    lo6, hi6, _ = xNextDouble(lo6, hi6)
    _, _, ob6_n = xNextLong(lo6, hi6)
    ob6 = (ob6_n >> 11) * (1.0 / (1 << 53)) * 256.0

    lo15 = ixlo_B ^ MD5_OCTAVE_IDX3[0][0]
    hi15 = ixhi_B ^ MD5_OCTAVE_IDX3[0][1]
    lo15, hi15, _ = xNextDouble(lo15, hi15)
    _, _, ob15_n = xNextLong(lo15, hi15)
    ob15 = (ob15_n >> 11) * (1.0 / (1 << 53)) * 256.0

    dy6 = ob6 - math.floor(ob6)
    dy15 = ob15 - math.floor(ob15)
    return CONT_AMP_SQ * (lut_lookup(dy6) + lut_lookup(dy15))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    islands = []
    with open('gpu/islands_4m.jsonl') as f:
        for line in f:
            islands.append(json.loads(line))

    print(f"Loaded {len(islands)} islands\n")

    scored = []
    for entry in islands:
        score = compute_prefilter_score(entry['seed'])
        scored.append((entry['area'], score))

    scored.sort(key=lambda x: x[0])
    all_scores = sorted([s for _, s in scored])

    # ── Percentile table ─────────────────────────────────────────────────
    print("Percentile table (all islands):")
    for p in range(0, 101, 1):
        idx = min(int(len(all_scores) * p / 100), len(all_scores) - 1)
        if p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
            print(f"  p{p:>3}: {all_scores[idx]:.6f}")

    # ── Size buckets ─────────────────────────────────────────────────────
    buckets = [
        ('4.0M+', 4_000_000, 5_000_000),
        ('4.5M+', 4_500_000, 5_000_000),
        ('5.0M+', 5_000_000, 6_000_000),
        ('5.5M+', 5_500_000, 6_000_000),
        ('6.0M+', 6_000_000, float('inf')),
    ]

    print(f"\n{'Bucket':>8}  {'N':>5}  {'Median':>8}  {'Mean':>8}  {'StdDev':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 72)
    for label, lo, hi in buckets:
        subset = [s for a, s in scored if lo <= a < hi]
        if not subset:
            print(f"{label:>8}  {'0':>5}")
            continue
        subset.sort()
        n = len(subset)
        median = subset[n // 2]
        mean = sum(subset) / n
        var = sum((x - mean) ** 2 for x in subset) / n
        std = var ** 0.5
        print(f"{label:>8}  {n:>5}  {median:>8.6f}  {mean:>8.6f}  {std:>8.6f}  {subset[0]:>8.6f}  {subset[-1]:>8.6f}")

    # ── Concentration analysis ───────────────────────────────────────────
    median_score = all_scores[len(all_scores) // 2]
    print(f"\nConcentration around median ({median_score:.6f}):")
    for delta in [0.000001, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001]:
        count = sum(1 for s in all_scores if abs(s - median_score) <= delta)
        pct = 100 * count / len(all_scores)
        print(f"  ±{delta:.6f} → {count:>6} / {len(all_scores)} ({pct:>5.1f}%)")

    # ── ASCII histogram (fine bins) ──────────────────────────────────────
    print("\nHistogram (0.00001-wide bins):")
    bin_counts = defaultdict(int)
    for s in all_scores:
        b = round(s / 0.00001) * 0.00001
        bin_counts[b] += 1

    max_count = max(bin_counts.values())
    bar_width = 60
    for b in sorted(bin_counts.keys()):
        c = bin_counts[b]
        bar = '#' * max(1, int(c / max_count * bar_width)) if c > 0 else ''
        if c >= 10:  # only show bins with 10+ entries
            print(f"  {b:.5f}  {c:>5}  {bar}")

    # ── Top 20 largest with scores ───────────────────────────────────────
    print(f"\nTop 20 largest islands:")
    by_area = sorted(scored, key=lambda x: -x[0])
    for area, score in by_area[:20]:
        print(f"  area={area:>8,}  score={score:.6f}")

    # ── PNG plot ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        scores_arr = np.array(all_scores)
        areas_arr = np.array([a for a, _ in scored])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Main histogram
        ax = axes[0, 0]
        bins = np.arange(0.0429, 0.04355, 0.00001)
        ax.hist(scores_arr, bins=bins, color='steelblue', edgecolor='black', linewidth=0.3)
        ax.axvline(median_score, color='red', linestyle='--', label=f'median={median_score:.6f}')
        ax.set_xlabel('Prefilter score')
        ax.set_ylabel('Count')
        ax.set_title(f'Score distribution (n={len(scores_arr):,})')
        ax.legend()
        ax.ticklabel_format(useOffset=False)

        # 2. Zoomed into the spike
        ax = axes[0, 1]
        bins_zoom = np.arange(0.04320, 0.04342, 0.000002)
        ax.hist(scores_arr, bins=bins_zoom, color='steelblue', edgecolor='black', linewidth=0.3)
        ax.axvline(median_score, color='red', linestyle='--', label=f'median')
        ax.set_xlabel('Prefilter score')
        ax.set_ylabel('Count')
        ax.set_title('Zoom: [0.04320, 0.04342)')
        ax.legend()
        ax.ticklabel_format(useOffset=False)

        # 3. Score vs area scatter — highlight 5.5M+ islands
        ax = axes[1, 0]
        by_score = sorted(scored, key=lambda x: x[1])
        # All islands (gray)
        sc_x = np.array([s for _, s in by_score])
        sc_y = np.array([a / 1e6 for a, _ in by_score])
        ax.scatter(sc_x, sc_y, s=1, alpha=0.15, c='gray')
        # 5.5M+ islands (red)
        big = [(a, s) for a, s in scored if a >= 5_500_000]
        big.sort(key=lambda x: x[1])
        bx = np.array([s for _, s in big])
        by = np.array([a / 1e6 for a, _ in big])
        ax.scatter(bx, by, s=15, alpha=0.8, c='red', zorder=5, label=f'5.5M+ (n={len(big)})')
        ax.axvline(median_score, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prefilter score')
        ax.set_ylabel('Area (M blocks²)')
        ax.set_title('Score vs Island Area')
        ax.ticklabel_format(useOffset=False)
        ax.legend()

        # 4. Score distribution for 5.5M+ islands vs all
        ax = axes[1, 1]
        all_s = [s for _, s in scored]
        big_s = [s for a, s in scored if a >= 5_500_000]
        bins_zoom = np.arange(0.0429, 0.04355, 0.00001)
        ax.hist(all_s, bins=bins_zoom, alpha=0.4, color='gray', label=f'All 4M+ (n={len(all_s):,})', density=True)
        ax.hist(big_s, bins=bins_zoom, alpha=0.8, color='red', label=f'5.5M+ (n={len(big_s)})', density=True)
        ax.axvline(median_score, color='black', linestyle='--', alpha=0.5, label='median')
        ax.set_xlabel('Prefilter score')
        ax.set_ylabel('Density')
        ax.set_title('Score distribution: 5.5M+ vs all')
        ax.legend()
        ax.ticklabel_format(useOffset=False)

        plt.tight_layout()
        plt.savefig('prefilter_score_analysis.png', dpi=150)
        print(f"\nSaved plot to prefilter_score_analysis.png")
    except ImportError:
        print("\nmatplotlib not available — skipping plot")

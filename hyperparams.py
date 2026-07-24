"""
hyperparams.py — All search parameters derived from target size S (blocks^2).

Theory:
  Island diameter D = √S
  Nyquist step  s = D/k  where k≥2 for 2×2 detection (need 2+ cells across island)
  Grid cells   G = fixed small number (8-24), chosen for GPU thread efficiency

  Total samples per seed:  B = T × W  (T threads/block, W wavefronts)
  Samples per grid:        g²
  Grids per seed:          N = B / g²
  Coverage per seed:       N × (g × s)² = B × s²

  Key insight: coverage grows with s² = S.
  Larger targets = larger step = more area per sample = more seeds scanned.

  The false positive filter: require a K×K block of cells ALL < -1.05.
  Random noise passes at rate ~(0.02)^(K²) per block — effectively zero for K≥3.
  Real islands of diameter D span cells across, so K=2 works when s ≤ D/2.
"""

import math

def derive(target_area: float = 10_000_000):
    """
    Derive all hyperparameters from target island area S (blocks^2 at 1:1).

    Returns a dict ready for the scanner.
    """
    S = target_area
    D = math.sqrt(S)           # characteristic diameter (blocks at 1:1)

    # ---- Detection guarantees ----
    # We need a K×K block of grid cells to all register as mushroom.
    # An island of diameter D, sampled with step s, spans ceil(D/s) cells.
    # We need ceil(D/s) ≥ K, so s ≤ D/K.
    # K=2 gives 2×2 detection (4 cells), works when s ≤ D/2.
    # K=3 gives 3×3 detection (9 cells), stricter false-positive filter.

    if S >= 10_000_000:
        K = 2   # huge island, 2×2 is sufficient
    elif S >= 1_000_000:
        K = 3   # medium, need 3×3 for low false positive
    else:
        K = 4   # small target, need 4×4

    s = D / K                  # grid step (blocks at 1:1)
    s = max(s, 4)              # don't go below 1:4 resolution
    s = round(s / 4) * 4       # align to 4-block boundaries

    # ---- GPU block layout ----
    T = 256                    # threads per block (fixed by kernel)
    W = 8                      # wavefronts per block
    total_samples = T * W      # 2048 samples per seed

    # ---- Grid cells per sparse patch ----
    # Small G = many patches, wide coverage. Large G = better stats per patch.
    # Pick G so each grid has 128-256 cells (half to one wavefront).
    G = 16                     # 16×16 = 256 cells per grid
    samples_per_grid = G * G   # 256

    num_grids = total_samples // samples_per_grid  # 8 grids per seed

    # ---- Coverage ----
    grid_width = G * s                        # blocks per grid side
    coverage_per_grid = grid_width ** 2       # = g²s²
    total_coverage = num_grids * coverage_per_grid  # = B × s²

    # ---- Grid placement ----
    # Place grids on a coarse lattice so they don't overlap much.
    # Spacing between grid centers = grid_width × 1.5 (some overlap for safety)
    grid_spacing = int(grid_width * 1.5)
    grid_offsets = []
    cols = int(math.sqrt(num_grids))
    for i in range(num_grids):
        gx = (i % cols) * grid_spacing
        gz = (i // cols) * grid_spacing
        grid_offsets.append((gx, gz))

    # ---- False positive rate (estimate) ----
    # P(single cell < -1.05) ≈ 0.02 (empirical)
    # P(K×K block all < -1.05) ≈ 0.02^(K²) per block (naive, ignores correlation)
    # Blocks per grid: (G-K+1)²
    # False positive per seed ≈ num_grids × (G-K+1)² × 0.02^(K²)
    blocks_per_grid = (G - K + 1) ** 2
    fp_per_block = 0.02 ** (K * K)
    fp_per_seed = num_grids * blocks_per_grid * fp_per_block

    # ---- Required seeds to find one ----
    # If island probability per seed is p, need ~4.6/p seeds for 99% confidence.
    # p is unknown — must be estimated empirically.

    return {
        'target_area': S,
        'diameter': D,
        'step': s,
        'kernel_size': K,
        'threads_per_block': T,
        'wavefronts': W,
        'total_samples': total_samples,
        'grid_cells': G,
        'samples_per_grid': samples_per_grid,
        'num_grids': num_grids,
        'grid_width': grid_width,
        'coverage_per_grid': coverage_per_grid,
        'total_coverage': total_coverage,
        'grid_offsets': grid_offsets,
        'false_positive_per_seed': fp_per_seed,
    }


def print_params(p: dict):
    """Human-readable parameter summary."""
    S = p['target_area']
    print(f"Target: {S:,.0f} blocks^2")
    print(f"  Diameter:              {p['diameter']:,.0f} blocks")
    print(f"  Step:                  {p['step']} blocks")
    print(f"  Detection kernel:      {p['kernel_size']}×{p['kernel_size']}")
    print(f"  Grid:                  {p['grid_cells']}×{p['grid_cells']} cells")
    print(f"  Samples per grid:      {p['samples_per_grid']}")
    print(f"  Grids per seed:        {p['num_grids']}")
    print(f"  Grid coverage:         {p['grid_width']/1000:,.1f}K × "
          f"{p['grid_width']/1000:,.1f}K blocks per grid")
    print(f"  Total coverage/seed:   {p['total_coverage']/1e9:,.1f}B blocks^2")
    print(f"  Total samples/seed:    {p['total_samples']}")
    print(f"  False + per seed:      {p['false_positive_per_seed']:.2e}")

if __name__ == '__main__':
    for S in [100_000, 1_000_000, 10_000_000, 100_000_000]:
        print_params(derive(S))
        print()

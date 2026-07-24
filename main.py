#!/usr/bin/env python3
"""main.py — Largest Mushroom Island Finder."""

import sys, os, time, json, argparse, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gpu.scanner import MushroomScanner
from hyperparams import derive


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--time', type=int, default=60)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--target', type=float, default=10_000_000)
    p.add_argument('--batch', type=int, default=50000)
    p.add_argument('--refine-top', type=int, default=100)
    args = p.parse_args()

    hp = derive(args.target)
    D = hp['diameter']
    # For CPU: use a grid that covers enough area to find D-sized islands
    # Grid step = D/2 (Nyquist), coverage = 4× island area
    grid = max(32, int(math.sqrt(args.target) / 100))
    radius = int(D * 2)  # Scan ±2 diameters from origin

    print(f"Target: {args.target:,.0f} blocks^2 (diameter {D:,.0f})")
    print(f"Grid: {grid}x{grid}, step={2*radius//grid}, radius={radius}")
    print(f"Workers: {args.workers}, batch: {args.batch}")

    scanner = MushroomScanner(grid=grid, radius=radius, workers=args.workers)
    deadline = time.time() + args.time
    best, scanned = None, 0

    print(f"\nRunning {args.time}s...")
    try:
        while time.time() < deadline:
            seeds = np.random.randint(0, 2**63, size=args.batch, dtype=np.uint64)
            t0 = time.perf_counter()
            cands = scanner.scan_seeds(seeds=seeds)
            scanned += args.batch

            if cands:
                cands.sort(key=lambda c: c['min_continentalness'])
                refined = scanner.refine_candidates(cands[:args.refine_top])
                for r in (refined or []):
                    if best is None or r['area'] > best['area']:
                        best = r
                        print(f"  NEW BEST: seed {r['seed']}, "
                              f"flood fill {r['area']:,} blocks^2 "
                              f"({r['cells']} cells) at {r['center']}")
                        with open('best.json', 'w') as f:
                            json.dump(best, f, indent=2)

            dt = time.perf_counter() - t0
            remaining = max(0, deadline - time.time())
            best_area = best['area'] if best else 0
            print(f"  {scanned:,} seeds, {args.batch/dt:.0f}/s, "
                  f"best {best_area:,}, {remaining:.0f}s left")

    except KeyboardInterrupt:
        pass

    print(f"\n{'='*60}")
    print(f"Scanned: {scanned:,} seeds")
    if best:
        print(f"BEST: seed {best['seed']}")
        print(f"  Area: {best['area']:,} blocks^2 ({best['cells']} cells at 1:4)")
        print(f"  Center: {best['center']}")

if __name__ == '__main__':
    main()

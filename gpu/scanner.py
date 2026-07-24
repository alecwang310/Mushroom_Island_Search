"""
gpu/scanner.py — Mushroom island scanner.

Currently: CPU multi-threaded (C engine + ThreadPoolExecutor).
GPU upgrade: CUDA kernel is compiled (perlin_kernel.o), wiring TBD.

Usage:
    scanner = MushroomScanner(grid=64, radius=4096)
    candidates = scanner.scan_seeds(start=0, count=100000)
    for c in candidates:
        island = scanner.refine(c['seed'], c['sample_coords'])
"""

import os, time, math, ctypes
from concurrent.futures import ThreadPoolExecutor
import numpy as np

_engine_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'engine')
_lib = ctypes.CDLL(os.path.join(_engine_dir, 'continentalness.dll'))

_CONT_ENGINE_SIZE = 8192

_lib.cont_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
_lib.cont_engine_init.restype = None
_lib.cont_sample.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.cont_sample.restype = ctypes.c_double
_lib.cont_sample_grid.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.cont_sample_grid.restype = None


class MushroomScanner:
    """Multi-threaded mushroom island scanner with CPU refinement."""

    def __init__(self, grid: int = 64, radius: int = 4096, workers: int = 16):
        self.grid = grid                    # grid points per axis
        self.radius = radius                # search radius in blocks
        self.step = (radius * 2) // grid    # spacing between samples
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)

        # Stats
        self.seeds_scanned = 0
        self.candidates_found = 0
        self.best_island = None

    def _init_engine(self, seed: int) -> bytes:
        """Create a C ContEngine for one seed."""
        buf = (ctypes.c_ubyte * _CONT_ENGINE_SIZE)()
        _lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
        return bytes(buf)

    def _scan_one_seed(self, seed: int) -> dict | None:
        """Scan one seed at the coarse grid. Returns None if no mushroom found."""
        buf = (ctypes.c_ubyte * _CONT_ENGINE_SIZE)()
        _lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)

        gs = self.grid
        grid = np.empty((gs, gs), dtype=np.float32)
        _lib.cont_sample_grid(
            ctypes.cast(buf, ctypes.c_void_p),
            grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            -self.radius, -self.radius, gs, gs, self.step)

        # Mushroom: guaranteed in [-1.2, -1.05], but nearest-neighbor in 6D
        # climate space means continentalness < -1.05 is almost always mushroom.
        # (No other biome has continentalness this low — mushroom_fields is the
        # closest biome in 6D Euclidean distance.)
        ys, xs = np.where(grid < -1.05)
        if len(ys) == 0:
            return None

        # Return candidate info
        hit_coords = []
        for y, x in zip(ys[:20], xs[:20]):
            bx = -self.radius + int(x) * self.step
            bz = -self.radius + int(y) * self.step
            hit_coords.append((bx, bz))

        return {
            'seed': seed,
            'hit_count': len(ys),
            'min_continentalness': float(grid.min()),
            'sample_coords': hit_coords,
        }

    def scan_seeds(self, start: int = 0, count: int = 0,
                   seeds: np.ndarray | None = None) -> list[dict]:
        """Scan many seeds. Specify either start+count or seeds array."""
        if seeds is None:
            seeds = np.arange(start, start + count, dtype=np.uint64)
        n = len(seeds)

        t0 = time.perf_counter()
        futures = [self.pool.submit(self._scan_one_seed, int(s)) for s in seeds]
        results = []
        for i, f in enumerate(futures):
            r = f.result()
            if r:
                results.append(r)
            if i % 10000 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i:,}/{n:,} seeds ({i/elapsed:.0f}/sec), "
                      f"{len(results)} candidates")

        self.seeds_scanned += n
        self.candidates_found += len(results)
        elapsed = time.perf_counter() - t0
        print(f"  Done: {n:,} seeds in {elapsed:.1f}s ({n/elapsed:.0f}/sec), "
              f"{len(results)} candidates ({100*len(results)/n:.2f}%)")
        return results

    def refine_island(self, seed: int, start_coords: list[tuple]) -> dict | None:
        """
        Refine a candidate seed: find the lowest continentalness point as center,
        ray-march for bounding box, then flood fill for exact area.

        Returns: {'seed', 'area', 'bounds', 'center', 'cells'}
        """
        buf = (ctypes.c_ubyte * _CONT_ENGINE_SIZE)()
        _lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
        sampler = lambda x, z: _lib.cont_sample(buf, x, z)

        # ---- Step 1: find deepest point (lowest continentalness) ----
        best_x = sum(c[0] for c in start_coords) // len(start_coords)
        best_z = sum(c[1] for c in start_coords) // len(start_coords)
        best_val = sampler(best_x, best_z)

        # Search locally for a deeper point
        for dx, dz in [(0, 0), (64, 0), (-64, 0), (0, 64), (0, -64),
                       (128, 0), (-128, 0), (0, 128), (0, -128),
                       (192, 0), (-192, 0), (0, 192), (0, -192)]:
            v = sampler(best_x + dx, best_z + dz)
            if v < best_val:
                best_x += dx; best_z += dz
                best_val = v

        if not (best_val < -1.05):
            return None

        # ---- Step 2: fast ray-march for bounding box estimate ----
        min_x, max_x = best_x, best_x
        min_z, max_z = best_z, best_z
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            for dist in range(4, 5000, 16):
                sx = int(best_x + dist * math.cos(rad))
                sz = int(best_z + dist * math.sin(rad))
                if sampler(sx, sz) >= -1.05:
                    break
                if sx < min_x: min_x = sx
                if sx > max_x: max_x = sx
                if sz < min_z: min_z = sz
                if sz > max_z: max_z = sz

        est_area = (max_x - min_x) * (max_z - min_z)

        # ---- Step 3: flood fill for exact area (only if promising) ----
        exact_area, cells = self._flood_fill_island(
            sampler, best_x, best_z, max_area=1_000_000)

        area = exact_area if exact_area > 0 else est_area
        return {
            'seed': seed,
            'area': area,
            'area_estimate': est_area,
            'bounds': (min_x, min_z, max_x, max_z),
            'center': (best_x, best_z),
            'cells': cells,
        }

    def _flood_fill_island(self, sampler, start_x: int, start_z: int,
                           max_area: int = 1_000_000,
                           resolution: int = 4) -> tuple[int, int]:
        """
        BFS flood fill at 1:4 resolution to find exact mushroom island area.
        Returns (area_in_blocks, cell_count).
        Each cell is resolution×resolution blocks at 1:1 scale.
        """
        from collections import deque

        # Align start to resolution grid
        sx = (start_x // resolution) * resolution
        sz = (start_z // resolution) * resolution

        if sampler(sx, sz) >= -1.05:
            return 0, 0

        queue = deque([(sx, sz)])
        visited = {(sx, sz)}
        cache = { (sx, sz): sampler(sx, sz) }
        cells = 0

        while queue:
            x, z = queue.popleft()
            cells += 1

            if cells * resolution * resolution > max_area:
                break  # sanity cap

            for dx, dz in [(resolution, 0), (-resolution, 0),
                           (0, resolution), (0, -resolution)]:
                nx, nz = x + dx, z + dz
                if (nx, nz) in visited:
                    continue
                visited.add((nx, nz))

                # Cache continentalness values
                key = (nx, nz)
                if key not in cache:
                    cache[key] = sampler(nx, nz)

                if cache[key] < -1.05:
                    queue.append((nx, nz))

        block_area = cells * resolution * resolution
        return block_area, cells

    def refine_candidates(self, candidates: list[dict]) -> list[dict]:
        """Refine a list of candidate dicts (from scan_seeds) to find island boundaries."""
        if not candidates:
            return []

        print(f"\nRefining {len(candidates)} candidates...")
        refined = []
        for i, c in enumerate(candidates):
            island = self.refine_island(c['seed'], c['sample_coords'])
            if island:
                refined.append(island)
                if self.best_island is None or island['area'] > self.best_island['area']:
                    self.best_island = island
                    cells = island.get('cells', 0)
                    print(f"  NEW BEST: seed {island['seed']}, "
                          f"flood fill {island['area']:,} blocks^2 "
                          f"({cells} cells at 1:4), "
                          f"estimate {island.get('area_estimate', 0):,}, "
                          f"center {island['center']}")
            if i % 50 == 0 and i > 0:
                print(f"  refined {i}/{len(candidates)}...")
        return refined

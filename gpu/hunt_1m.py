"""hunt_1m.py — GPU hunt for large mushroom islands using C++ engine.

GPU scans seeds as fast as possible, dumping hits to a queue.
A thread pool flood-fills in parallel — GPU never waits on CPU.
"""
import sys, os, time, ctypes, json, math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

# ------- Load C++ hunt engine DLL -------
_hunt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'hunt_engine.dll'))
_hunt.hunt_batch.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
]
_hunt.hunt_batch.restype = ctypes.c_int
_hunt.hunt_cleanup.argtypes = []
_hunt.hunt_cleanup.restype = None


def hunt_batch(start_seed: int, n: int, step: int, K: int, G: int):
    results = (ctypes.c_int64 * (n * 3))()
    hc = _hunt.hunt_batch(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_int(step), ctypes.c_int(K), ctypes.c_int(G), results)
    return [(int(results[i*3]), int(results[i*3+1]), int(results[i*3+2]))
            for i in range(hc)]


def flood_fill(seed: int, cx: int, cz: int,
               max_cells: int = 10_000_000) -> dict | None:
    """C-side flood fill at 1:4 scale. Returns dict with area + center, or None."""
    area = _eng._lib.cont_flood_fill(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(cx), ctypes.c_int(cz),
        ctypes.c_int(max_cells))
    if area <= 0:
        return None
    return {
        'seed': seed,
        'area': area,
        'center_1_4': (cx, cz),
        'center_1_1': (cx * 4, cz * 4),
    }


if __name__ == '__main__':
    # ---- Config ----
    TARGET = 2_000_000
    S = 5_000_000
    D = math.sqrt(S)
    step = int(D / 4 / 3) // 4 * 4
    step = max(step, 4)
    K = 2
    G = 256
    batch = 2048
    START_SEED = 77777
    FF_WORKERS = 8                     # parallel flood fill threads

    print(f'Target: >= {TARGET:,} blocks^2')
    print(f'step={step}  K={K}  G={G}  batch={batch}  ff_workers={FF_WORKERS}')
    print(f'Grid: {(G-1)*step}x{(G-1)*step} blocks at 1:1')
    print(f'Seeds: sequential from {START_SEED}')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned = 0
    hits_total = 0
    big_count = 0
    next_seed = START_SEED
    lock = threading.Lock()
    t0 = time.perf_counter()

    def on_done(future):
        """Callback: runs in worker thread. Log if big enough."""
        global best, big_count
        result = future.result()
        if result is None or result['area'] < TARGET:
            return
        with lock:
            big_count += 1
            with open('big_islands.jsonl', 'a') as f:
                f.write(json.dumps(result) + '\n')
            print(f'\n  BIG ({result["area"]:,}): seed {result["seed"]} '
                  f'at 1:4 ({result["center_1_4"][0]},{result["center_1_4"][1]})')
            if best is None or result['area'] > best['area']:
                best = result
                print(f'  *** NEW BEST: seed {result["seed"]}, '
                      f'{result["area"]:,} blocks^2 ***')
                with open('best.json', 'w') as f:
                    json.dump(best, f)

    try:
        while True:
            hits = hunt_batch(next_seed, batch, step, K, G)
            scanned += batch
            hits_total += len(hits)
            next_seed += batch

            # Submit all hits to the thread pool — non-blocking
            for seed, hx, hz in hits:
                f = pool.submit(flood_fill, seed, hx, hz)
                f.add_done_callback(on_done)

            elapsed = time.perf_counter() - t0
            rate = scanned / elapsed
            if scanned % (batch * 10) == 0:
                ba = best['area'] if best else 0
                print(f'\r  {scanned:,} seeds ({rate:,.0f}/s) | '
                      f'{hits_total} hits | {big_count} big | best {ba:,}  ',
                      end='')

    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down thread pool...')
        pool.shutdown(wait=True)
        _hunt.hunt_cleanup()

    elapsed = time.perf_counter() - t0
    print(f'\nScanned {scanned:,} seeds in {elapsed:.0f}s ({scanned/elapsed:,.0f}/s)')
    print(f'{hits_total} GPU hits, {big_count} islands >= {TARGET:,}')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} blocks^2 '
              f'at {best["center_1_4"]}')

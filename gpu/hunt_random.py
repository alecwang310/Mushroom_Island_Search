"""hunt_random.py — Hunt 5M+ islands with random seeds.

Random start per batch, K=2, targeting >=5M block^2 islands.
Logs seed + area + center to islands_2m.jsonl for later octave analysis.
"""
import sys, os, time, ctypes, json, math, random
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

_hunt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'hunt_engine.dll'))
_hunt.hunt_batch.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
]
_hunt.hunt_batch.restype = ctypes.c_int
_hunt.hunt_cleanup.argtypes = []
_hunt.hunt_cleanup.restype = None


def hunt_batch(start_seed, n, step, K, G):
    results = (ctypes.c_int64 * (n * 3))()
    hc = _hunt.hunt_batch(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_int(step), ctypes.c_int(K), ctypes.c_int(G), results)
    return [(int(results[i*3]), int(results[i*3+1]), int(results[i*3+2]))
            for i in range(hc)]


def flood_fill(seed, cx, cz, max_cells=10_000_000):
    area = _eng._lib.cont_flood_fill(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(cx), ctypes.c_int(cz),
        ctypes.c_int(max_cells))
    if area <= 0:
        return None
    return {'seed': seed, 'area': area, 'cx': cx, 'cz': cz}


if __name__ == '__main__':
    TARGET = 2_000_000
    S = 5_000_000
    D = math.sqrt(S)
    step = int(D / 4 / 3) // 4 * 4
    step = max(step, 4)
    K, G, batch = 2, 256, 2048
    FF_WORKERS = 12

    print(f'Target: >= {TARGET:,} blocks^2')
    print(f'step={step}  K={K}  G={G}  batch={batch}  workers={FF_WORKERS}')
    print(f'Grid: {(G-1)*step:,}x{(G-1)*step:,} blocks at 1:1')
    print(f'Random start per batch (sequential within batch, decorrelated via Xoroshiro)')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, hits_gpu, hits_big = 0, 0, 0
    lock = threading.Lock()
    t0 = time.perf_counter()

    def on_done(future):
        global best, hits_big
        r = future.result()
        if r is None or r['area'] < TARGET:
            return
        with lock:
            hits_big += 1
            entry = {'seed': r['seed'], 'area': r['area'], 'cx': r['cx'], 'cz': r['cz']}
            with open('islands_2m.jsonl', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f'\n  BIG ({r["area"]:,}): seed {r["seed"]} at ({r["cx"]},{r["cz"]})')
            if best is None or r['area'] > best['area']:
                best = r
                print(f'  *** NEW BEST: seed {r["seed"]}, {r["area"]:,} ***')
                with open('best.json', 'w') as f:
                    json.dump(entry, f)

    try:
        while True:
            start = random.getrandbits(64)
            hits = hunt_batch(start, batch, step, K, G)
            scanned += batch
            hits_gpu += len(hits)

            for seed, gx, gz in hits:
                pool.submit(flood_fill, seed, gx, gz).add_done_callback(on_done)

            if scanned % (batch * 10) == 0:
                ba = best['area'] if best else 0
                elapsed = time.perf_counter() - t0
                print(f'\r  {scanned:,} seeds ({scanned/elapsed:,.0f}/s) | '
                      f'{hits_gpu} hits | {hits_big} big | best {ba:,}  ',
                      end='', flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down...')
        pool.shutdown(wait=True)
        _hunt.hunt_cleanup()

    elapsed = time.perf_counter() - t0
    print(f'\nScanned {scanned:,} seeds in {elapsed:.0f}s ({scanned/elapsed:,.0f}/s)')
    print(f'{hits_gpu} GPU hits -> {hits_big} islands >= {TARGET:,}')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} blocks^2 at ({best["cx"]},{best["cz"]})')

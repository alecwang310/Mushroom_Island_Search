"""hunt_tiered.py — GPU tiered hunt: 2x coarse + adjacency + CPU verify + flood fill.

step_1x=140, step_2x=280. GPU finds adjacent pairs on 2x grid.
CPU verifies with 13-point 1x sampling. Only verified hits get flood filled.
"""
import sys, os, time, ctypes, json, math, random
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

_hunt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'hunt_engine.dll'))
_hunt.hunt_batch_tiered.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),           # hit_counts_out
    ctypes.POINTER(ctypes.c_int64),         # hit_results
]
_hunt.hunt_batch_tiered.restype = ctypes.c_int
_hunt.hunt_cleanup.argtypes = []
_hunt.hunt_cleanup.restype = None

MAX_HITS = 256  # matches MAX_HITS_PER_SEED in hunt_engine.cu

def hunt_batch_tiered(start_seed, n, step_2x, G):
    counts = (ctypes.c_int * n)()
    results = (ctypes.c_int64 * (n * MAX_HITS * 3))()
    hc = _hunt.hunt_batch_tiered(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_int(step_2x), ctypes.c_int(1), ctypes.c_int(G),
        counts, results)
    return [(int(results[i*3]), int(results[i*3+1]), int(results[i*3+2]))
            for i in range(hc)]


def verify_pair_cpu(seed, gx, gz, step_1x, step_2x):
    """CPU: sample 13 new 1x points around an adjacent pair on the 2x grid.
    Checks both vertical and horizontal pair directions. Returns True if
    any 2x2 block at 1x has >= 3 of 4 cells < -1.05 (lenient threshold)."""
    e = _eng.ContEngine(seed)
    S1, S2 = float(step_1x), float(step_2x)
    T = -1.05

    def passes(v00, v10, v01, v11):
        return sum(1 for v in (v00,v10,v01,v11) if v < T) >= 3

    # New 1x points for vertical pair (anchor at top):
    #   (-S1,-S1)(0,-S1)(S1,-S1)  (-S1,0)(S1,0)  (-S1,S1)(0,S1)(S1,S1)
    #   (-S1,S2)(S1,S2)  (-S1,S2+S1)(0,S2+S1)(S1,S2+S1)
    vx_v = [-S1, 0, S1, -S1, S1, -S1, 0, S1, -S1, S1, -S1, 0, S1]
    vz_v = [-S1, -S1, -S1, 0, 0, S1, S1, S1, S2, S2, S2+S1, S2+S1, S2+S1]
    vv = [e.sample(int(gx+x), int(gz+z)) for x, z in zip(vx_v, vz_v)]
    # Known: K0 at (0,0) and K1 at (0,S2) are both < threshold (from GPU adjacency)
    K0, K1 = -2.0, -2.0  # treated as passing
    # Check 8 2x2 blocks in 3x5 grid
    g = [vv[0], vv[1], vv[2],  vv[3], K0, vv[4],  vv[5], vv[6], vv[7],
         vv[8], K1, vv[9],  vv[10], vv[11], vv[12]]
    if passes(g[0],g[1],g[3],g[4]) or passes(g[1],g[2],g[4],g[5]) or \
       passes(g[3],g[4],g[6],g[7]) or passes(g[4],g[5],g[7],g[8]) or \
       passes(g[6],g[7],g[9],g[10]) or passes(g[7],g[8],g[10],g[11]) or \
       passes(g[9],g[10],g[12],g[13]) or passes(g[10],g[11],g[13],g[14]):
        return True

    # New 1x points for horizontal pair (anchor at left):
    #   (-S1,-S1)(0,-S1)(S1,-S1)(S2,-S1)(S2+S1,-S1)
    #   (-S1,0)(S1,0)(S2+S1,0)
    #   (-S1,S1)(0,S1)(S1,S1)(S2,S1)(S2+S1,S1)
    vx_h = [-S1, 0, S1, S2, S2+S1,  -S1, S1, S2+S1,  -S1, 0, S1, S2, S2+S1]
    vz_h = [-S1, -S1, -S1, -S1, -S1,  0, 0, 0,  S1, S1, S1, S1, S1]
    vh = [e.sample(int(gx+x), int(gz+z)) for x, z in zip(vx_h, vz_h)]
    g2 = [vh[0], vh[1], vh[2], vh[3], vh[4],  vh[5], K0, vh[6], K1, vh[7],
          vh[8], vh[9], vh[10], vh[11], vh[12]]
    if passes(g2[0],g2[1],g2[5],g2[6]) or passes(g2[1],g2[2],g2[6],g2[7]) or \
       passes(g2[2],g2[3],g2[7],g2[8]) or passes(g2[3],g2[4],g2[8],g2[9]) or \
       passes(g2[5],g2[6],g2[10],g2[11]) or passes(g2[6],g2[7],g2[11],g2[12]) or \
       passes(g2[7],g2[8],g2[12],g2[13]) or passes(g2[8],g2[9],g2[13],g2[14]):
        return True

    return False


def flood_fill(seed, cx, cz, max_cells=10_000_000):
    area = _eng._lib.cont_flood_fill(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(cx), ctypes.c_int(cz),
        ctypes.c_int(max_cells))
    if area <= 0:
        return None
    return {'seed': seed, 'area': area, 'cx': cx, 'cz': cz}


def verify_and_flood(seed, gx, gz, step_1x, step_2x):
    t0 = time.perf_counter()
    ok = verify_pair_cpu(seed, gx, gz, step_1x, step_2x)
    t_vfy = time.perf_counter() - t0
    if not ok:
        return (False, None, t_vfy, 0)
    t0 = time.perf_counter()
    ff = flood_fill(seed, gx, gz)
    t_ff = time.perf_counter() - t0
    return (True, ff, t_vfy, t_ff)


if __name__ == '__main__':
    TARGET = 2_500_000
    step_1x = 140
    step_2x = 280
    G, batch = 512, 2048
    FF_WORKERS = 28

    print(f'Target: >= {TARGET:,} blocks^2')
    print(f'step_1x={step_1x}  step_2x={step_2x}  G={G}  batch={batch}')
    print(f'Coarse grid: {(G-1)*step_2x:,}x{(G-1)*step_2x:,} blocks at 1:1')
    print(f'GPU: 2x coarse + adjacent pair. CPU: 13-point verify + flood fill.')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, hits_gpu, hits_verified, hits_ok, hits_big = 0, 0, 0, 0, 0
    t_gpu, t_vfy, t_ff = 0.0, 0.0, 0.0
    seen = set()  # dedup by (seed, area)
    lock = threading.Lock()
    t0 = time.perf_counter()
    running = True

    def on_done(future):
        global best, hits_ok, hits_big, hits_verified, t_vfy, t_ff
        verified, r, tv, tf = future.result()
        with lock:
            if verified:
                hits_verified += 1
            t_vfy += tv
            t_ff += tf
        if r is None:
            return
        with lock:
            hits_ok += 1
        if r['area'] < TARGET:
            return
        with lock:
            key = (r['seed'], r['area'])
            if key in seen:
                return
            seen.add(key)
            hits_big += 1
            entry = {'seed': r['seed'], 'area': r['area'], 'cx': r['cx'], 'cz': r['cz']}
            with open('islands_25m.jsonl', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f'\n  BIG ({r["area"]:,}): seed {r["seed"]} at ({r["cx"]},{r["cz"]})')
            if best is None or r['area'] > best['area']:
                best = r
                print(f'  *** NEW BEST: {r["seed"]}, {r["area"]:,} ***')
                with open('best_25m.jsonl', 'w') as f:
                    json.dump(entry, f)

    def gpu_thread():
        global scanned, hits_gpu, t_gpu
        while running:
            start = random.getrandbits(64)
            t1 = time.perf_counter()
            hits = hunt_batch_tiered(start, batch, step_2x, G)
            t2 = time.perf_counter()
            with lock:
                scanned += batch
                hits_gpu += len(hits)
                t_gpu += (t2 - t1)
            for seed, gx, gz in hits:
                pool.submit(verify_and_flood, seed, gx, gz,
                           step_1x, step_2x).add_done_callback(on_done)

    gpu_t = threading.Thread(target=gpu_thread, daemon=True)
    gpu_t.start()

    try:
        while True:
            time.sleep(2)
            if scanned > 0:
                ba = best['area'] if best else 0
                vrate = 100 * hits_verified / max(hits_gpu, 1)
                elapsed = time.perf_counter() - t0
                tg = t_gpu / max(scanned / batch, 1)
                tv = t_vfy * 1e6 / max(hits_gpu, 1)
                tf = t_ff * 1e6 / max(hits_verified, 1)
                print(f'\r  {scanned:,} seeds ({scanned/elapsed:,.0f}/s) | '
                      f'GPU {tg:.1f}s/batch | '
                      f'vfy {tv:.0f}us | ff {tf:.0f}us | '
                      f'{hits_verified} ok ({vrate:.1f}%) | '
                      f'{hits_big} big | best {ba:,}  ', end='', flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down...')
        running = False
        gpu_t.join(timeout=10)
        pool.shutdown(wait=True)
        _hunt.hunt_cleanup()

    elapsed = time.perf_counter() - t0
    print(f'\nScanned {scanned:,} seeds in {elapsed:.0f}s ({scanned/elapsed:,.0f}/s)')
    print(f'Timing: GPU {t_gpu/max(scanned/batch,1):.3f}s/batch | '
          f'vfy {t_vfy*1e6/max(hits_gpu,1):.0f}us/hit | '
          f'ff {t_ff*1e6/max(hits_verified,1):.0f}us/ff')
    print(f'{hits_gpu} GPU pairs -> {hits_verified} verified -> {hits_big} big (>= {TARGET:,})')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} at ({best["cx"]},{best["cz"]})')

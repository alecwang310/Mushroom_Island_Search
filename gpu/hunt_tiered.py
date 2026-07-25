"""hunt_tiered.py — Hex grid O6+O15 GPU prefilter + CPU verify + flood fill.

GPU: hex grid (staggered rows, 280-block spacing), only O6+O15 octaves,
     threshold -1.0, 6-neighbor adjacent pair detection. ~20K seeds/s.
CPU: 5-point hex verification with all cont octaves (6-23, no shift).
     Any hex point < -1.0 triggers full 24-octave cont_flood_fill.
     Logs >=3M block^2 islands to islands_3m.jsonl.
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

MAX_HITS = 512  # plenty for G=512

def hunt_batch_tiered(start_seed, n, step_2x, G):
    counts = (ctypes.c_int * n)()
    results = (ctypes.c_int64 * (n * MAX_HITS * 3))()
    hc = _hunt.hunt_batch_tiered(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_int(step_2x), ctypes.c_int(1), ctypes.c_int(G),
        counts, results)
    return [(int(results[i*3]), int(results[i*3+1]), int(results[i*3+2]))
            for i in range(hc)]


def verify_pair_cpu(seed, gx, gz, step_1x, step_2x, engine_buf=None):
    """Hex verification with all cont octaves (6-23, no shifts), threshold -1.0.
    5 new hex points. If ANY one is < -1.0, trigger full flood fill."""
    import ctypes, struct
    buf = (ctypes.c_ubyte * 8192)()
    _eng._lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
    off_amp = 24*257 + 24*8*3
    # Zero only shift octaves (0-5), keep all cont (6-23)
    for j in range(6):
        struct.pack_into('d', buf, off_amp + j*8, 0.0)

    def s(x, z):
        return _eng._lib.cont_sample(buf, x, z)

    S1, S2 = float(step_1x), float(step_2x)
    T = -1.00
    rt32 = 0.8660254037844386

    # Vertical pair: A=(gx,gz), B=(gx,gz+S2). Center C=(gx, gz+S1)
    pv = [
        s(gx, int(gz+S1)),                                         # center
        s(int(gx+S1*rt32), int(gz+0.5*S1)),                        # V1
        s(int(gx+S1*rt32), int(gz+1.5*S1)),                        # V2
        s(int(gx-S1*rt32), int(gz+1.5*S1)),                        # V4
        s(int(gx-S1*rt32), int(gz+0.5*S1)),                        # V5
    ]
    if any(v < T for v in pv):
        return True

    # Horizontal pair: A=(gx,gz), B=(gx+S2,gz). Center C=(gx+S1, gz)
    ph = [
        s(int(gx+S1), gz),                                         # center
        s(int(gx+0.5*S1), int(gz-S1*rt32)),
        s(int(gx+1.5*S1), int(gz-S1*rt32)),
        s(int(gx+1.5*S1), int(gz+S1*rt32)),
        s(int(gx+0.5*S1), int(gz+S1*rt32)),
    ]
    if any(v < T for v in ph):
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
    ok = verify_pair_cpu(seed, gx, gz, step_1x, step_2x)  # O6+O15 only, -0.95
    t_vfy = time.perf_counter() - t0
    if not ok:
        return (False, None, t_vfy, 0)
    t0 = time.perf_counter()
    ff = flood_fill(seed, gx, gz)  # full C engine
    t_ff = time.perf_counter() - t0
    return (True, ff, t_vfy, t_ff)


if __name__ == '__main__':
    TARGET = 3_000_000
    step_1x = 140
    step_2x = 280
    G, batch = 512, 8192
    FF_WORKERS = 16

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
            with open('islands_3m.jsonl', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f'\n  BIG ({r["area"]:,}): seed {r["seed"]} at ({r["cx"]},{r["cz"]})')
            if best is None or r['area'] > best['area']:
                best = r
                print(f'  *** NEW BEST: {r["seed"]}, {r["area"]:,} ***')
                with open('best_3m.jsonl', 'w') as f:
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

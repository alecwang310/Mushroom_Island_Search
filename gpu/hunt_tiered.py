"""hunt_tiered.py — Hex grid O6+O15 GPU prefilter + CPU verify + flood fill.

GPU: hex grid (staggered rows, 280-block spacing), only O6+O15 octaves,
     threshold -1.0, 6-neighbor adjacent pair detection. ~20K seeds/s.
     Optional pre-filter: GPU LUT-variance filter generates a consecutive
     seed range on-device and returns only compacted survivors.

CPU: 5-point hex verification with all cont octaves (6-23, no shift).
     Any hex point < -1.0 triggers full 24-octave cont_flood_fill.
     Logs >=3M block^2 islands to islands_3m.jsonl.
"""
import sys, os, time, ctypes, json, random
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

GPU_DIR = os.path.dirname(__file__)
_hunt = ctypes.CDLL(os.path.join(GPU_DIR, 'hunt_engine.dll'))

_hunt.hunt_batch_tiered.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
_hunt.hunt_batch_tiered.restype = ctypes.c_int

_hunt.prefilter_range.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
]
_hunt.prefilter_range.restype = ctypes.c_int

_hunt.tiered_scan_mem.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
_hunt.tiered_scan_mem.restype = ctypes.c_int

_hunt.hunt_cleanup.argtypes = []
_hunt.hunt_cleanup.restype = None

INITIAL_HIT_CAP = 65_536

# ── Pre-filter config ──────────────────────────────────────────────────
# p99.15-p99.25 band: score [0.04330, 0.04331]
# Keeps seeds BETWEEN these scores (0.1% of randoms)
# Lower rejects noise, upper rejects score-ceiling randoms
PREFT_LO = 0.0429   # captures 65% of 4M+ in [0.0429, 0.0433]
PREFT_HI = 0.0433   # 23× enrichment, ~0.95% random pass
PREFT_ENABLED = True
PREF_BATCH = 100_000_000
PREF_SURVIVOR_CAP = 2_000_000


def hunt_batch_tiered(start_seed, n, step_2x, G):
    hit_capacity = INITIAL_HIT_CAP
    while True:
        results = (ctypes.c_int64 * (hit_capacity * 3))()
        hc = _hunt.hunt_batch_tiered(
            ctypes.c_uint64(start_seed), ctypes.c_int(n),
            ctypes.c_int(step_2x), ctypes.c_int(G), ctypes.c_int(hit_capacity),
            results)
        if hc >= 0:
            break
        hit_capacity = -hc
    return [(int(results[i*3]), int(results[i*3+1]), int(results[i*3+2]))
            for i in range(hc)]


def prefilter_range(start_seed, n, lo, hi, survivors):
    """Run the band-pass prefilter on a consecutive seed range."""
    return _hunt.prefilter_range(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_float(lo), ctypes.c_float(hi), survivors,
        ctypes.c_int(len(survivors)))


CHUNK = 8192

def scan_survivors_stream(seeds, n, step_2x, G, submit_verification, s1x, s2x):
    """Scan compacted survivors in chunks and submit hits immediately."""
    if n == 0:
        return 0

    total_hits = 0
    hit_capacity = INITIAL_HIT_CAP
    results = (ctypes.c_int64 * (hit_capacity * 3))()
    for off in range(0, n, CHUNK):
        sz = min(CHUNK, n - off)
        ptr = ctypes.cast(
            ctypes.byref(seeds, off * ctypes.sizeof(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_uint64))
        hc = _hunt.tiered_scan_mem(
            ptr, sz, step_2x, G, hit_capacity, results)
        if hc < 0:
            hit_capacity = -hc
            results = (ctypes.c_int64 * (hit_capacity * 3))()
            hc = _hunt.tiered_scan_mem(
                ptr, sz, step_2x, G, hit_capacity, results)
        if hc < 0:
            raise RuntimeError(f'tiered scan overflow retry failed: {-hc} hits')
        total_hits += hc
        for i in range(hc):
            seed, gx, gz = int(results[i*3]), int(results[i*3+1]), int(results[i*3+2])
            if not submit_verification(seed, gx, gz, s1x, s2x):
                return total_hits
    return total_hits


def verify_pair_cpu(seed, gx, gz, step_1x, step_2x):
    import struct as _struct
    buf = (ctypes.c_ubyte * 8192)()
    _eng._lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
    off_amp = 24*257 + 24*8*3
    for j in range(6):
        _struct.pack_into('d', buf, off_amp + j*8, 0.0)

    def s(x, z):
        return _eng._lib.cont_sample(buf, x, z)

    S1, T, rt32 = float(step_1x), -1.00, 0.8660254037844386

    pv = [s(gx, int(gz+S1)), s(int(gx+S1*rt32), int(gz+0.5*S1)),
          s(int(gx+S1*rt32), int(gz+1.5*S1)), s(int(gx-S1*rt32), int(gz+1.5*S1)),
          s(int(gx-S1*rt32), int(gz+0.5*S1))]
    if sum(1 for v in pv if v < T) >= 2: return True

    ph = [s(int(gx+S1), gz), s(int(gx+0.5*S1), int(gz-S1*rt32)),
          s(int(gx+1.5*S1), int(gz-S1*rt32)), s(int(gx+1.5*S1), int(gz+S1*rt32)),
          s(int(gx+0.5*S1), int(gz+S1*rt32))]
    if sum(1 for v in ph if v < T) >= 2: return True
    return False


def flood_fill_6oct(seed, cx, cz):
    return _eng._lib.cont_flood_fill_6oct(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(cx), ctypes.c_int(cz), ctypes.c_int(10000000))


def flood_fill_full(seed, cx, cz, max_cells=10_000_000):
    area = _eng._lib.cont_flood_fill(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(cx), ctypes.c_int(cz), ctypes.c_int(max_cells))
    if area <= 0: return None
    return {'seed': seed, 'area': area, 'cx': cx, 'cz': cz}


def verify_and_flood(seed, gx, gz, step_1x, step_2x):
    t0 = time.perf_counter()
    ok = verify_pair_cpu(seed, gx, gz, step_1x, step_2x)
    t_vfy = time.perf_counter() - t0
    if not ok: return (False, None, t_vfy, 0)
    t0 = time.perf_counter()
    area_6 = flood_fill_6oct(seed, gx, gz)
    if area_6 < 3_000_000: return (True, None, t_vfy, time.perf_counter() - t0)
    t1 = time.perf_counter()
    ff = flood_fill_full(seed, gx, gz)
    return (True, ff, t_vfy, (t1 - t0) + (time.perf_counter() - t1))


if __name__ == '__main__':
    TARGET = 4_000_000
    step_1x, step_2x = 140, 280
    G, batch = 512, 8192
    FF_WORKERS = 16
    MAX_PENDING_VERIFICATIONS = FF_WORKERS * 2

    print(f'Target: >= {TARGET:,} blocks^2 (4M+)')
    print(f'step_1x={step_1x}  step_2x={step_2x}  G={G}  batch={batch}')
    print(f'Coarse grid: {(G-1)*step_2x:,}x{(G-1)*step_2x:,} blocks at 1:1')
    if PREFT_ENABLED:
        print(f'Pre-filter: ON  band=[{PREFT_LO:.5f}, {PREFT_HI:.5f}]  '
              f'(p99.15-p99.25, {PREF_BATCH//1_000_000}M seeds/batch)')
    else:
        print(f'Pre-filter: OFF')
    print(f'GPU: hex grid + adjacent pair. CPU: verify + flood fill.')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, tiered_scanned, hits_gpu, hits_verified, hits_ok, hits_big = 0, 0, 0, 0, 0, 0
    t_gpu, t_vfy, t_ff = 0.0, 0.0, 0.0
    seen = set()
    lock = threading.Lock()
    pending = threading.BoundedSemaphore(MAX_PENDING_VERIFICATIONS)
    t0 = time.perf_counter()
    running = True

    def on_done(future):
        global best, hits_ok, hits_big, hits_verified, t_vfy, t_ff
        verified, r, tv, tf = future.result()
        with lock:
            if verified: hits_verified += 1
            t_vfy += tv; t_ff += tf
        if r is None: return
        with lock: hits_ok += 1
        if r['area'] < TARGET: return
        with lock:
            key = (r['seed'], r['area'])
            if key in seen: return
            seen.add(key)
            hits_big += 1
            entry = {'seed': r['seed'], 'area': r['area'], 'cx': r['cx'], 'cz': r['cz']}
            with open('islands_4m.jsonl', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f'\n  BIG ({r["area"]:,}): seed {r["seed"]} at ({r["cx"]},{r["cz"]})')
            if best is None or r['area'] > best['area']:
                best = r
                print(f'  *** NEW BEST: {r["seed"]}, {r["area"]:,} ***')
                with open('best_4m.jsonl', 'w') as f:
                    json.dump(entry, f)

    def on_done_releasing(future):
        try:
            on_done(future)
        finally:
            pending.release()

    def submit_verification(seed, gx, gz, s1x, s2x):
        while running:
            if pending.acquire(timeout=0.5):
                break
        else:
            return False
        try:
            future = pool.submit(verify_and_flood, seed, gx, gz, s1x, s2x)
            future.add_done_callback(on_done_releasing)
        except BaseException:
            pending.release()
            raise
        return True

    def gpu_thread():
        global scanned, tiered_scanned, hits_gpu, t_gpu
        use_prefilter = PREFT_ENABLED
        batch_num = 0
        survivor_capacity = PREF_SURVIVOR_CAP
        survivors = (ctypes.c_uint64 * survivor_capacity)()
        while running:
            start = random.getrandbits(64)
            t1 = time.perf_counter()

            if use_prefilter:
                batch_num += 1
                n_pass = prefilter_range(start, PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
                if n_pass < 0:
                    survivor_capacity = -n_pass
                    survivors = (ctypes.c_uint64 * survivor_capacity)()
                    n_pass = prefilter_range(
                        start, PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
                if n_pass < 0:
                    raise RuntimeError(f'prefilter overflow retry failed: {-n_pass} survivors')
                t_pref = time.perf_counter()
                with lock: scanned += PREF_BATCH

                if n_pass <= 0:
                    print(f'\n  batch {batch_num}: {PREF_BATCH//1_000_000}M seeds -> 0 survivors '
                          f'(pref {t_pref-t1:.2f}s)')
                    continue

                hc = scan_survivors_stream(
                    survivors, n_pass, step_2x, G,
                    submit_verification, step_1x, step_2x)
                t_end = time.perf_counter()
                with lock:
                    hits_gpu += hc
                    tiered_scanned += n_pass

                t_total = t_end - t1
                t_scan_time = t_end - t_pref
                print(f'\n  batch {batch_num}: prefilter {PREF_BATCH//1_000_000}M -> {n_pass:,} survivors '
                      f'({t_total:.1f}s: pref {t_pref-t1:.2f}s)')
                print(f'    tiered scan: {n_pass:,} seeds in {t_scan_time:.1f}s = {n_pass/t_scan_time:,.0f} seeds/s, {hc} GPU hits')
            else:
                hits = hunt_batch_tiered(start, batch, step_2x, G)
                with lock:
                    scanned += batch
                    tiered_scanned += batch
                    hits_gpu += len(hits)
                for seed, gx, gz in hits:
                    submit_verification(seed, gx, gz, step_1x, step_2x)

    gpu_t = threading.Thread(target=gpu_thread, daemon=True)
    gpu_t.start()

    try:
        while True:
            time.sleep(2)
            if scanned > 0:
                ba = best['area'] if best else 0
                elapsed = time.perf_counter() - t0
                print(f'\r  tiered: {tiered_scanned:,} seeds '
                      f'({tiered_scanned/elapsed:,.0f}/s) | '
                      f'{hits_verified} ok ({hits_ok} flood) | {hits_big} big | best {ba:,}  ',
                      end='', flush=True)
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
    print(f'{hits_gpu} GPU pairs -> {hits_verified} verified -> {hits_big} big (>= {TARGET:,})')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} at ({best["cx"]},{best["cz"]})')

"""hunt_tiered.py — Hex grid GPU prefilter/coarse verify + CPU flood fill.

GPU: hex grid (staggered rows, 500-block spacing), only O6+O15 octaves,
     threshold -1.0, 3-connected hit detection. ~20K seeds/s.
     Optional pre-filter: GPU LUT-variance filter generates a consecutive
     seed range on-device and returns only compacted survivors.

GPU: a separate six-octave R=2, 1x hex screen at 250-block spacing removes
     small first-layer hits before host download.

CPU: a cached 0.5x hex lookup samples the connected low-cell component
     touching the three coarse hit points; only probable >=6M islands reach
     the six-octave flood and then the full flood.
     Logs >=4M block^2 islands to islands_4m.jsonl.
"""
import sys, os, time, ctypes, json, random
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

GPU_DIR = os.path.dirname(__file__)
_hunt = ctypes.CDLL(os.path.join(GPU_DIR, 'hunt_engine.dll'))

_hunt.hunt_batch_tiered.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_int,
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
    ctypes.c_float, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
_hunt.tiered_scan_mem.restype = ctypes.c_int

_hunt.tiered_scan_mem_profiled.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
]
_hunt.tiered_scan_mem_profiled.restype = ctypes.c_int

_hunt.hunt_cleanup.argtypes = []
_hunt.hunt_cleanup.restype = None

INITIAL_HIT_CAP = 65_536
ESTIMATE_TARGET_AREA = int(
    os.environ.get('HUNT_ESTIMATE_TARGET', '6000000'))
GPU_COARSE_MIN_AREA = int(
    os.environ.get('HUNT_GPU_COARSE_MIN_AREA', '6000000'))

# ── Pre-filter config ──────────────────────────────────────────────────
# p99.15-p99.25 band: score [0.04330, 0.04331]
# Keeps seeds BETWEEN these scores (0.1% of randoms)
# Lower rejects noise, upper rejects score-ceiling randoms
PREFT_LO = 0.0432   # captures 65% of 4M+ in [0.0429, 0.0433]
PREFT_HI = 0.0434   # 23× enrichment, ~0.95% random pass
PREFT_ENABLED = True
PREF_BATCH = 500_000_000
PREF_SURVIVOR_CAP = 2_000_000


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value, 0) if value else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ('0', 'false', 'no', 'off')


def hunt_batch_tiered(start_seed, n, step_2x, G, threshold=-1.0):
    hit_capacity = INITIAL_HIT_CAP
    while True:
        results = (ctypes.c_int64 * (hit_capacity * 4))()
        hc = _hunt.hunt_batch_tiered(
            ctypes.c_uint64(start_seed), ctypes.c_int(n),
            ctypes.c_int(step_2x), ctypes.c_int(G), ctypes.c_float(threshold),
            ctypes.c_int(hit_capacity), results)
        if hc >= 0:
            break
        hit_capacity = -hc
    return [(int(results[i*4]), int(results[i*4+1]),
             int(results[i*4+2]), int(results[i*4+3]))
            for i in range(hc)]


def prefilter_range(start_seed, n, lo, hi, survivors):
    """Run the band-pass prefilter on a consecutive seed range."""
    return _hunt.prefilter_range(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_float(lo), ctypes.c_float(hi), survivors,
        ctypes.c_int(len(survivors)))


CHUNK = 8192

def scan_survivors_stream(seeds, n, step_2x, G, threshold,
                          submit_verification, step_05x, s2x,
                          profile_stats=None):
    """Scan compacted survivors in chunks and submit hits immediately."""
    if n == 0:
        return 0

    total_hits = 0
    hit_capacity = INITIAL_HIT_CAP
    results = (ctypes.c_int64 * (hit_capacity * 4))()
    for off in range(0, n, CHUNK):
        sz = min(CHUNK, n - off)
        ptr = ctypes.cast(
            ctypes.byref(seeds, off * ctypes.sizeof(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_uint64))
        call_started = time.perf_counter()
        hc = _hunt.tiered_scan_mem(
            ptr, sz, step_2x, G, ctypes.c_float(threshold),
            hit_capacity, results)
        if hc < 0:
            hit_capacity = -hc
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            hc = _hunt.tiered_scan_mem(
                ptr, sz, step_2x, G, ctypes.c_float(threshold),
                hit_capacity, results)
        if hc < 0:
            raise RuntimeError(f'tiered scan overflow retry failed: {-hc} hits')
        if profile_stats is not None:
            profile_stats['dll_seconds'] += time.perf_counter() - call_started
            profile_stats['dll_calls'] += 1
            profile_stats['chunks'] += 1
        total_hits += hc
        submit_started = time.perf_counter()
        keep_scanning = True
        for i in range(hc):
            hit_offset = i * 4
            seed = int(results[hit_offset])
            gx = int(results[hit_offset + 1])
            gz = int(results[hit_offset + 2])
            geometry_code = int(results[hit_offset + 3])
            if not submit_verification(
                    seed, gx, gz, geometry_code, step_05x, s2x):
                keep_scanning = False
                break
        if profile_stats is not None:
            profile_stats['decode_submit_seconds'] += (
                time.perf_counter() - submit_started)
            profile_stats['chunk_hits'] += hc
        if not keep_scanning:
            return total_hits
    return total_hits


def estimate_triple_area(seed, gx, gz, geometry_code, step_05x, step_2x):
    return _eng._lib.cont_estimate_triple_area(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(gx), ctypes.c_int(gz), ctypes.c_int(geometry_code),
        ctypes.c_int(step_05x), ctypes.c_int(step_2x))


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


def verify_and_flood(seed, gx, gz, geometry_code, step_05x, step_2x):
    t0 = time.perf_counter()
    estimated_area = estimate_triple_area(
        seed, gx, gz, geometry_code, step_05x, step_2x)
    t_vfy = time.perf_counter() - t0
    if estimated_area < ESTIMATE_TARGET_AREA:
        return (False, None, t_vfy, 0)
    t0 = time.perf_counter()
    area_6 = flood_fill_6oct(seed, gx, gz)
    if area_6 < 3_000_000: return (True, None, t_vfy, time.perf_counter() - t0)
    t1 = time.perf_counter()
    ff = flood_fill_full(seed, gx, gz)
    return (True, ff, t_vfy, (t1 - t0) + (time.perf_counter() - t1))


def verify_and_flood_profiled(submitted_at, seed, gx, gz, geometry_code,
                              step_05x, step_2x):
    started_at = time.perf_counter()
    result = verify_and_flood(
        seed, gx, gz, geometry_code, step_05x, step_2x)
    finished_at = time.perf_counter()
    return (*result, started_at - submitted_at, finished_at - started_at)


if __name__ == '__main__':
    TARGET = _env_int('HUNT_TARGET_AREA', 4_000_000)
    step_05x = _env_int('HUNT_STEP_05X', 125)
    step_2x = _env_int('HUNT_STEP_2X', 500)
    G = _env_int('HUNT_GRID_SIZE', 512)
    batch = _env_int('HUNT_BATCH_SIZE', 8192)
    THRESHOLD = _env_float('HUNT_THRESHOLD', -0.95)
    FF_WORKERS = _env_int('HUNT_CPU_WORKERS', 24)
    MAX_PENDING_VERIFICATIONS = _env_int(
        'HUNT_MAX_PENDING', max(2048, FF_WORKERS * 2))
    pref_batch = _env_int('HUNT_PREF_BATCH', PREF_BATCH)
    survivor_capacity = _env_int(
        'HUNT_SURVIVOR_CAP', PREF_SURVIVOR_CAP)
    scan_limit = _env_int('HUNT_SCAN_LIMIT', 0)
    max_prefilter_batches = _env_int('HUNT_MAX_PREFILTER_BATCHES', 0)
    profile_stages = _env_flag('HUNT_PROFILE_STAGES')
    fixed_start_value = os.environ.get('HUNT_START_SEED')
    fixed_start = int(fixed_start_value, 0) if fixed_start_value else None

    print(f'Target: >= {TARGET:,} blocks^2 (final flood)')
    print(f'GPU coarse gate: >= {GPU_COARSE_MIN_AREA:,} estimated blocks^2')
    print(f'step_05x={step_05x}  step_2x={step_2x}  G={G}  threshold={THRESHOLD}  batch={batch}')
    print(f'Coarse grid: {(G-1)*step_2x:,}x{(G-1)*step_2x:,} blocks at 1:1')
    if PREFT_ENABLED:
        print(f'Pre-filter: ON  band=[{PREFT_LO:.5f}, {PREFT_HI:.5f}]  '
              f'(p99.15-p99.25, {pref_batch//1_000_000}M seeds/batch)')
    else:
        print(f'Pre-filter: OFF')
    print('GPU: hex grid + connected triple + R=2 coarse area screen.')
    print('CPU: 0.5x area estimate + flood fill.')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, tiered_scanned, hits_gpu, hits_estimated, hits_ok, hits_big = 0, 0, 0, 0, 0, 0
    t_gpu, t_vfy, t_ff = 0.0, 0.0, 0.0
    seen = set()
    lock = threading.Lock()
    profile_lock = threading.Lock()
    pending = threading.BoundedSemaphore(MAX_PENDING_VERIFICATIONS)
    stage_profile = {
        'prefilter_seconds': 0.0,
        'dll_seconds': 0.0,
        'decode_submit_seconds': 0.0,
        'pending_wait_seconds': 0.0,
        'pool_submit_seconds': 0.0,
        'queue_delay_seconds': 0.0,
        'queue_delay_max': 0.0,
        'service_seconds': 0.0,
        'service_max': 0.0,
        'first_submit_at': None,
        'last_complete_at': None,
        'dll_calls': 0,
        'chunks': 0,
        'chunk_hits': 0,
        'submissions': 0,
        'blocked_submissions': 0,
        'completed': 0,
    } if profile_stages else None
    t0 = time.perf_counter()
    running = True

    def on_done(future):
        global best, hits_ok, hits_big, hits_estimated, t_vfy, t_ff
        result = future.result()
        verified, r, tv, tf = result[:4]
        if profile_stages:
            queue_delay, service_seconds = result[4:]
            completed_at = time.perf_counter()
            with profile_lock:
                stage_profile['queue_delay_seconds'] += queue_delay
                stage_profile['queue_delay_max'] = max(
                    stage_profile['queue_delay_max'], queue_delay)
                stage_profile['service_seconds'] += service_seconds
                stage_profile['service_max'] = max(
                    stage_profile['service_max'], service_seconds)
                stage_profile['completed'] += 1
                stage_profile['last_complete_at'] = completed_at
        with lock:
            if verified: hits_estimated += 1
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

    def submit_verification(seed, gx, gz, geometry_code, step05x, step2x):
        wait_started = time.perf_counter()
        acquired = pending.acquire(blocking=False)
        if not acquired:
            if profile_stages:
                stage_profile['blocked_submissions'] += 1
            while running:
                if pending.acquire(timeout=0.5):
                    acquired = True
                    break
        if profile_stages:
            stage_profile['pending_wait_seconds'] += (
                time.perf_counter() - wait_started)
        if not acquired:
            return False
        try:
            submitted_at = time.perf_counter()
            if profile_stages:
                future = pool.submit(
                    verify_and_flood_profiled, submitted_at,
                    seed, gx, gz, geometry_code, step05x, step2x)
            else:
                future = pool.submit(
                    verify_and_flood,
                    seed, gx, gz, geometry_code, step05x, step2x)
            future.add_done_callback(on_done_releasing)
            if profile_stages:
                stage_profile['pool_submit_seconds'] += (
                    time.perf_counter() - submitted_at)
                stage_profile['submissions'] += 1
                if stage_profile['first_submit_at'] is None:
                    stage_profile['first_submit_at'] = submitted_at
        except BaseException:
            pending.release()
            raise
        return True

    def gpu_thread():
        global scanned, tiered_scanned, hits_gpu, t_gpu, running
        use_prefilter = PREFT_ENABLED
        batch_num = 0
        active_survivor_capacity = survivor_capacity
        survivors = (ctypes.c_uint64 * active_survivor_capacity)()
        while running:
            start = (fixed_start + batch_num * pref_batch) & 0xFFFFFFFFFFFFFFFF \
                if fixed_start is not None else random.getrandbits(64)
            t1 = time.perf_counter()

            if use_prefilter:
                batch_num += 1
                n_pass = prefilter_range(
                    start, pref_batch, PREFT_LO, PREFT_HI, survivors)
                if n_pass < 0:
                    active_survivor_capacity = -n_pass
                    survivors = (ctypes.c_uint64 * active_survivor_capacity)()
                    n_pass = prefilter_range(
                        start, pref_batch, PREFT_LO, PREFT_HI, survivors)
                if n_pass < 0:
                    raise RuntimeError(f'prefilter overflow retry failed: {-n_pass} survivors')
                t_pref = time.perf_counter()
                if profile_stages:
                    stage_profile['prefilter_seconds'] += t_pref - t1
                with lock: scanned += pref_batch

                if n_pass <= 0:
                    print(f'\n  batch {batch_num}: {pref_batch//1_000_000}M seeds -> 0 survivors '
                          f'(pref {t_pref-t1:.2f}s)')
                    if max_prefilter_batches and batch_num >= max_prefilter_batches:
                        running = False
                    continue

                n_scan = min(n_pass, scan_limit) if scan_limit else n_pass
                hc = scan_survivors_stream(
                    survivors, n_scan, step_2x, G, THRESHOLD,
                    submit_verification, step_05x, step_2x, stage_profile)
                t_end = time.perf_counter()
                with lock:
                    hits_gpu += hc
                    tiered_scanned += n_scan

                t_total = t_end - t1
                t_scan_time = t_end - t_pref
                print(f'\n  batch {batch_num}: prefilter {pref_batch//1_000_000}M -> {n_pass:,} survivors '
                      f'({t_total:.1f}s: pref {t_pref-t1:.2f}s)')
                print(f'    tiered+coarse: {n_scan:,} seeds in {t_scan_time:.1f}s = {n_scan/t_scan_time:,.0f} seeds/s, {hc} coarse candidates')
                if max_prefilter_batches and batch_num >= max_prefilter_batches:
                    running = False
            else:
                hits = hunt_batch_tiered(start, batch, step_2x, G, THRESHOLD)
                with lock:
                    scanned += batch
                    tiered_scanned += batch
                    hits_gpu += len(hits)
                for seed, gx, gz, geometry_code in hits:
                    submit_verification(
                        seed, gx, gz, geometry_code, step_05x, step_2x)

    gpu_t = threading.Thread(target=gpu_thread, daemon=True)
    gpu_t.start()

    try:
        while running:
            time.sleep(2)
            if scanned > 0:
                ba = best['area'] if best else 0
                elapsed = time.perf_counter() - t0
                print(f'\r  tiered: {tiered_scanned:,} seeds '
                      f'({tiered_scanned/elapsed:,.0f}/s) | '
                      f'{hits_estimated} estimates ({hits_ok} flood) | {hits_big} big | best {ba:,}  ',
                      end='', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down...')
        running = False
        gpu_t.join(timeout=10)
        pool.shutdown(wait=True)
        _hunt.hunt_cleanup()

    if profile_stages:
        submissions = stage_profile['submissions']
        completed = stage_profile['completed']
        active_wall = 0.0
        if (stage_profile['first_submit_at'] is not None
                and stage_profile['last_complete_at'] is not None):
            active_wall = (stage_profile['last_complete_at']
                           - stage_profile['first_submit_at'])
        average_queue_ms = (stage_profile['queue_delay_seconds'] * 1000
                            / completed) if completed else 0.0
        average_service_ms = (stage_profile['service_seconds'] * 1000
                              / completed) if completed else 0.0
        effective_workers = (stage_profile['service_seconds'] / active_wall
                             if active_wall else 0.0)
        print('\nStage profile:')
        print(f'  prefilter: {stage_profile["prefilter_seconds"]:.3f}s')
        print(f'  DLL scan: {stage_profile["dll_seconds"]:.3f}s across '
              f'{stage_profile["dll_calls"]} chunks')
        print(f'  decode + submit loop: '
              f'{stage_profile["decode_submit_seconds"]:.3f}s')
        print(f'    semaphore wait: {stage_profile["pending_wait_seconds"]:.3f}s '
              f'({stage_profile["blocked_submissions"]:,} blocked)')
        print(f'    pool submit/callback: '
              f'{stage_profile["pool_submit_seconds"]:.3f}s')
        print(f'  CPU tasks: {submissions:,} submitted, {completed:,} completed')
        print(f'    queue delay: avg {average_queue_ms:.3f}ms, '
              f'max {stage_profile["queue_delay_max"]*1000:.3f}ms')
        print(f'    service time: avg {average_service_ms:.3f}ms, '
              f'max {stage_profile["service_max"]*1000:.3f}ms')
        print(f'    worker active wall: {active_wall:.3f}s, '
              f'effective parallelism {effective_workers:.2f}')

    elapsed = time.perf_counter() - t0
    print(f'\nScanned {scanned:,} seeds in {elapsed:.0f}s ({scanned/elapsed:,.0f}/s)')
    print(f'{hits_gpu} GPU coarse candidates -> {hits_estimated} CPU estimates passed -> {hits_big} big (>= {TARGET:,})')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} at ({best["cx"]},{best["cz"]})')

"""hunt_tiered.py — O6 translation hunt with GPU area screening.

The first GPU tier classifies the repeating 256x256 O6 Perlin-vector cells,
then evaluates the exact 1x hex grid only in promising cells and their coarse
two-step perimeter. Connected components meeting the O6 area gate emit a
capped eight-cell shape. Each shape is translated to every O6-period position
inside the world border, expanded by one 250-grid ring, and evaluated with
O6+O15 before CPU six-octave validation and the final flood fills.
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

_hunt.tiered_scan_mem_translated.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
]
_hunt.tiered_scan_mem_translated.restype = ctypes.c_int

_hunt.hunt_batch_tiered_translated.argtypes = [
    ctypes.c_uint64, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
]
_hunt.hunt_batch_tiered_translated.restype = ctypes.c_int

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

O6_THRESHOLD = -0.47
O6_STEP_2X = 500
GPU_GRID_SIZE = 512
O6_TRANSLATION_PERIOD = 512 * 256
WORLD_BORDER_PIPELINE = 7_500_000
GPU_ESTIMATE_STEP_1X = 250
GPU_ESTIMATE_STEP_2X = 500
GPU_ESTIMATE_TARGET_AREA = 6_000_000
GPU_ESTIMATE_THRESHOLD = -0.88
O6_TARGET_AREA = 6_000_000
O6_VECTOR_MARGIN = 0.20
CPU_6OCT_GATE_AREA = 6_000_000
CPU_VALIDATION_STEP_1X = 250
CPU_VALIDATION_STEP_2X = 500
CPU_VALIDATION_TARGET_AREA = 6_000_000
CPU_VALIDATION_THRESHOLD = -1.0
FINAL_TARGET_AREA = 6_000_000
CPU_WORKERS = 28
OUTPUT_FILE = 'islands_6m.jsonl'
BEST_FILE = 'best_6m.jsonl'
DEBUG_VALIDATION = os.environ.get('HUNT_DEBUG_VALIDATION', '0').lower() \
    not in ('0', 'false', 'no', 'off')

# ── Pre-filter config ──────────────────────────────────────────────────
# Keeps seeds BETWEEN these scores
# Lower rejects noise, upper rejects score-ceiling randoms
PREFT_LO = 0.04335
PREFT_HI = 0.0435 
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


def cap_o6_grid_size(grid_size, step_2x, o6_period):
    """Cap the sampled span to one open O6 period without duplicate edges."""
    if grid_size <= 0:
        raise ValueError('HUNT_GRID_SIZE must be positive')
    if step_2x <= 0 or o6_period <= 0:
        return grid_size
    max_grid = ((o6_period - 1) // step_2x) + 1
    return min(grid_size, max_grid)


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


def hunt_batch_tiered_translated(
        start_seed, n, scan_step_2x, G, o6_threshold,
        translation_period, world_border,
        estimate_step_1x, estimate_step_2x, estimate_target_area):
    hit_capacity = INITIAL_HIT_CAP
    while True:
        results = (ctypes.c_int64 * (hit_capacity * 4))()
        hc = _hunt.hunt_batch_tiered_translated(
            ctypes.c_uint64(start_seed), ctypes.c_int(n),
            ctypes.c_int(scan_step_2x), ctypes.c_int(G),
            ctypes.c_float(o6_threshold),
            ctypes.c_int(translation_period), ctypes.c_int(world_border),
            ctypes.c_int(estimate_step_1x), ctypes.c_int(estimate_step_2x),
            ctypes.c_int64(estimate_target_area),
            ctypes.c_int(hit_capacity), results)
        if hc >= 0:
            break
        hit_capacity = -hc
    return [(int(results[i * 4]), int(results[i * 4 + 1]),
             int(results[i * 4 + 2]), int(results[i * 4 + 3]))
            for i in range(hc)]


def estimate_triple_area_6oct(seed, gx, gz, geometry_code,
                              step_1x, step_2x, threshold):
    """Run the native six-octave 1x connected-grid validation."""
    return _eng._lib.cont_estimate_triple_area_6oct(
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_int(gx), ctypes.c_int(gz), ctypes.c_int(geometry_code),
        ctypes.c_int(step_1x), ctypes.c_int(step_2x),
        ctypes.c_double(threshold))


def prefilter_range(start_seed, n, lo, hi, survivors):
    """Run the band-pass prefilter on a consecutive seed range."""
    return _hunt.prefilter_range(
        ctypes.c_uint64(start_seed), ctypes.c_int(n),
        ctypes.c_float(lo), ctypes.c_float(hi), survivors,
        ctypes.c_int(len(survivors)))


CHUNK = 8192

def scan_survivors_stream(
        seeds, n, scan_step_2x, G, o6_threshold,
        translation_period, world_border,
        estimate_step_1x, estimate_step_2x, estimate_target_area,
        submit_verification, profile_stats=None):
    """Scan survivors and submit GPU translation estimates immediately."""
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
        hc = _hunt.tiered_scan_mem_translated(
            ptr, sz, ctypes.c_int(scan_step_2x), ctypes.c_int(G),
            ctypes.c_float(o6_threshold),
            ctypes.c_int(translation_period), ctypes.c_int(world_border),
            ctypes.c_int(estimate_step_1x), ctypes.c_int(estimate_step_2x),
            ctypes.c_int64(estimate_target_area),
            ctypes.c_int(hit_capacity), results)
        if hc < 0:
            hit_capacity = -hc
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            hc = _hunt.tiered_scan_mem_translated(
                ptr, sz, ctypes.c_int(scan_step_2x), ctypes.c_int(G),
                ctypes.c_float(o6_threshold),
                ctypes.c_int(translation_period), ctypes.c_int(world_border),
                ctypes.c_int(estimate_step_1x), ctypes.c_int(estimate_step_2x),
                ctypes.c_int64(estimate_target_area),
                ctypes.c_int(hit_capacity), results)
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
                    seed, gx, gz, geometry_code):
                keep_scanning = False
                break
        if profile_stats is not None:
            profile_stats['decode_submit_seconds'] += (
                time.perf_counter() - submit_started)
            profile_stats['chunk_hits'] += hc
        if not keep_scanning:
            return total_hits
    return total_hits


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


def verify_and_flood(seed, gx, gz, geometry_code,
                     validation_step_1x, validation_step_2x,
                     validation_target_area, validation_threshold,
                     cpu_6oct_gate):
    validation_started = time.perf_counter()
    validation_area = estimate_triple_area_6oct(
        seed, gx, gz, geometry_code, validation_step_1x, validation_step_2x,
        validation_threshold)
    validation_seconds = time.perf_counter() - validation_started
    if DEBUG_VALIDATION:
        print(f'\n  validation: seed={seed} x={gx} z={gz} '
              f'geometry=0x{geometry_code:02x} area={validation_area:,.0f}',
              flush=True)
    if validation_area < validation_target_area:
        return (False, False, None, validation_seconds, 0.0, 0.0,
                validation_area)

    flood6_started = time.perf_counter()
    area_6 = flood_fill_6oct(seed, gx, gz)
    flood6_seconds = time.perf_counter() - flood6_started
    if area_6 < cpu_6oct_gate:
        return (True, False, None, validation_seconds, flood6_seconds, 0.0,
                validation_area)

    full_flood_started = time.perf_counter()
    result = flood_fill_full(seed, gx, gz)
    full_flood_seconds = time.perf_counter() - full_flood_started
    return (True, True, result, validation_seconds, flood6_seconds,
            full_flood_seconds, validation_area)


def verify_and_flood_profiled(submitted_at, seed, gx, gz, geometry_code,
                              validation_step_1x, validation_step_2x,
                              validation_target_area, validation_threshold,
                              cpu_6oct_gate):
    started_at = time.perf_counter()
    result = verify_and_flood(
        seed, gx, gz, geometry_code, validation_step_1x, validation_step_2x,
        validation_target_area, validation_threshold, cpu_6oct_gate)
    finished_at = time.perf_counter()
    return (*result, started_at - submitted_at, finished_at - started_at)


if __name__ == '__main__':
    TARGET = _env_int('HUNT_TARGET_AREA', FINAL_TARGET_AREA)
    scan_step_2x = _env_int('HUNT_O6_STEP', O6_STEP_2X)
    o6_threshold = _env_float('HUNT_O6_THRESHOLD', O6_THRESHOLD)
    o6_target_area = _env_int('HUNT_O6_TARGET_AREA', O6_TARGET_AREA)
    o6_vector_screen = _env_flag('HUNT_O6_VECTOR_SCREEN', True)
    o6_vector_margin = _env_float(
        'HUNT_O6_VECTOR_MARGIN', O6_VECTOR_MARGIN)
    translation_period = _env_int(
        'HUNT_TRANSLATION_PERIOD', O6_TRANSLATION_PERIOD)
    world_border = _env_int(
        'HUNT_WORLD_BORDER_PIPELINE', WORLD_BORDER_PIPELINE)
    estimate_step_1x = _env_int(
        'HUNT_TRANSLATION_ESTIMATE_STEP', GPU_ESTIMATE_STEP_1X)
    estimate_step_2x = _env_int(
        'HUNT_TRANSLATION_ESTIMATE_STEP_2X', GPU_ESTIMATE_STEP_2X)
    estimate_target = _env_int(
        'HUNT_TRANSLATION_ESTIMATE_TARGET', GPU_ESTIMATE_TARGET_AREA)
    estimate_threshold = _env_float(
        'HUNT_TRANSLATION_ESTIMATE_THRESHOLD', GPU_ESTIMATE_THRESHOLD)
    translation_grouped = _env_flag('HUNT_TRANSLATION_GROUPED', True)
    translation_grouped_threads = _env_int(
        'HUNT_TRANSLATION_GROUPED_THREADS', 256)
    cpu_6oct_gate = _env_int(
        'HUNT_CPU_6OCT_GATE', CPU_6OCT_GATE_AREA)
    validation_step_1x = _env_int(
        'HUNT_CPU_VALIDATION_STEP_1X', CPU_VALIDATION_STEP_1X)
    validation_step_2x = _env_int(
        'HUNT_CPU_VALIDATION_STEP_2X', CPU_VALIDATION_STEP_2X)
    validation_target_area = _env_int(
        'HUNT_CPU_VALIDATION_TARGET', CPU_VALIDATION_TARGET_AREA)
    validation_threshold = _env_float(
        'HUNT_CPU_VALIDATION_THRESHOLD', CPU_VALIDATION_THRESHOLD)
    requested_G = _env_int('HUNT_GRID_SIZE', GPU_GRID_SIZE)
    if scan_step_2x <= 0 or (scan_step_2x & 1):
        raise ValueError('HUNT_O6_STEP must be a positive even 2x reference')
    scan_step_1x = scan_step_2x // 2
    G = cap_o6_grid_size(requested_G, scan_step_1x, translation_period)
    batch = _env_int('HUNT_BATCH_SIZE', 8192)
    FF_WORKERS = _env_int('HUNT_CPU_WORKERS', CPU_WORKERS)
    MAX_PENDING_VERIFICATIONS = _env_int(
        'HUNT_MAX_PENDING', max(2048, FF_WORKERS * 2))
    pref_batch = _env_int('HUNT_PREF_BATCH', PREF_BATCH)
    survivor_capacity = _env_int(
        'HUNT_SURVIVOR_CAP', PREF_SURVIVOR_CAP)
    scan_limit = _env_int('HUNT_SCAN_LIMIT', 0)
    max_prefilter_batches = _env_int('HUNT_MAX_PREFILTER_BATCHES', 0)
    profile_stages = _env_flag('HUNT_PROFILE_STAGES')
    output_file = os.environ.get('HUNT_OUTPUT_FILE', OUTPUT_FILE)
    best_file = os.environ.get('HUNT_BEST_FILE', BEST_FILE)
    fixed_start_value = os.environ.get('HUNT_START_SEED')
    fixed_start = int(fixed_start_value, 0) if fixed_start_value else None
    if estimate_step_2x != estimate_step_1x * 2:
        raise ValueError('GPU estimate step_2x must equal 2 * step_1x')
    if translation_grouped and estimate_step_1x != scan_step_1x:
        raise ValueError(
            'shape-aware GPU estimation requires matching O6 and estimate steps')
    if validation_step_2x != validation_step_1x * 2:
        raise ValueError('CPU validation step_2x must equal 2 * step_1x')

    print(f'Target: >= {TARGET:,} blocks^2 (final flood)')
    print(f'O6 area scan: step={scan_step_1x} threshold<{o6_threshold} '
          f'target>={o6_target_area:,} (2x reference={scan_step_2x})')
    print(f'O6 vector screen: {"ON" if o6_vector_screen else "OFF"} '
          f'center margin={o6_vector_margin:.3f}, 256x256 cells, '
          'one-cell coarse dilation')
    print(f'Translations: period={translation_period:,} border={world_border:,}')
    if G != requested_G:
        print(f'O6 grid cap: requested G={requested_G}, using G={G} '
              f'(span={(G - 1) * scan_step_1x:,} < '
              f'{translation_period:,})')
    print(f'GPU estimate: O6+O15<{estimate_threshold} step={estimate_step_1x} '
          f'component+one-ring target>={estimate_target:,} '
          f'mode={"grouped" if translation_grouped else "expanded"} '
          f'threads={translation_grouped_threads}')
    print(f'First-tier grid: {(G-1)*scan_step_1x:,}x'
          f'{(G-1)*scan_step_1x:,} pipeline blocks')
    if PREFT_ENABLED:
        print(f'Pre-filter: ON  band=[{PREFT_LO:.5f}, {PREFT_HI:.5f}]  '
              f'(p99.15-p99.25, {pref_batch//1_000_000}M seeds/batch)')
    else:
        print(f'Pre-filter: OFF')
    print('GPU: vector-screened O6 components -> translated shape+halo estimate.')
    print(f'CPU: six-octave 1x validation step={validation_step_1x} '
          f'threshold<{validation_threshold} target>={validation_target_area:,} -> '
          f'flood gate>={cpu_6oct_gate:,} -> full flood.')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, tiered_scanned, hits_gpu = 0, 0, 0
    hits_validation, hits_flood6, hits_big = 0, 0, 0
    t_gpu, t_validation, t_flood6, t_ff = 0.0, 0.0, 0.0, 0.0
    seen = set()
    submitted = set()
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
        global best, hits_validation, hits_flood6, hits_big
        global t_validation, t_flood6, t_ff
        result = future.result()
        validation_passed, flood6_passed, r = result[:3]
        tv, tf6, tf = result[3:6]
        if profile_stages:
            queue_delay, service_seconds = result[7:9]
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
            if validation_passed: hits_validation += 1
            if flood6_passed: hits_flood6 += 1
            t_validation += tv
            t_flood6 += tf6
            t_ff += tf
        if r is None: return
        if r['area'] < TARGET: return
        with lock:
            key = (r['seed'], r['area'])
            if key in seen: return
            seen.add(key)
            hits_big += 1
            entry = {'seed': r['seed'], 'area': r['area'], 'cx': r['cx'], 'cz': r['cz']}
            with open(output_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f'\n  BIG ({r["area"]:,}): seed {r["seed"]} at ({r["cx"]},{r["cz"]})')
            if best is None or r['area'] > best['area']:
                best = r
                print(f'  *** NEW BEST: {r["seed"]}, {r["area"]:,} ***')
                with open(best_file, 'w') as f:
                    json.dump(entry, f)

    def on_done_releasing(future):
        try:
            on_done(future)
        finally:
            pending.release()

    def submit_verification(seed, gx, gz, geometry_code):
        with lock:
            key = (seed, gx, gz)
            if key in submitted:
                return True
            submitted.add(key)
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
                    seed, gx, gz, geometry_code,
                    validation_step_1x, validation_step_2x,
                    validation_target_area, validation_threshold,
                    cpu_6oct_gate)
            else:
                future = pool.submit(
                    verify_and_flood,
                    seed, gx, gz, geometry_code,
                    validation_step_1x, validation_step_2x,
                    validation_target_area, validation_threshold,
                    cpu_6oct_gate)
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
                with lock:
                    submitted.clear()
                    seen.clear()
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
                    survivors, n_scan, scan_step_2x, G, o6_threshold,
                    translation_period, world_border,
                    estimate_step_1x, estimate_step_2x, estimate_target,
                    submit_verification, stage_profile)
                t_end = time.perf_counter()
                with lock:
                    hits_gpu += hc
                    tiered_scanned += n_scan

                t_total = t_end - t1
                t_scan_time = t_end - t_pref
                print(f'\n  batch {batch_num}: prefilter {pref_batch//1_000_000}M -> {n_pass:,} survivors '
                      f'({t_total:.1f}s: pref {t_pref-t1:.2f}s)')
                print(f'    tiered+translation GPU: {n_scan:,} seeds in '
                      f'{t_scan_time:.1f}s = '
                      f'{n_scan/t_scan_time:,.0f} seeds/s, '
                      f'{hc} GPU estimates')
                if max_prefilter_batches and batch_num >= max_prefilter_batches:
                    running = False
            else:
                hits = hunt_batch_tiered_translated(
                    start, batch, scan_step_2x, G, o6_threshold,
                    translation_period, world_border,
                    estimate_step_1x, estimate_step_2x, estimate_target)
                with lock:
                    scanned += batch
                    tiered_scanned += batch
                    hits_gpu += len(hits)
                for seed, gx, gz, geometry_code in hits:
                    submit_verification(seed, gx, gz, geometry_code)

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
                      f'{hits_validation} validations ({hits_flood6} six-oct floods) | '
                      f'{hits_big} big | best {ba:,}  ',
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
    print(f'{hits_gpu} GPU translation estimates -> '
          f'{hits_validation} CPU validations passed -> '
          f'{hits_flood6} six-octave floods passed -> '
          f'{hits_big} final islands (>= {TARGET:,})')
    print(f'CPU service totals: validation {t_validation:.3f}s, '
          f'six-octave flood {t_flood6:.3f}s, full flood {t_ff:.3f}s')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} at ({best["cx"]},{best["cz"]})')

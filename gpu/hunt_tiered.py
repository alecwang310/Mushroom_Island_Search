"""hunt_tiered.py — Hex grid O6+O15 GPU prefilter + CPU verify + flood fill.

GPU: hex grid (staggered rows, 280-block spacing), only O6+O15 octaves,
     threshold -1.0, 3-connected hit detection. ~20K seeds/s.
     Optional pre-filter: GPU LUT-variance filter generates a consecutive
     seed range on-device and returns only compacted survivors.

CPU: a cached 0.5x hex lookup samples the connected low-cell component
     touching the three coarse hit points; only probable >=4M islands reach
     the six-octave flood and then the full flood.
     Logs >=4M block^2 islands to islands_4m.jsonl.
"""
import sys, os, time, ctypes, json, random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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
ISLAND_THRESHOLD = -1.05
ESTIMATE_TARGET_AREA = 4_000_000
_SQRT3_OVER_2 = 0.8660254037844386
_GRID_CELL_AREA_SCALE = 16.0
_DEFAULT_STEP_05X = 70
_DEFAULT_STEP_1X = 140
_DEFAULT_STEP_2X = 280

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
        results = (ctypes.c_int64 * (hit_capacity * 4))()
        hc = _hunt.hunt_batch_tiered(
            ctypes.c_uint64(start_seed), ctypes.c_int(n),
            ctypes.c_int(step_2x), ctypes.c_int(G), ctypes.c_int(hit_capacity),
            results)
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

def scan_survivors_stream(seeds, n, step_2x, G, submit_verification,
                          step_05x, s2x):
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
        hc = _hunt.tiered_scan_mem(
            ptr, sz, step_2x, G, hit_capacity, results)
        if hc < 0:
            hit_capacity = -hc
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            hc = _hunt.tiered_scan_mem(
                ptr, sz, step_2x, G, hit_capacity, results)
        if hc < 0:
            raise RuntimeError(f'tiered scan overflow retry failed: {-hc} hits')
        total_hits += hc
        for i in range(hc):
            hit_offset = i * 4
            seed = int(results[hit_offset])
            gx = int(results[hit_offset + 1])
            gz = int(results[hit_offset + 2])
            geometry_code = int(results[hit_offset + 3])
            if not submit_verification(
                    seed, gx, gz, geometry_code, step_05x, s2x):
                return total_hits
    return total_hits


def _coarse_neighbor_offsets(step_2x, row_parity):
    stagger_x = int(0.5 * step_2x)
    row_z = int(_SQRT3_OVER_2 * step_2x)
    signed_stagger_x = stagger_x if row_parity == 0 else -stagger_x
    return (
        (-step_2x, 0),
        (step_2x, 0),
        (signed_stagger_x, -row_z),
        (-signed_stagger_x, -row_z),
        (signed_stagger_x, row_z),
        (-signed_stagger_x, row_z),
    )


@lru_cache(maxsize=None)
def _hex_disk_lattice(radius_steps, ring_only=False):
    lattice = []
    for axial_x in range(-radius_steps, radius_steps + 1):
        for row_delta in range(-radius_steps, radius_steps + 1):
            distance = max(
                abs(axial_x), abs(row_delta), abs(axial_x + row_delta))
            if distance > radius_steps:
                continue
            if ring_only and distance != radius_steps:
                continue
            lattice.append((axial_x, row_delta))
    return tuple(lattice)


def _lattice_offset(step, row_parity, axial_x, row_delta):
    half_step = int(0.5 * step)
    center_stagger = half_step if row_parity else 0
    target_parity = row_parity ^ (row_delta & 1)
    target_stagger = half_step if target_parity else 0
    return (
        axial_x * step + target_stagger - center_stagger,
        int(row_delta * _SQRT3_OVER_2 * step),
    )


@lru_cache(maxsize=None)
def _hex_disk_offsets(step, radius_steps, row_parity, ring_only=False):
    return tuple(
        _lattice_offset(step, row_parity, axial_x, row_delta)
        for axial_x, row_delta in _hex_disk_lattice(radius_steps, ring_only))


def _point_row_parity(center_parity, direction):
    return center_parity if direction < 2 else center_parity ^ 1


@lru_cache(maxsize=None)
def _build_triple_estimate_lookup(step_05x, step_2x):
    radius_steps = max(1, int(round(step_2x / step_05x)))
    coarse_steps = radius_steps
    lookup = {}
    for row_parity in (0, 1):
        coarse_offsets = ((0, 0),) + _coarse_neighbor_offsets(
            step_2x, row_parity)
        for pair_mask in range(64):
            if pair_mask.bit_count() != 2:
                continue
            directions = [
                direction for direction in range(6)
                if pair_mask & (1 << direction)]
            points = [(0, (0, 0))]
            for direction in directions:
                point_x, _ = coarse_offsets[direction + 1]
                if direction < 2:
                    point_row = 0
                    point_x_lattice = (-coarse_steps if direction == 0
                                       else coarse_steps)
                else:
                    point_row = (-coarse_steps if direction < 4
                                 else coarse_steps)
                    row_shift = _lattice_offset(
                        step_05x, row_parity, 0, point_row)[0]
                    point_x_lattice = round(
                        (point_x - row_shift) / step_05x)
                points.append((direction, (point_x_lattice, point_row)))

            inner = set()
            for _, (point_x, point_row) in points:
                for local_x, local_row in _hex_disk_lattice(radius_steps):
                    inner.add((point_x + local_x, point_row + local_row))

            lattice_points = tuple(sorted(inner))
            index_by_lattice = {
                lattice: index for index, lattice in enumerate(lattice_points)}
            offsets = tuple(
                _lattice_offset(step_05x, row_parity, axial_x, row_delta)
                for axial_x, row_delta in lattice_points)
            root_indices = tuple(
                index_by_lattice[point] for _, point in points)
            neighbor_indices = []
            for axial_x, row_delta in lattice_points:
                neighbors = (
                    (axial_x - 1, row_delta),
                    (axial_x + 1, row_delta),
                    (axial_x, row_delta - 1),
                    (axial_x, row_delta + 1),
                    (axial_x - 1, row_delta + 1),
                    (axial_x + 1, row_delta - 1),
                )
                neighbor_indices.append(tuple(
                    index_by_lattice[neighbor]
                    for neighbor in neighbors
                    if neighbor in index_by_lattice))
            lookup[(row_parity, pair_mask)] = (
                offsets, root_indices, tuple(neighbor_indices))
    return lookup


TRIPLE_ESTIMATE_LOOKUP = _build_triple_estimate_lookup(
    _DEFAULT_STEP_05X, _DEFAULT_STEP_2X)


def _estimate_triple_area_python(seed, gx, gz, geometry_code,
                                 step_05x, step_2x):
    buf = (ctypes.c_ubyte * 8192)()
    _eng._lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)

    row_parity = (geometry_code >> 6) & 1
    pair_mask = geometry_code & 0x3F
    if (step_05x, step_2x) == (_DEFAULT_STEP_05X, _DEFAULT_STEP_2X):
        lookup = TRIPLE_ESTIMATE_LOOKUP
    else:
        lookup = _build_triple_estimate_lookup(step_05x, step_2x)
    offsets, root_indices, neighbor_indices = lookup.get(
        (row_parity, pair_mask), ((), (), ()))
    if not offsets:
        return 0.0

    low = []
    for index, (offset_x, offset_z) in enumerate(offsets):
        low.append(_eng._lib.cont_sample(
            buf, gx + offset_x, gz + offset_z) < ISLAND_THRESHOLD)

    sample_area = step_05x * step_05x * _SQRT3_OVER_2 * _GRID_CELL_AREA_SCALE
    seed_indices = set(root_indices)
    for root_index in root_indices:
        seed_indices.update(neighbor_indices[root_index])
    queue = deque(index for index in seed_indices if low[index])
    connected = set(queue)
    while queue:
        index = queue.popleft()
        for neighbor_index in neighbor_indices[index]:
            if low[neighbor_index] and neighbor_index not in connected:
                connected.add(neighbor_index)
                queue.append(neighbor_index)
    return len(connected) * sample_area


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


if __name__ == '__main__':
    TARGET = 4_000_000
    step_05x, step_2x = 70, 280
    G, batch = 512, 8192
    FF_WORKERS = int(os.environ.get('HUNT_CPU_WORKERS', '24'))
    MAX_PENDING_VERIFICATIONS = FF_WORKERS * 2

    print(f'Target: >= {TARGET:,} blocks^2 (4M+)')
    print(f'step_05x={step_05x}  step_2x={step_2x}  G={G}  batch={batch}')
    print(f'Coarse grid: {(G-1)*step_2x:,}x{(G-1)*step_2x:,} blocks at 1:1')
    if PREFT_ENABLED:
        print(f'Pre-filter: ON  band=[{PREFT_LO:.5f}, {PREFT_HI:.5f}]  '
              f'(p99.15-p99.25, {PREF_BATCH//1_000_000}M seeds/batch)')
    else:
        print(f'Pre-filter: OFF')
    print(f'GPU: hex grid + connected triple. CPU: area estimate + flood fill.')
    print()

    pool = ThreadPoolExecutor(max_workers=FF_WORKERS)
    best = None
    scanned, tiered_scanned, hits_gpu, hits_estimated, hits_ok, hits_big = 0, 0, 0, 0, 0, 0
    t_gpu, t_vfy, t_ff = 0.0, 0.0, 0.0
    seen = set()
    lock = threading.Lock()
    pending = threading.BoundedSemaphore(MAX_PENDING_VERIFICATIONS)
    t0 = time.perf_counter()
    running = True

    def on_done(future):
        global best, hits_ok, hits_big, hits_estimated, t_vfy, t_ff
        verified, r, tv, tf = future.result()
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
        while running:
            if pending.acquire(timeout=0.5):
                break
        else:
            return False
        try:
            future = pool.submit(
                verify_and_flood,
                seed, gx, gz, geometry_code, step05x, step2x)
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
                    submit_verification, step_05x, step_2x)
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
                for seed, gx, gz, geometry_code in hits:
                    submit_verification(
                        seed, gx, gz, geometry_code, step_05x, step_2x)

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

    elapsed = time.perf_counter() - t0
    print(f'\nScanned {scanned:,} seeds in {elapsed:.0f}s ({scanned/elapsed:,.0f}/s)')
    print(f'{hits_gpu} GPU triples -> {hits_estimated} area estimates passed -> {hits_big} big (>= {TARGET:,})')
    if best:
        print(f'BEST: seed {best["seed"]}, {best["area"]:,} at ({best["cx"]},{best["cz"]})')

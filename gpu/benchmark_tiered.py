"""Benchmark the prefilter, GPU tiered scan, and CPU area-estimate stages."""

import ctypes
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hunt_tiered import (
    CHUNK,
    INITIAL_HIT_CAP,
    ESTIMATE_TARGET_AREA,
    PREF_BATCH,
    PREF_SURVIVOR_CAP,
    PREFT_HI,
    PREFT_LO,
    _hunt,
    flood_fill_6oct,
    flood_fill_full,
    prefilter_range,
    estimate_triple_area,
)


STEP_05X = int(os.environ.get('HUNT_STEP_05X', '75'))
STEP_2X = int(os.environ.get('HUNT_STEP_2X', '300'))
GRID_SIZE = int(os.environ.get('HUNT_GRID_SIZE', '512'))
THRESHOLD = float(os.environ.get('HUNT_THRESHOLD', '-0.95'))
BENCH_PREF_BATCH = int(os.environ.get('HUNT_PREF_BATCH', str(PREF_BATCH)))
SCAN_LIMIT = int(os.environ.get('HUNT_SCAN_LIMIT', '0'))
SKIP_CPU = os.environ.get('HUNT_SKIP_CPU', '0') == '1'
CPU_WORKERS = int(os.environ.get('HUNT_CPU_WORKERS', '24'))
MAX_PENDING = CPU_WORKERS * 2
CPU_HIT_LIMIT = int(os.environ.get('HUNT_CPU_HIT_LIMIT', '0'))
TIER_TIMING_NAMES = (
    'total', 'init', 'h2d', 'memset', 'kernel',
    'count_d2h', 'hits_d2h', 'pack',
)


def benchmark_verify_and_flood(seed, grid_x, grid_z, geometry_code):
    estimate_started = time.perf_counter()
    estimated_area = estimate_triple_area(
        seed, grid_x, grid_z, geometry_code, STEP_05X, STEP_2X)
    estimate_seconds = time.perf_counter() - estimate_started
    if estimated_area < ESTIMATE_TARGET_AREA:
        return False, None, estimate_seconds, 0.0, 0.0, False, estimated_area

    flood6_started = time.perf_counter()
    flood6_area = flood_fill_6oct(seed, grid_x, grid_z)
    flood6_seconds = time.perf_counter() - flood6_started
    if flood6_area < 3_000_000:
        return (
            True, None, estimate_seconds, flood6_seconds,
            0.0, False, estimated_area,
        )

    full_flood_started = time.perf_counter()
    result = flood_fill_full(seed, grid_x, grid_z)
    full_flood_seconds = time.perf_counter() - full_flood_started
    return (
        True, result, estimate_seconds, flood6_seconds,
        full_flood_seconds, True, estimated_area,
    )


def scan_survivors_gpu(seeds, seed_count):
    hits = []
    hit_capacity = INITIAL_HIT_CAP
    results = (ctypes.c_int64 * (hit_capacity * 4))()
    call_seconds = 0.0
    decode_seconds = 0.0
    tier_timings = {name: 0.0 for name in TIER_TIMING_NAMES}
    scan_started = time.perf_counter()

    def run_profiled(seed_ptr, chunk_size):
        nonlocal call_seconds
        timing_values = (ctypes.c_double * len(TIER_TIMING_NAMES))()
        call_started = time.perf_counter()
        hit_count = _hunt.tiered_scan_mem_profiled(
            seed_ptr, chunk_size, STEP_2X, GRID_SIZE,
            ctypes.c_float(THRESHOLD), hit_capacity, results,
            timing_values, len(TIER_TIMING_NAMES))
        call_seconds += time.perf_counter() - call_started
        for index, name in enumerate(TIER_TIMING_NAMES):
            tier_timings[name] += timing_values[index] / 1000.0
        return hit_count

    for offset in range(0, seed_count, CHUNK):
        chunk_size = min(CHUNK, seed_count - offset)
        seed_ptr = ctypes.cast(
            ctypes.byref(seeds, offset * ctypes.sizeof(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_uint64))
        hit_count = run_profiled(seed_ptr, chunk_size)

        if hit_count < 0:
            hit_capacity = -hit_count
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            hit_count = run_profiled(seed_ptr, chunk_size)
        if hit_count < 0:
            raise RuntimeError(f'GPU hit buffer overflow retry failed: {-hit_count}')

        decode_started = time.perf_counter()
        for hit_index in range(hit_count):
            hit_offset = hit_index * 4
            hits.append((
                int(results[hit_offset]),
                int(results[hit_offset + 1]),
                int(results[hit_offset + 2]),
                int(results[hit_offset + 3]),
            ))
        decode_seconds += time.perf_counter() - decode_started

    return (
        hits, call_seconds, decode_seconds,
        time.perf_counter() - scan_started, tier_timings,
    )


def benchmark_cpu(hits):
    if not hits:
        return {
            'wall_seconds': 0.0,
            'submit_seconds': 0.0,
            'service_seconds': 0.0,
            'estimate_seconds': 0.0,
            'flood6_seconds': 0.0,
            'full_flood_seconds': 0.0,
            'estimated_passes': 0,
            'estimated_area_sum': 0.0,
            'estimated_area_max': 0.0,
            'flood6_calls': 0,
            'full_flood_calls': 0,
            'big': 0,
            'peak_pending': 0,
            'blocked_submissions': 0,
            'submit_wait_seconds': 0.0,
        }

    state_lock = threading.Lock()
    pending = threading.BoundedSemaphore(MAX_PENDING)
    completed = threading.Event()
    pending_count = 0
    peak_pending = 0
    completed_count = 0
    estimated_pass_count = 0
    flood6_calls = 0
    full_flood_calls = 0
    big_count = 0
    estimated_area_sum = 0.0
    estimated_area_max = 0.0
    estimate_seconds = 0.0
    flood6_seconds = 0.0
    full_flood_seconds = 0.0
    blocked_submissions = 0
    submit_wait_seconds = 0.0

    def on_done(future):
        nonlocal pending_count, completed_count, estimated_pass_count
        nonlocal flood6_calls, full_flood_calls, big_count
        nonlocal estimated_area_sum, estimated_area_max
        nonlocal estimate_seconds, flood6_seconds, full_flood_seconds
        try:
            (
                estimated, result, estimate_time, flood6_time,
                full_flood_time, full_flood_called, estimated_area,
            ) = future.result()
            with state_lock:
                estimated_area_sum += estimated_area
                estimated_area_max = max(estimated_area_max, estimated_area)
                if estimated:
                    estimated_pass_count += 1
                    flood6_calls += 1
                if full_flood_called:
                    full_flood_calls += 1
                if result is not None:
                    if result['area'] >= 4_000_000:
                        big_count += 1
                estimate_seconds += estimate_time
                flood6_seconds += flood6_time
                full_flood_seconds += full_flood_time
        finally:
            with state_lock:
                pending_count -= 1
                completed_count += 1
                if completed_count == len(hits):
                    completed.set()
            pending.release()

    pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)
    cpu_started = time.perf_counter()
    for seed, grid_x, grid_z, geometry_code in hits:
        wait_started = time.perf_counter()
        pending.acquire()
        waited = time.perf_counter() - wait_started
        if waited >= 0.001:
            blocked_submissions += 1
            submit_wait_seconds += waited
        with state_lock:
            pending_count += 1
            peak_pending = max(peak_pending, pending_count)
        try:
            future = pool.submit(
                benchmark_verify_and_flood,
                seed, grid_x, grid_z, geometry_code)
            future.add_done_callback(on_done)
        except BaseException:
            with state_lock:
                pending_count -= 1
            pending.release()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    submit_seconds = time.perf_counter() - cpu_started
    completed.wait()
    pool.shutdown(wait=True)
    wall_seconds = time.perf_counter() - cpu_started

    return {
        'wall_seconds': wall_seconds,
        'submit_seconds': submit_seconds,
        'service_seconds': estimate_seconds + flood6_seconds + full_flood_seconds,
        'estimate_seconds': estimate_seconds,
        'flood6_seconds': flood6_seconds,
        'full_flood_seconds': full_flood_seconds,
        'estimated_passes': estimated_pass_count,
        'estimated_area_sum': estimated_area_sum,
        'estimated_area_max': estimated_area_max,
        'flood6_calls': flood6_calls,
        'full_flood_calls': full_flood_calls,
        'big': big_count,
        'peak_pending': peak_pending,
        'blocked_submissions': blocked_submissions,
        'submit_wait_seconds': submit_wait_seconds,
    }


def sample_hits(hits, limit):
    if limit <= 0 or len(hits) <= limit:
        return hits
    stride = len(hits) / limit
    return [hits[min(int(index * stride), len(hits) - 1)]
            for index in range(limit)]


def main():
    start_seed_text = os.environ.get('HUNT_START_SEED')
    start_seed = (
        int(start_seed_text, 0) & ((1 << 64) - 1)
        if start_seed_text else random.getrandbits(64))
    survivors = (ctypes.c_uint64 * PREF_SURVIVOR_CAP)()

    prefilter_started = time.perf_counter()
    survivor_count = prefilter_range(
        start_seed, BENCH_PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        survivors = (ctypes.c_uint64 * -survivor_count)()
        survivor_count = prefilter_range(
            start_seed, BENCH_PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        raise RuntimeError(
            f'prefilter overflow retry failed: {-survivor_count} survivors')
    prefilter_seconds = time.perf_counter() - prefilter_started

    scan_count = (
        min(survivor_count, SCAN_LIMIT) if SCAN_LIMIT > 0
        else survivor_count)
    hits, call_seconds, decode_seconds, scan_seconds, tier_timings = (
        scan_survivors_gpu(survivors, scan_count))
    cpu_hits = sample_hits(hits, CPU_HIT_LIMIT)
    cpu = benchmark_cpu(cpu_hits) if not SKIP_CPU else benchmark_cpu([])

    survivor_ratio = survivor_count / BENCH_PREF_BATCH
    gpu_hit_rate = len(hits) / call_seconds if call_seconds else 0.0
    cpu_rate = len(cpu_hits) / cpu['wall_seconds'] if cpu['wall_seconds'] else 0.0
    if SKIP_CPU:
        bottleneck = 'CPU stage skipped'
    elif cpu_rate < gpu_hit_rate:
        bottleneck = 'CPU area-estimate/flood stage'
    else:
        bottleneck = 'GPU tiered scan stage'

    print(f'Benchmark seed range: {BENCH_PREF_BATCH:,} consecutive seeds')
    print(
        f'Tiered config: G={GRID_SIZE}, step_2x={STEP_2X}, '
        f'step_05x={STEP_05X}, threshold={THRESHOLD}')
    print(f'Prefilter band: [{PREFT_LO:.5f}, {PREFT_HI:.5f}]')
    print(
        f'Prefilter: {survivor_count:,} survivors in {prefilter_seconds:.3f}s '
        f'({survivor_ratio:.3%} pass, {(1.0 - survivor_ratio):.3%} rejected)')
    print(
        f'GPU tiered: {scan_count:,}/{survivor_count:,} survivors -> '
        f'{len(hits):,} hits in {scan_seconds:.3f}s '
        f'({scan_count / scan_seconds if scan_seconds else 0:,.0f} survivors/s, '
        f'{len(hits) / scan_count if scan_count else 0:.3%} hits/survivor)')
    print(
        f'Tiered DLL calls: {call_seconds:.3f}s '
        f'({scan_count / call_seconds if call_seconds else 0:,.0f} survivors/s, '
        f'{gpu_hit_rate:,.0f} hits/s); Python decode {decode_seconds:.3f}s')
    print(
        'Tiered DLL breakdown: '
        + ', '.join(
            f'{name}={tier_timings[name]:.3f}s'
            for name in TIER_TIMING_NAMES))
    kernel_seconds = tier_timings['kernel']
    if kernel_seconds:
        print(
            f'Pure CUDA kernel: {scan_count / kernel_seconds:,.0f} survivors/s '
            f'({kernel_seconds:.3f}s)')
    if SKIP_CPU:
        print('CPU verification: skipped')
    else:
        print(
            f'CPU workers: {CPU_WORKERS}; {len(cpu_hits):,}/{len(hits):,} '
            f'sampled hits in {cpu["wall_seconds"]:.3f}s '
            f'({cpu_rate:,.0f} hits/s)')
        print(
            f'CPU results: {cpu["estimated_passes"]:,} area estimates passed, '
            f'{cpu["flood6_calls"]:,} 6-octave floods, '
            f'{cpu["full_flood_calls"]:,} full floods, '
            f'{cpu["big"]:,} >=4M')
        print(
            f'CPU service time: {cpu["service_seconds"]:.3f}s '
            f'(estimate {cpu["estimate_seconds"]:.3f}s, '
            f'6-octave flood {cpu["flood6_seconds"]:.3f}s, '
            f'full flood {cpu["full_flood_seconds"]:.3f}s)')
    if cpu_hits and not SKIP_CPU:
        print(
            f'CPU estimate: mean {cpu["estimated_area_sum"] / len(cpu_hits):,.0f}, '
            f'max {cpu["estimated_area_max"]:,.0f}, '
            f'gate {ESTIMATE_TARGET_AREA:,}')
    if not SKIP_CPU:
        print(
            f'CPU queue: peak {cpu["peak_pending"]}/{MAX_PENDING} pending, '
            f'{cpu["blocked_submissions"]:,} blocked submissions, '
            f'{cpu["submit_wait_seconds"]:.3f}s waiting')
    print(f'Backlog bottleneck: {bottleneck}')


if __name__ == '__main__':
    try:
        main()
    finally:
        _hunt.hunt_cleanup()

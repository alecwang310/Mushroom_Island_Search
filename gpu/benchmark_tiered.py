"""Benchmark the prefilter, GPU tiered scan, and CPU verification stages."""

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
    PREF_BATCH,
    PREF_SURVIVOR_CAP,
    PREFT_HI,
    PREFT_LO,
    _hunt,
    prefilter_range,
    verify_and_flood,
)


STEP_1X = 140
STEP_2X = 280
GRID_SIZE = 512
CPU_WORKERS = int(os.environ.get('HUNT_CPU_WORKERS', '24'))
MAX_PENDING = CPU_WORKERS * 2
CPU_HIT_LIMIT = int(os.environ.get('HUNT_CPU_HIT_LIMIT', '0'))


def scan_survivors_gpu(seeds, seed_count):
    hits = []
    hit_capacity = INITIAL_HIT_CAP
    results = (ctypes.c_int64 * (hit_capacity * 3))()
    kernel_seconds = 0.0
    scan_started = time.perf_counter()

    for offset in range(0, seed_count, CHUNK):
        chunk_size = min(CHUNK, seed_count - offset)
        seed_ptr = ctypes.cast(
            ctypes.byref(seeds, offset * ctypes.sizeof(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_uint64))
        kernel_started = time.perf_counter()
        hit_count = _hunt.tiered_scan_mem(
            seed_ptr, chunk_size, STEP_2X, GRID_SIZE,
            hit_capacity, results)
        kernel_seconds += time.perf_counter() - kernel_started

        if hit_count < 0:
            hit_capacity = -hit_count
            results = (ctypes.c_int64 * (hit_capacity * 3))()
            kernel_started = time.perf_counter()
            hit_count = _hunt.tiered_scan_mem(
                seed_ptr, chunk_size, STEP_2X, GRID_SIZE,
                hit_capacity, results)
            kernel_seconds += time.perf_counter() - kernel_started
        if hit_count < 0:
            raise RuntimeError(f'GPU hit buffer overflow retry failed: {-hit_count}')

        for hit_index in range(hit_count):
            hit_offset = hit_index * 3
            hits.append((
                int(results[hit_offset]),
                int(results[hit_offset + 1]),
                int(results[hit_offset + 2]),
            ))

    return hits, kernel_seconds, time.perf_counter() - scan_started


def benchmark_cpu(hits):
    if not hits:
        return {
            'wall_seconds': 0.0,
            'submit_seconds': 0.0,
            'service_seconds': 0.0,
            'verify_seconds': 0.0,
            'flood_seconds': 0.0,
            'verified': 0,
            'flooded': 0,
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
    verified_count = 0
    flooded_count = 0
    big_count = 0
    verify_seconds = 0.0
    flood_seconds = 0.0
    blocked_submissions = 0
    submit_wait_seconds = 0.0

    def on_done(future):
        nonlocal pending_count, completed_count, verified_count
        nonlocal flooded_count, big_count, verify_seconds, flood_seconds
        try:
            verified, result, verify_time, flood_time = future.result()
            with state_lock:
                if verified:
                    verified_count += 1
                if result is not None:
                    flooded_count += 1
                    if result['area'] >= 4_000_000:
                        big_count += 1
                verify_seconds += verify_time
                flood_seconds += flood_time
        finally:
            with state_lock:
                pending_count -= 1
                completed_count += 1
                if completed_count == len(hits):
                    completed.set()
            pending.release()

    pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)
    cpu_started = time.perf_counter()
    for seed, grid_x, grid_z in hits:
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
                verify_and_flood, seed, grid_x, grid_z, STEP_1X, STEP_2X)
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
        'service_seconds': verify_seconds + flood_seconds,
        'verify_seconds': verify_seconds,
        'flood_seconds': flood_seconds,
        'verified': verified_count,
        'flooded': flooded_count,
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
    start_seed = random.getrandbits(64)
    survivors = (ctypes.c_uint64 * PREF_SURVIVOR_CAP)()

    prefilter_started = time.perf_counter()
    survivor_count = prefilter_range(
        start_seed, PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        survivors = (ctypes.c_uint64 * -survivor_count)()
        survivor_count = prefilter_range(
            start_seed, PREF_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        raise RuntimeError(
            f'prefilter overflow retry failed: {-survivor_count} survivors')
    prefilter_seconds = time.perf_counter() - prefilter_started

    hits, kernel_seconds, scan_seconds = scan_survivors_gpu(
        survivors, survivor_count)
    cpu_hits = sample_hits(hits, CPU_HIT_LIMIT)
    cpu = benchmark_cpu(cpu_hits)

    survivor_ratio = survivor_count / PREF_BATCH
    gpu_hit_rate = len(hits) / scan_seconds if scan_seconds else 0.0
    cpu_rate = len(cpu_hits) / cpu['wall_seconds'] if cpu['wall_seconds'] else 0.0
    if cpu_rate < gpu_hit_rate:
        bottleneck = 'CPU verification/flood stage'
    else:
        bottleneck = 'GPU tiered scan stage'

    print(f'Benchmark seed range: {PREF_BATCH:,} consecutive seeds')
    print(f'Prefilter band: [{PREFT_LO:.5f}, {PREFT_HI:.5f}]')
    print(
        f'Prefilter: {survivor_count:,} survivors in {prefilter_seconds:.3f}s '
        f'({survivor_ratio:.3%} pass, {(1.0 - survivor_ratio):.3%} rejected)')
    print(
        f'GPU tiered: {survivor_count:,} survivors -> {len(hits):,} hits '
        f'in {scan_seconds:.3f}s ({gpu_hit_rate:,.0f} hits/s)')
    print(
        f'GPU kernel/transfer calls: {kernel_seconds:.3f}s '
        f'({survivor_count / kernel_seconds:,.0f} survivors/s)')
    print(
        f'CPU workers: {CPU_WORKERS}; {len(cpu_hits):,}/{len(hits):,} sampled hits in '
        f'{cpu["wall_seconds"]:.3f}s ({cpu_rate:,.0f} hits/s)')
    print(
        f'CPU results: {cpu["verified"]:,} verified, '
        f'{cpu["flooded"]:,} flood fills, {cpu["big"]:,} >=4M')
    print(
        f'CPU service time: {cpu["service_seconds"]:.3f}s '
        f'(verify {cpu["verify_seconds"]:.3f}s, '
        f'flood {cpu["flood_seconds"]:.3f}s)')
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

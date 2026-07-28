"""Benchmark the translated O6+O15 GPU estimator without CPU verification."""

import ctypes
import hashlib
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault('HUNT_DEBUG_TRANSLATION_STATS', '1')

from hunt_tiered import (
    CHUNK,
    GPU_ESTIMATE_STEP_1X,
    GPU_ESTIMATE_STEP_2X,
    GPU_ESTIMATE_TARGET_AREA,
    GPU_ESTIMATE_THRESHOLD,
    GPU_GRID_SIZE,
    INITIAL_HIT_CAP,
    O6_STEP_2X,
    O6_THRESHOLD,
    O6_TRANSLATION_PERIOD,
    PREF_BATCH,
    PREF_SURVIVOR_CAP,
    PREFT_HI,
    PREFT_LO,
    WORLD_BORDER_PIPELINE,
    _hunt,
    cap_o6_grid_size,
    prefilter_range,
)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value, 0) if value else default


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


START_SEED = env_int('HUNT_START_SEED', 0x123456789ABCDEF0)
PREFILTER_BATCH = env_int('HUNT_PREF_BATCH', PREF_BATCH)
SCAN_LIMIT = env_int('HUNT_SCAN_LIMIT', 1600)
O6_STEP = env_int('HUNT_O6_STEP', O6_STEP_2X)
O6_CUTOFF = env_float('HUNT_O6_THRESHOLD', O6_THRESHOLD)
TRANSLATION_PERIOD = env_int(
    'HUNT_TRANSLATION_PERIOD', O6_TRANSLATION_PERIOD)
WORLD_BORDER = env_int(
    'HUNT_WORLD_BORDER_PIPELINE', WORLD_BORDER_PIPELINE)
ESTIMATE_STEP_1X = env_int(
    'HUNT_TRANSLATION_ESTIMATE_STEP', GPU_ESTIMATE_STEP_1X)
ESTIMATE_STEP_2X = env_int(
    'HUNT_TRANSLATION_ESTIMATE_STEP_2X', GPU_ESTIMATE_STEP_2X)
ESTIMATE_THRESHOLD = env_float(
    'HUNT_TRANSLATION_ESTIMATE_THRESHOLD', GPU_ESTIMATE_THRESHOLD)
ESTIMATE_TARGET = env_int(
    'HUNT_TRANSLATION_ESTIMATE_TARGET', GPU_ESTIMATE_TARGET_AREA)
GRID_SIZE = cap_o6_grid_size(
    env_int('HUNT_GRID_SIZE', GPU_GRID_SIZE), O6_STEP, TRANSLATION_PERIOD)


def result_digest(hits):
    digest = hashlib.sha256()
    for seed, x, z, geometry in sorted(hits):
        digest.update(struct.pack(
            '<Qqqq', seed & 0xFFFFFFFFFFFFFFFF, x, z, geometry))
    return digest.hexdigest()


def run_translated(survivors, survivor_count):
    hits = []
    dll_seconds = 0.0
    decode_seconds = 0.0
    for offset in range(0, survivor_count, CHUNK):
        chunk_size = min(CHUNK, survivor_count - offset)
        seed_pointer = ctypes.cast(
            ctypes.byref(
                survivors, offset * ctypes.sizeof(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.c_uint64))
        hit_capacity = INITIAL_HIT_CAP
        while True:
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            started = time.perf_counter()
            hit_count = _hunt.tiered_scan_mem_translated(
                seed_pointer, ctypes.c_int(chunk_size),
                ctypes.c_int(O6_STEP), ctypes.c_int(GRID_SIZE),
                ctypes.c_float(O6_CUTOFF),
                ctypes.c_int(TRANSLATION_PERIOD), ctypes.c_int(WORLD_BORDER),
                ctypes.c_int(ESTIMATE_STEP_1X),
                ctypes.c_int(ESTIMATE_STEP_2X),
                ctypes.c_int64(ESTIMATE_TARGET),
                ctypes.c_int(hit_capacity), results)
            dll_seconds += time.perf_counter() - started
            if hit_count >= 0:
                break
            hit_capacity = -hit_count

        started = time.perf_counter()
        for hit_index in range(hit_count):
            result_index = hit_index * 4
            hits.append((
                int(results[result_index]),
                int(results[result_index + 1]),
                int(results[result_index + 2]),
                int(results[result_index + 3]),
            ))
        decode_seconds += time.perf_counter() - started
    return hits, dll_seconds, decode_seconds


def main():
    survivors = (ctypes.c_uint64 * PREF_SURVIVOR_CAP)()
    started = time.perf_counter()
    survivor_count = prefilter_range(
        START_SEED, PREFILTER_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        survivors = (ctypes.c_uint64 * -survivor_count)()
        survivor_count = prefilter_range(
            START_SEED, PREFILTER_BATCH, PREFT_LO, PREFT_HI, survivors)
    if survivor_count < 0:
        raise RuntimeError(
            f'prefilter overflow retry failed: {-survivor_count} survivors')
    prefilter_seconds = time.perf_counter() - started
    scan_count = min(survivor_count, SCAN_LIMIT) if SCAN_LIMIT else survivor_count

    started = time.perf_counter()
    hits, dll_seconds, decode_seconds = run_translated(
        survivors, scan_count)
    wall_seconds = time.perf_counter() - started

    print(f'Start seed: 0x{START_SEED:016X}')
    print(f'Prefilter: {PREFILTER_BATCH:,} seeds -> {survivor_count:,} survivors '
          f'in {prefilter_seconds:.3f}s')
    print(f'O6 scan: {scan_count:,} survivors, G={GRID_SIZE}, step={O6_STEP}, '
          f'threshold<{O6_CUTOFF}')
    print(f'Translated estimate: O6+O15<{ESTIMATE_THRESHOLD}, '
          f'step={ESTIMATE_STEP_1X}, target={ESTIMATE_TARGET:,}, '
          f'mode={os.environ.get("HUNT_TRANSLATION_GROUPED", "1")}, '
          f'threads={os.environ.get("HUNT_TRANSLATION_GROUPED_THREADS", "256")}')
    print(f'Result: {len(hits):,} candidates in {wall_seconds:.3f}s '
          f'({scan_count / wall_seconds if wall_seconds else 0:,.0f} '
          f'survivors/s)')
    print(f'DLL: {dll_seconds:.3f}s; Python decode: {decode_seconds:.3f}s')
    print(f'Unique candidates: {len(set(hits)):,}')
    print(f'Sorted result SHA-256: {result_digest(hits)}')


if __name__ == '__main__':
    try:
        main()
    finally:
        _hunt.hunt_cleanup()

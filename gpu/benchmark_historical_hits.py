"""Measure current triple-filter retention against historical islands."""

import argparse
import ctypes
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hunt_tiered import (
    ESTIMATE_TARGET_AREA,
    INITIAL_HIT_CAP,
    estimate_triple_area,
    _hunt,
)


UINT64_MASK = (1 << 64) - 1
DEFAULT_STEP_05X = 70
DEFAULT_STEP_2X = 280
DEFAULT_GRID_SIZE = 512
DEFAULT_CHUNK_SIZE = 8192


def seed_key(seed):
    return int(seed) & UINT64_MASK


def load_records(path, limit):
    records = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if limit and len(records) >= limit:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
            for field in ('seed', 'area', 'cx', 'cz'):
                if field not in record:
                    raise ValueError(f'{path}:{line_number}: missing {field}')
            records.append({
                'seed': int(record['seed']),
                'area': int(record['area']),
                'cx': int(record['cx']),
                'cz': int(record['cz']),
            })
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def scan_hits(seed_values, step_2x, grid_size, chunk_size):
    hits = []
    hit_capacity = INITIAL_HIT_CAP
    scan_started = time.perf_counter()

    for offset in range(0, len(seed_values), chunk_size):
        chunk = seed_values[offset:offset + chunk_size]
        seed_buffer = (ctypes.c_uint64 * len(chunk))(
            *(seed_key(seed) for seed in chunk))
        results = (ctypes.c_int64 * (hit_capacity * 4))()
        hit_count = _hunt.tiered_scan_mem(
            seed_buffer, len(chunk), step_2x, grid_size,
            hit_capacity, results)
        if hit_count < 0:
            hit_capacity = -hit_count
            results = (ctypes.c_int64 * (hit_capacity * 4))()
            hit_count = _hunt.tiered_scan_mem(
                seed_buffer, len(chunk), step_2x, grid_size,
                hit_capacity, results)
        if hit_count < 0:
            raise RuntimeError(
                f'tiered scan overflow retry failed: {-hit_count}')

        for hit_index in range(hit_count):
            result_offset = hit_index * 4
            hits.append((
                int(results[result_offset]) & UINT64_MASK,
                int(results[result_offset + 1]),
                int(results[result_offset + 2]),
                int(results[result_offset + 3]),
            ))

    return hits, time.perf_counter() - scan_started


def area_bucket(area):
    if area >= 4_000_000:
        return '4M+'
    if area >= 3_500_000:
        return '3.5M-4M'
    return '3M-3.5M'


def nearest_hit(record, seed_hits):
    if not seed_hits:
        return None
    return min(
        seed_hits,
        key=lambda hit: math.hypot(hit[1] - record['cx'], hit[2] - record['cz']))


def evaluate_records(records, hits, match_radius, step_05x, step_2x, estimate):
    hits_by_seed = {}
    for hit in hits:
        hits_by_seed.setdefault(hit[0], []).append(hit)

    results = []
    estimate_passes = 0
    estimate_values = []
    for record in records:
        seed_hits = hits_by_seed.get(seed_key(record['seed']), ())
        nearest = nearest_hit(record, seed_hits)
        distance = None
        if nearest is not None:
            distance = math.hypot(
                nearest[1] - record['cx'], nearest[2] - record['cz'])
        local = distance is not None and distance <= match_radius
        entry = {
            'seed': record['seed'],
            'area': record['area'],
            'cx': record['cx'],
            'cz': record['cz'],
            'seed_hit': bool(seed_hits),
            'exact_hit': distance == 0,
            'within_one_coarse_step': (
                distance is not None and distance <= step_2x),
            'local_hit': local,
            'nearest_distance': distance,
        }
        if estimate and local:
            estimated_area = estimate_triple_area(
                record['seed'], nearest[1], nearest[2], nearest[3],
                step_05x, step_2x)
            entry['estimated_area'] = estimated_area
            entry['estimate_pass'] = estimated_area >= ESTIMATE_TARGET_AREA
            estimate_values.append((record['area'], estimated_area))
            if entry['estimate_pass']:
                estimate_passes += 1
        results.append(entry)

    return results, estimate_passes, estimate_values


def count_records(entries, predicate):
    return sum(1 for entry in entries if predicate(entry))


def print_summary(records, hits, entries, scan_seconds, step_2x, match_radius,
                  estimate, estimate_passes, estimate_values):
    unique_historical_seeds = {seed_key(record['seed']) for record in records}
    current_hit_seeds = {hit[0] for hit in hits}
    seed_retained = {
        seed for seed in unique_historical_seeds if seed in current_hit_seeds}
    local_count = count_records(entries, lambda entry: entry['local_hit'])
    one_step_count = count_records(
        entries, lambda entry: entry['within_one_coarse_step'])
    exact_count = count_records(entries, lambda entry: entry['exact_hit'])

    print(f'Historical records: {len(records):,}')
    print(f'Historical unique seeds: {len(unique_historical_seeds):,}')
    print(f'Current GPU hits: {len(hits):,} from {len(current_hit_seeds):,} seeds')
    print(f'GPU scan time: {scan_seconds:.3f}s')
    print(f'Exact coordinate retention: {exact_count:,}/{len(records):,}')
    print(f'Within one coarse step ({step_2x}): '
          f'{one_step_count:,}/{len(records):,}')
    print(f'Within match radius ({match_radius:g}): '
          f'{local_count:,}/{len(records):,}')
    print(f'Seed-level retention: {len(seed_retained):,}/'
          f'{len(unique_historical_seeds):,}')
    print(f'Coordinate-local sacrificed: {len(records) - local_count:,}/'
          f'{len(records):,}')

    print('\nArea bucket retention:')
    print('  bucket       records  seed-hit  local-hit  sacrificed')
    for bucket in ('3M-3.5M', '3.5M-4M', '4M+'):
        bucket_entries = [
            entry for entry in entries if area_bucket(entry['area']) == bucket]
        seed_hit_count = count_records(
            bucket_entries, lambda entry: entry['seed_hit'])
        local_hit_count = count_records(
            bucket_entries, lambda entry: entry['local_hit'])
        print(f'  {bucket:10} {len(bucket_entries):8,} '
              f'{seed_hit_count:9,} {local_hit_count:10,} '
              f'{len(bucket_entries) - local_hit_count:10,}')

    if estimate:
        actual_big = sum(
            1 for actual, _ in estimate_values if actual >= ESTIMATE_TARGET_AREA)
        estimate_false_positive = sum(
            1 for actual, estimated in estimate_values
            if estimated >= ESTIMATE_TARGET_AREA and actual < ESTIMATE_TARGET_AREA)
        estimate_false_negative = sum(
            1 for actual, estimated in estimate_values
            if actual >= ESTIMATE_TARGET_AREA and estimated < ESTIMATE_TARGET_AREA)
        mean_estimate = (
            sum(estimated for _, estimated in estimate_values)
            / len(estimate_values)
            if estimate_values else 0.0)
        print('\nEstimator diagnostics on coordinate-local matches:')
        print(f'  evaluated: {len(estimate_values):,}')
        print(f'  estimate passes: {estimate_passes:,}')
        print(f'  actual >=4M: {actual_big:,}')
        print(f'  false positives: {estimate_false_positive:,}')
        print(f'  false negatives: {estimate_false_negative:,}')
        print(f'  mean estimate: {mean_estimate:,.0f}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure current GPU triple-filter retention.')
    parser.add_argument('--input', required=True,
                        help='historical islands JSONL file')
    parser.add_argument('--grid-size', type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument('--step-2x', type=int, default=DEFAULT_STEP_2X)
    parser.add_argument('--step-05x', type=int, default=DEFAULT_STEP_05X)
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--match-radius', type=float, default=None,
                        help='coordinate-local radius; defaults to 2*step-2x')
    parser.add_argument('--limit', type=int, default=0,
                        help='optional historical record limit')
    parser.add_argument('--estimate', action='store_true',
                        help='run the current CPU estimator on local matches')
    parser.add_argument('--report',
                        help='optional JSON report path')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_size <= 0 or args.chunk_size > 8192:
        raise ValueError('--chunk-size must be between 1 and 8192')
    match_radius = (
        2 * args.step_2x if args.match_radius is None else args.match_radius)
    records = load_records(args.input, args.limit)
    seed_values = sorted({record['seed'] for record in records})

    try:
        hits, scan_seconds = scan_hits(
            seed_values, args.step_2x, args.grid_size, args.chunk_size)
        entries, estimate_passes, estimate_values = evaluate_records(
            records, hits, match_radius, args.step_05x, args.step_2x,
            args.estimate)
        print_summary(
            records, hits, entries, scan_seconds, args.step_2x, match_radius,
            args.estimate, estimate_passes, estimate_values)
        if args.report:
            report = {
                'input': args.input,
                'grid_size': args.grid_size,
                'step_2x': args.step_2x,
                'match_radius': match_radius,
                'scan_seconds': scan_seconds,
                'hits': len(hits),
                'records': entries,
            }
            with open(args.report, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2)
                handle.write('\n')
    finally:
        _hunt.hunt_cleanup()


if __name__ == '__main__':
    main()

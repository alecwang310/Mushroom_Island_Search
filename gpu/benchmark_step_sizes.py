"""Compare GPU triple-filter spacing on known positives and controls."""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_historical_hits import (
    load_records,
    scan_hits,
    seed_key,
)
from hunt_tiered import _hunt


DEFAULT_STEPS = (140, 210, 280, 350, 420, 560)
DEFAULT_GRID_SIZE = 512
DEFAULT_CONTROL_COUNT = 10_000
DEFAULT_MATCH_RADIUS = 560
DEFAULT_RANDOM_SEED = 0x4D555348524F4F4D


def build_control_seeds(count, excluded):
    generator = random.Random(DEFAULT_RANDOM_SEED)
    controls = []
    excluded = {seed_key(seed) for seed in excluded}
    while len(controls) < count:
        candidate = generator.getrandbits(64)
        if candidate in excluded:
            continue
        excluded.add(candidate)
        if candidate >= (1 << 63):
            candidate -= 1 << 64
        controls.append(candidate)
    return controls


def local_record_retention(records, hits, radius):
    hits_by_seed = {}
    for hit in hits:
        hits_by_seed.setdefault(hit[0], []).append(hit)

    radius_squared = radius * radius
    retained = 0
    for record in records:
        for hit in hits_by_seed.get(seed_key(record['seed']), ()):
            dx = hit[1] - record['cx']
            dz = hit[2] - record['cz']
            if dx * dx + dz * dz <= radius_squared:
                retained += 1
                break
    return retained


def run(args):
    all_records = load_records(args.input, 0)
    positive_records = [
        record for record in all_records if record['area'] >= args.min_area]
    positive_seeds = sorted({record['seed'] for record in positive_records})
    control_seeds = build_control_seeds(args.control_count, positive_seeds)

    report = {
        'input': args.input,
        'min_area': args.min_area,
        'grid_size': args.grid_size,
        'match_radius': args.match_radius,
        'control_count': len(control_seeds),
        'steps': [],
    }

    print(f'Positive records: {len(positive_records):,}')
    print(f'Positive unique seeds: {len(positive_seeds):,}')
    print(f'Random control seeds: {len(control_seeds):,}')
    print(f'Grid size: {args.grid_size}; local match radius: {args.match_radius}')
    print()
    print(' step  coverage    gpu_hits  pos_seed  pos_local  ctl_pass  ctl_pruned')

    try:
        for step_2x in args.steps:
            positive_hits, positive_seconds = scan_hits(
                positive_seeds, step_2x, args.grid_size, args.chunk_size)
            control_hits, control_seconds = scan_hits(
                control_seeds, step_2x, args.grid_size, args.chunk_size)

            positive_hit_seeds = {hit[0] for hit in positive_hits}
            control_hit_seeds = {hit[0] for hit in control_hits}
            positive_seed_count = sum(
                seed_key(seed) in positive_hit_seeds for seed in positive_seeds)
            positive_local_count = local_record_retention(
                positive_records, positive_hits, args.match_radius)
            control_pass_count = len(control_hit_seeds)
            control_pruned_count = len(control_seeds) - control_pass_count
            result = {
                'step_2x': step_2x,
                'coverage': (args.grid_size - 1) * step_2x,
                'gpu_hits': len(positive_hits),
                'positive_seed_count': len(positive_seeds),
                'positive_seed_retained': positive_seed_count,
                'positive_record_local_retained': positive_local_count,
                'control_count': len(control_seeds),
                'control_passed': control_pass_count,
                'control_pruned': control_pruned_count,
                'control_prune_fraction': (
                    control_pruned_count / len(control_seeds)
                    if control_seeds else 0.0),
                'positive_scan_seconds': positive_seconds,
                'control_scan_seconds': control_seconds,
            }
            report['steps'].append(result)
            print(f'{step_2x:5d} {result["coverage"]:9,d} '
                  f'{len(positive_hits):9,d} '
                  f'{positive_seed_count:9,d}/{len(positive_seeds):d} '
                  f'{positive_local_count:10,d}/{len(positive_records):d} '
                  f'{control_pass_count:9,d} {control_pruned_count:10,d}')
    finally:
        _hunt.hunt_cleanup()

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare GPU triple-filter spacing.')
    parser.add_argument('--input', required=True,
                        help='historical islands JSONL file')
    parser.add_argument('--steps', type=int, nargs='+', default=DEFAULT_STEPS)
    parser.add_argument('--min-area', type=int, default=4_000_000)
    parser.add_argument('--grid-size', type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument('--control-count', type=int,
                        default=DEFAULT_CONTROL_COUNT)
    parser.add_argument('--match-radius', type=float,
                        default=DEFAULT_MATCH_RADIUS)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--report')
    args = parser.parse_args()
    if any(step <= 0 for step in args.steps):
        parser.error('--steps values must be positive')
    if args.control_count < 0:
        parser.error('--control-count must be non-negative')
    if args.chunk_size <= 0 or args.chunk_size > 8192:
        parser.error('--chunk-size must be between 1 and 8192')
    return args


if __name__ == '__main__':
    run(parse_args())

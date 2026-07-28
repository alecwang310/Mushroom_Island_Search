"""Benchmark the compressed full-scale O6/O15 hash-LUT experiment."""
import argparse
import ctypes
import os

GPU_DIR = os.path.dirname(__file__)
_dll_name = 'full_lut_benchmark.dll'
_dll = ctypes.CDLL(os.path.join(GPU_DIR, _dll_name))
_dll.full_lut_benchmark.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
]
_dll.full_lut_benchmark.restype = ctypes.c_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()

    timings = (ctypes.c_double * 8)()
    status = _dll.full_lut_benchmark(
        args.blocks, args.grid, args.repeats, timings, len(timings))
    if status != 0:
        raise RuntimeError(f'full_lut_benchmark failed with status {status}')

    build_ms, extract_ms, lut_ms, chain_ms = timings[:4]
    lut_cells_s, chain_cells_s, speedup, build_entries_s = timings[4:]
    total_cells = args.blocks * args.grid * args.grid
    print(f'Compressed LUT: 2 octaves x 65,536 entries x 4 bytes = 512 KiB')
    print(f'Workload: {args.blocks:,} blocks x {args.grid}x{args.grid} cells '
          f'= {total_cells:,} cells/repeat')
    print(f'LUT build: {build_ms:.4f} ms/seed '
          f'({build_entries_s:,.0f} entries/s)')
    print(f'LUT extraction only: {extract_ms:.4f} ms '
          f'({total_cells / (extract_ms * 1e-3):,.0f} cells/s)')
    print(f'LUT extraction + interpolation: {lut_ms:.4f} ms '
          f'({lut_cells_s:,.0f} cells/s)')
    print(f'Permutation chain + interpolation: {chain_ms:.4f} ms '
          f'({chain_cells_s:,.0f} cells/s)')
    print(f'Full LUT speedup over chain: {speedup:.3f}x')
    print('Note: the LUT is synthetic and global-memory resident; this is a '
          'lookup/scheduling experiment, not a correctness benchmark.')


if __name__ == '__main__':
    main()

"""Compare shared-memory vs warp-shuffle perm table performance."""
import ctypes, os, sys, time, math
sys.path.insert(0, '.')
from continentalness_pipeline import ContinentalnessSampler

_hunt = ctypes.CDLL(os.path.join('gpu', 'hunt_engine.dll'))

# -- shared memory version --
_hunt.hunt_batch.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_hunt.hunt_batch.restype = ctypes.c_int
_hunt.hunt_get_timings.argtypes = [ctypes.POINTER(ctypes.c_float)]
_hunt.hunt_reset_timings.argtypes = []

# -- warp shuffle version --
_hunt.hunt_batch_ws.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_hunt.hunt_batch_ws.restype = ctypes.c_int
_hunt.hunt_get_timings_ws.argtypes = [ctypes.POINTER(ctypes.c_float)]

S = 5_000_000; D = math.sqrt(S)
step = int(D/4/3)//4*4; step = max(step, 4)
G, K, batch = 256, 2, 2048

results = (ctypes.c_int64 * (batch * 3))()
timings = (ctypes.c_float * 5)()

def test(name, batch_fn, timing_fn, N=5):
    """Run N batches and return (kernel_ms, total_ms, seeds_per_sec, correctness_str)"""
    _hunt.hunt_reset_timings()
    # Warmup
    batch_fn(ctypes.c_uint64(77777), batch, step, K, G, results)
    _hunt.hunt_reset_timings()

    t0 = time.perf_counter()
    for i in range(N):
        batch_fn(ctypes.c_uint64(77777 + i*batch), batch, step, K, G, results)
    t_total = time.perf_counter() - t0

    timing_fn(timings)
    init_ms, upload_ms, kernel_ms, down_ms, nbatches = list(timings)
    total_ms = init_ms + upload_ms + kernel_ms + down_ms
    rate = batch * 1000 / total_ms

    # Correctness
    hc = batch_fn(ctypes.c_uint64(77777), 2048, step, K, G, results)
    ok = 0
    for i in range(min(hc, 30)):
        seed = int(results[i*3])
        gx, gz = int(results[i*3+1]), int(results[i*3+2])
        cs = ContinentalnessSampler(seed)
        cells = [(gx-step,gz-step), (gx,gz-step), (gx-step,gz), (gx,gz)]
        if all(cs.sample(x,z) < -1.05 for x,z in cells):
            ok += 1

    return kernel_ms, total_ms, rate, f'{ok}/{min(hc, 30)}', hc

print('=== Side-by-side comparison (G=%d, batch=%d, %d iterations) ===' % (G, batch, 5))
print()
print(f'{"Variant":<25s} {"Kernel ms":>10s} {"Total ms":>10s} {"Seeds/s":>10s} {"Correct":>10s} {"Hits":>6s}')
print('-' * 75)

for name, fn, tfn in [
    ('Shared memory (s_perm)', _hunt.hunt_batch, _hunt.hunt_get_timings),
    ('Warp shuffle (reg)', _hunt.hunt_batch_ws, _hunt.hunt_get_timings_ws),
]:
    kms, tms, rate, corr, hc = test(name, fn, tfn)
    print(f'{name:<25s} {kms:>10.2f} {tms:>10.2f} {rate:>10,.0f} {corr:>10s} {hc:>6d}')

print()
print('smem:  256B shared memory perm table (~20-cycle ld.shared per lookup)')
print('shfl:  perm in registers, __shfl_sync lookup (~5-cycle, dual-shuffle')
print('       to prevent register spilling on sm_120). No shared-memory perm.')

"""Tight-loop C++ hunt engine wrapper. Python only does flood fill."""
import ctypes, numpy as np, time, json, sys, os
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'hunt_engine.dll'))
lib.hunt_batch.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64)]
lib.hunt_batch.restype = ctypes.c_int
lib.hunt_cleanup.argtypes = []; lib.hunt_cleanup.restype = None

ENG_SZ = 8192

def flood_fill(seed, cx, cz):
    buf = (ctypes.c_ubyte * ENG_SZ)()
    _eng._lib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
    def s(x, z): return _eng._lib.cont_sample(buf, x, z)
    sx, sz = int(cx), int(cz)
    if s(sx, sz) >= -1.05: return 0
    q = deque([(sx, sz)]); seen = {(sx, sz)}; cache = {(sx, sz): s(sx, sz)}
    cells = 0
    while q and cells < 2_000_000:
        x, z = q.popleft(); cells += 1
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, nz = x + dx, z + dz
            if (nx, nz) in seen: continue
            seen.add((nx, nz)); cache[(nx, nz)] = s(nx, nz)
            if cache[(nx, nz)] < -1.05: q.append((nx, nz))
    return cells * 16

if __name__ == '__main__':
    step, K, G, NG = 184, 2, 4, 128
    radius, batch = 40000, 4096

    best, scanned, t0 = None, 0, time.perf_counter()
    hit_buf = np.zeros(batch * 3, dtype=np.int64)

    try:
        while True:
            seeds = np.random.randint(0, 2**63, size=batch, dtype=np.uint64)
            n_hits = lib.hunt_batch(
                seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), batch,
                step, K, G, NG, radius,
                hit_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
            scanned += batch
            elapsed = time.perf_counter() - t0
            rate = scanned / elapsed

            for i in range(min(n_hits, 5)):  # limit flood fills
                s = int(hit_buf[i*3]); x = int(hit_buf[i*3+1]); z = int(hit_buf[i*3+2])
                area = flood_fill(s, x, z)
                if best is None or area > best['area']:
                    best = {'seed': s, 'area': area, 'center': (x, z)}
                    print(f'\n  *** NEW BEST: {s}  {area:,} at ({x},{z}) ***')
                    with open('best_1m.json', 'w') as f: json.dump(best, f)
                if area >= 2_000_000:
                    with open('big_islands.jsonl', 'a') as f:
                        f.write(json.dumps({'seed': s, 'area': area,
                            'center_1_4': [x, z], 'center_1_1': [x*4, z*4]}) + '\n')

            if scanned % 200000 == 0:
                ba = best['area'] if best else 0
                print(f'\r  {scanned:,} ({rate:.0f}/s) hits={n_hits} best={ba:,}   ', end='')
    except KeyboardInterrupt:
        pass
    finally:
        lib.hunt_cleanup()
        print(f'\n{scanned:,} in {time.perf_counter()-t0:.0f}s = {scanned/(time.perf_counter()-t0):.0f} seeds/s')

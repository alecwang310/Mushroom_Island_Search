"""
gpu/sparse_scanner.py — GPU sparse-grid mushroom scanner.
Uses ctypes to call CUDA runtime + custom sparse_kernel.dll.

Usage:
    scanner = GPUSparseScanner(target_area=100_000)
    hits = scanner.scan(seeds_array)
    for seed, x, z in hits:
        area = scanner.flood_fill(seed, x, z)
"""

import ctypes, os, sys, time, math
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hyperparams import derive

# ---- Load libraries ----
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'sparse_kernel.dll'))
# gpu_scan_seeds(d_perm, d_oa, d_oc, d_amp, d_lac, d_h2, d_d2, d_t2,
#                 d_ranges, d_dbl_amps, num_seeds,
#                 d_grid_offsets, num_grids, G, step, K,
#                 d_hit_flags, d_hit_x, d_hit_z, d_hit_grid)
_lib.gpu_scan_seeds.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.gpu_scan_seeds.restype = ctypes.c_int

_cuda = ctypes.CDLL('cudart.dll')
_cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cuda.cudaMalloc.restype = ctypes.c_int
_cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cuda.cudaMemcpy.restype = ctypes.c_int
_cuda.cudaFree.argtypes = [ctypes.c_void_p]
_cuda.cudaFree.restype = ctypes.c_int
_cuda.cudaDeviceSynchronize.restype = ctypes.c_int

cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost = 1, 2

# ---- C engine ----
_edir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'engine')
_elib = ctypes.CDLL(os.path.join(_edir, 'continentalness.dll'))
_CONT_SIZE = 8192; _MAX_OCT = 24; _PERM = 257
_elib.cont_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
_elib.cont_sample.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_elib.cont_sample.restype = ctypes.c_double

# ---- Helpers ----
def _malloc(arr):
    p = ctypes.c_void_p()
    _cuda.cudaMalloc(ctypes.byref(p), arr.nbytes)
    _cuda.cudaMemcpy(p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, cudaMemcpyHostToDevice)
    return p

def _malloc_zero(size):
    p = ctypes.c_void_p()
    _cuda.cudaMalloc(ctypes.byref(p), size)
    return p

def _to_host(d_p, arr):
    _cuda.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), d_p, arr.nbytes, cudaMemcpyDeviceToHost)


def _extract(buf: bytes) -> dict:
    off=0
    perm=np.frombuffer(buf,dtype=np.uint8, count=_MAX_OCT*_PERM,offset=off); off+=_MAX_OCT*_PERM
    oa =np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    ob =np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    oc =np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    amp=np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    lac=np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    h2 =np.frombuffer(buf,dtype=np.uint8, count=_MAX_OCT,offset=off); off+=_MAX_OCT
    d2 =np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    t2 =np.frombuffer(buf,dtype=np.float64,count=_MAX_OCT,offset=off).astype(np.float32); off+=_MAX_OCT*8
    rng=np.frombuffer(buf,dtype=np.int32,  count=8,offset=off); off+=32
    dbl=np.frombuffer(buf,dtype=np.float64,count=2,offset=off).astype(np.float32)
    return {'perm':perm,'oa':oa,'oc':oc,'amp':amp,'lac':lac,
            'h2':h2,'d2':d2,'t2':t2,'rng':rng,'dbl':dbl}


class GPUSparseScanner:
    def __init__(self, target_area: float = 100_000, batch_size: int = 1024):
        p = derive(target_area)
        self.step = p['step']
        self.K = p['kernel_size']
        self.G = p['grid_cells']
        self.N = p['num_grids']
        self.offsets = np.array(p['grid_offsets'], dtype=np.int32)
        self.batch = batch_size
        self._offsets_d = _malloc(self.offsets)
        print(f"GPU S={target_area:,.0f}: step={self.step} K={self.K} "
              f"G={self.G} grids={self.N} batch={batch_size}")

    def _init_batch(self, seeds: np.ndarray) -> dict:
        n = len(seeds)
        d = {
            'perm': np.zeros((n,_MAX_OCT,_PERM),dtype=np.uint8),
            'oa':   np.zeros((n,_MAX_OCT),dtype=np.float32),
            'oc':   np.zeros((n,_MAX_OCT),dtype=np.float32),
            'amp':  np.zeros((n,_MAX_OCT),dtype=np.float32),
            'lac':  np.zeros((n,_MAX_OCT),dtype=np.float32),
            'h2':   np.zeros((n,_MAX_OCT),dtype=np.uint8),
            'd2':   np.zeros((n,_MAX_OCT),dtype=np.float32),
            't2':   np.zeros((n,_MAX_OCT),dtype=np.float32),
            'rng':  np.zeros((n,8),dtype=np.int32),
            'dbl':  np.zeros((n,2),dtype=np.float32),
        }
        buf = (ctypes.c_ubyte * _CONT_SIZE)()
        for i in range(n):
            _elib.cont_engine_init(buf, int(seeds[i]) & 0xFFFFFFFFFFFFFFFF, 0)
            e = _extract(bytes(buf))
            d['perm'][i]=e['perm'].reshape(_MAX_OCT,_PERM)
            d['oa'][i]=e['oa']; d['oc'][i]=e['oc']
            d['amp'][i]=e['amp']; d['lac'][i]=e['lac']
            d['h2'][i]=e['h2']; d['d2'][i]=e['d2']; d['t2'][i]=e['t2']
            d['rng'][i]=e['rng']; d['dbl'][i]=e['dbl']
        return d

    def scan(self, seeds: np.ndarray) -> list:
        n = len(seeds)
        data = self._init_batch(seeds)

        # Upload perlin data
        d = {k: _malloc(v) for k, v in data.items()}

        # Output buffers
        hf = np.zeros(n, dtype=np.int32); hx = np.zeros(n, dtype=np.int32)
        hz = np.zeros(n, dtype=np.int32); hg = np.zeros(n, dtype=np.int32)
        df = _malloc_zero(hf.nbytes); dx = _malloc_zero(hx.nbytes)
        dz = _malloc_zero(hz.nbytes); dg = _malloc_zero(hg.nbytes)

        _lib.gpu_scan_seeds(
            d['perm'], d['oa'], d['oc'], d['amp'], d['lac'],
            d['h2'], d['d2'], d['t2'], d['rng'], d['dbl'], n,
            self._offsets_d, self.N, self.G, self.step, self.K,
            df, dx, dz, dg)

        _to_host(df, hf); _to_host(dx, hx); _to_host(dz, hz)

        # Free GPU memory
        for p in d.values(): _cuda.cudaFree(p)
        for p in [df, dx, dz, dg]: _cuda.cudaFree(p)

        return [(int(seeds[i]), int(hx[i]), int(hz[i]))
                for i in range(n) if hf[i]]

    def flood_fill(self, seed: int, cx: int, cz: int) -> int:
        buf = (ctypes.c_ubyte * _CONT_SIZE)()
        _elib.cont_engine_init(buf, seed & 0xFFFFFFFFFFFFFFFF, 0)
        def s(x,z): return _elib.cont_sample(buf, x, z)
        sx, sz = (cx//4)*4, (cz//4)*4
        if s(sx, sz) >= -1.05: return 0
        q = deque([(sx, sz)]); seen = {(sx, sz)}
        cache = {(sx, sz): s(sx, sz)}; cells = 0
        while q and cells < 1_000_000:
            x, z = q.popleft(); cells += 1
            for dx, dz in [(4,0),(-4,0),(0,4),(0,-4)]:
                nx, nz = x+dx, z+dz
                if (nx, nz) in seen: continue
                seen.add((nx, nz)); cache[(nx,nz)] = s(nx, nz)
                if cache[(nx,nz)] < -1.05: q.append((nx, nz))
        return cells * 16


if __name__ == '__main__':
    import sys
    S = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    scanner = GPUSparseScanner(target_area=S, batch_size=256)
    seeds = np.random.randint(0, 2**63, size=256, dtype=np.uint64)
    t0 = time.perf_counter()
    hits = scanner.scan(seeds)
    dt = time.perf_counter() - t0
    print(f"Scanned {len(seeds)} in {dt:.2f}s ({len(seeds)/dt:.0f}/s), "
          f"{len(hits)} hits ({100*len(hits)/max(1,len(seeds)):.1f}%)")
    for seed, x, z in hits[:5]:
        area = scanner.flood_fill(seed, x, z)
        print(f"  seed {seed}: {area:,} blocks^2 at ({x}, {z})")

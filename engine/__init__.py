"""
engine/__init__.py — Python ctypes bindings for the C continentalness engine.

Usage:
    from engine import ContEngine
    e = ContEngine(seed)
    c = e.sample(x, z)
    grid = e.sample_grid(x0, z0, cols, rows, step)
"""

import ctypes
import os
from ctypes import c_double, c_float, c_int, c_int64, c_uint64, c_void_p, POINTER

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'continentalness.dll')
_lib = ctypes.CDLL(_lib_path)

# ContEngine is opaque — allocate enough bytes
# 24 perlin instances: 24*(257 + 8*5 + 1 + 8*2) ≈ 24*320 ≈ 7680 bytes
# Plus control fields: ~200 bytes
_CONT_ENGINE_SIZE = 8192

_lib.cont_engine_init.argtypes = [c_void_p, c_uint64, c_int]
_lib.cont_engine_init.restype = None

_lib.cont_sample.argtypes = [c_void_p, c_int, c_int]
_lib.cont_sample.restype = c_double

_lib.cont_sample_grid.argtypes = [c_void_p, POINTER(c_float),
                                   c_int, c_int, c_int, c_int, c_int]
_lib.cont_sample_grid.restype = None

_lib.cont_batch_init.argtypes = [
    POINTER(c_uint64), c_int, c_int,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
]
_lib.cont_batch_init.restype = None

_lib.cont_flood_fill.argtypes = [c_uint64, c_int, c_int, c_int]
_lib.cont_flood_fill.restype = c_int64

_lib.cont_flood_fill_6oct.argtypes = [c_uint64, c_int, c_int, c_int]
_lib.cont_flood_fill_6oct.restype = c_int64

MAX_OCTAVES = 24
PERM_SIZE = 256     # 32 lanes × 8 bytes, 8-byte aligned


class ContEngine:
    """C-accelerated continentalness sampler for one world seed."""

    def __init__(self, seed: int, large_biomes: bool = False):
        self._buf = (ctypes.c_ubyte * _CONT_ENGINE_SIZE)()
        seed_u64 = seed & 0xFFFFFFFFFFFFFFFF
        _lib.cont_engine_init(self._buf, seed_u64, 1 if large_biomes else 0)

    def sample(self, x: int, z: int) -> float:
        """Continentalness at block (x, z). Mushroom: -1.2 < val < -1.05"""
        return _lib.cont_sample(self._buf, x, z)

    def sample_grid(self, x0: int, z0: int, cols: int, rows: int,
                    step: int = 4) -> 'np.ndarray':
        """Sample a grid. Returns numpy float32 array (rows × cols)."""
        import numpy as np
        out = np.empty((rows, cols), dtype=np.float32)
        _lib.cont_sample_grid(self._buf,
                              out.ctypes.data_as(POINTER(c_float)),
                              x0, z0, cols, rows, step)
        return out

    def find_mushroom_bounds(self, x0: int, z0: int,
                             radius: int, step: int = 64) -> list:
        """Quick coarse scan for mushroom-range points. Returns list of (x,z)."""
        import numpy as np
        cols = rows = (radius * 2) // step + 1
        grid = self.sample_grid(x0 - radius, z0 - radius, cols, rows, step)
        ys, xs = np.where((grid > -1.2) & (grid < -1.05))
        return [(x0 - radius + int(x) * step, z0 - radius + int(y) * step)
                for y, x in zip(ys, xs)]


def batch_init(seeds: 'np.ndarray', large_biomes: bool = False) -> dict:
    """
    Init N seeds in one C call. Returns dict of numpy arrays ready for GPU upload.
    """
    import numpy as np
    n = len(seeds)
    seeds_u64 = np.asarray(seeds, dtype=np.uint64)

    perms  = np.zeros((n, MAX_OCTAVES, PERM_SIZE), dtype=np.uint8)
    oas    = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    obs    = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    ocs    = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    amps   = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    lacs   = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    h2s    = np.zeros((n, MAX_OCTAVES), dtype=np.uint8)
    d2s    = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    t2s    = np.zeros((n, MAX_OCTAVES), dtype=np.float32)
    ranges = np.zeros((n, 8), dtype=np.int32)
    dbls   = np.zeros((n, 2), dtype=np.float32)

    _lib.cont_batch_init(
        seeds_u64.ctypes.data_as(POINTER(c_uint64)), n, 1 if large_biomes else 0,
        perms.ctypes.data_as(c_void_p),
        oas.ctypes.data_as(c_void_p), obs.ctypes.data_as(c_void_p),
        ocs.ctypes.data_as(c_void_p),
        amps.ctypes.data_as(c_void_p), lacs.ctypes.data_as(c_void_p),
        h2s.ctypes.data_as(c_void_p), d2s.ctypes.data_as(c_void_p),
        t2s.ctypes.data_as(c_void_p), ranges.ctypes.data_as(c_void_p),
        dbls.ctypes.data_as(c_void_p))

    return {'perm': perms, 'oa': oas, 'ob': obs, 'oc': ocs, 'amp': amps, 'lac': lacs,
            'h2': h2s, 'd2': d2s, 't2': t2s, 'rng': ranges, 'dbl': dbls}

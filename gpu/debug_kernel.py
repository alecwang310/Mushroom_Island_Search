"""
debug_kernel.py — Step-by-step debug of GPU kernel via ctypes.
Tests each CUDA operation individually to isolate the crash.
"""

import ctypes, os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import ContEngine

# Load CUDA runtime
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll'
cuda = ctypes.CDLL(cuda_path)
cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda.cudaMalloc.restype = ctypes.c_int
cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cuda.cudaMemcpy.restype = ctypes.c_int
cuda.cudaFree.argtypes = [ctypes.c_void_p]
cuda.cudaFree.restype = ctypes.c_int
cuda.cudaDeviceSynchronize.restype = ctypes.c_int
cuda.cudaGetLastError.restype = ctypes.c_int
cuda.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cuda.cudaMemset.restype = ctypes.c_int

H2D, D2H = 1, 2

def check(rc, msg=""):
    if rc != 0:
        raise RuntimeError(f"CUDA error {rc}: {msg}")

def malloc(arr):
    p = ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(p), arr.nbytes), f"malloc {arr.nbytes} bytes")
    check(cuda.cudaMemcpy(p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, H2D), "memcpy H2D")
    return p

def malloc_zero(size):
    p = ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(p), size), f"malloc_zero {size} bytes")
    check(cuda.cudaMemset(p, 0, size), "memset")
    return p

def to_host(d_p, arr):
    check(cuda.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), d_p, arr.nbytes, D2H), "memcpy D2H")

# Load kernel DLL
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparse_kernel.dll')
lib = ctypes.CDLL(dll_path)
lib.gpu_scan_seeds.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
lib.gpu_scan_seeds.restype = ctypes.c_int

MAX_OCT = 24
PERM = 257

def extract_data(engine_buf):
    """Extract GPU-ready arrays from ContEngine buffer (matching sparse_kernel.cu layout)."""
    b = bytes(engine_buf)
    off = 0
    perm  = np.frombuffer(b, dtype=np.uint8,  count=MAX_OCT*PERM, offset=off); off += MAX_OCT*PERM
    oa    = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    ob    = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    oc    = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    amp   = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    lac   = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    h2    = np.frombuffer(b, dtype=np.uint8,  count=MAX_OCT, offset=off); off += MAX_OCT
    d2    = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    t2    = np.frombuffer(b, dtype=np.float64, count=MAX_OCT, offset=off).astype(np.float32); off += MAX_OCT*8
    rng   = np.frombuffer(b, dtype=np.int32,   count=8,       offset=off); off += 32
    dbl   = np.frombuffer(b, dtype=np.float64, count=2,        offset=off).astype(np.float32)
    return perm.reshape(MAX_OCT, PERM), oa, ob, oc, amp, lac, h2, d2, t2, rng, dbl


print("=== GPU Kernel Debug ===")

# Step 1: Init one seed via C engine
seed = 12345
print(f"\n1. Init seed {seed} via C engine...")
e = ContEngine(seed)
perm, oa, ob, oc, amp, lac, h2, d2, t2, rng, dbl = extract_data(e._buf)

print(f"   Perm shape: {perm.shape}")
print(f"   Ranges: {rng}")
print(f"   DBL amps: {dbl}")
print(f"   oa[0]={oa[0]:.4f} oc[0]={oc[0]:.4f} amp[0]={amp[0]:.6f} lac[0]={lac[0]:.6f}")
print(f"   h2[0]={h2[0]} d2[0]={d2[0]:.4f} t2[0]={t2[0]:.4f}")
print(f"   perm[0][:10] = {perm[0,:10]}")

# Step 2: Upload to GPU
print(f"\n2. Uploading to GPU...")
n_seeds = 1

# Flatten for GPU: (1, MAX_OCT, PERM) still works since strides are correct
d_perm   = malloc(perm.ravel())
d_oa     = malloc(oa)
d_oc     = malloc(oc)
d_amp    = malloc(amp)
d_lac    = malloc(lac)
d_h2     = malloc(h2)
d_d2     = malloc(d2)
d_t2     = malloc(t2)
d_ranges = malloc(rng)
d_dbl    = malloc(dbl)
print(f"   All GPU allocations successful")

# Step 3: Grid parameters
G, K, step = 16, 4, 80
num_grids = 8
grid_offsets = np.array([0,0, 1280,0, 0,1280, 1280,1280,
                         2560,0, 0,2560, 2560,2560, 3840,0], dtype=np.int32)
d_offsets = malloc(grid_offsets)
print(f"   Grid: G={G} K={K} step={step} grids={num_grids}")

# Step 4: Output buffers
hit = np.zeros(1, dtype=np.int32)
hx  = np.zeros(1, dtype=np.int32)
hz  = np.zeros(1, dtype=np.int32)
hg  = np.zeros(1, dtype=np.int32)
d_hit = malloc_zero(4)
d_hx  = malloc_zero(4)
d_hz  = malloc_zero(4)
d_hg  = malloc_zero(4)
print(f"   Output buffers allocated and zeroed")

# Step 5: Launch kernel
print(f"\n3. Launching kernel (1 block, 256 threads)...")
rc = lib.gpu_scan_seeds(
    d_perm, d_oa, d_oc, d_amp, d_lac, d_h2, d_d2, d_t2,
    d_ranges, d_dbl, n_seeds,
    d_offsets, num_grids, G, step, K,
    d_hit, d_hx, d_hz, d_hg)
print(f"   gpu_scan_seeds returned: {rc}")

err = cuda.cudaDeviceSynchronize()
print(f"   cudaDeviceSynchronize: {err}")

last = cuda.cudaGetLastError()
print(f"   cudaGetLastError: {last}")

# Step 6: Read results
print(f"\n4. Reading results...")
to_host(d_hit, hit)
to_host(d_hx, hx)
to_host(d_hz, hz)
print(f"   hit={hit[0]} x={hx[0]} z={hz[0]}")

# Step 7: Compare with CPU
print(f"\n5. CPU verification...")
cpu_cont = e.sample(0, 0)
print(f"   CPU cont_sample(0,0) = {cpu_cont:.6f}")

# If hit, verify at hit location
if hit[0]:
    print(f"   Mushroom candidate at ({hx[0]}, {hz[0]})")
    cpu_hit = e.sample(int(hx[0]), int(hz[0]))
    print(f"   CPU cont at hit = {cpu_hit:.6f}")
    print(f"   In mushroom range: {cpu_hit < -1.05}")

# Cleanup
for p in [d_perm, d_oa, d_oc, d_amp, d_lac, d_h2, d_d2, d_t2,
          d_ranges, d_dbl, d_offsets, d_hit, d_hx, d_hz, d_hg]:
    cuda.cudaFree(p)

print(f"\nDone.")

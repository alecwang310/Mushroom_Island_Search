"""
Build analytic variance LUT for single-octave Perlin noise at y=0.

Theory:
  The variance of a single octave's noise at y=0 depends ONLY on dy = frac(ob).
  oa, oc just translate → stationarity. lac stretches but doesn't change variance.
  Cross terms vanish (mean gradient = 0), and S = diag(0.625, 0.75, 0.625).

So: E[noise²] = Σ_i w_i² · (Sxx·δx² + Syy·δy² + Szz·δz²)
    LUT[dy] = ∫₀¹∫₀¹ E[noise²] d(dx)d(dz)
    octave_var = amp² × LUT[dy]

Usage: the LUT is precomputed once, then seed scoring is instant.
"""
import json, random, struct, ctypes, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as _eng

# ── Gradient table ───────────────────────────────────────────────────────
GRAD = np.array([
    [ 1, 1, 0], [-1, 1, 0], [ 1,-1, 0], [-1,-1, 0],
    [ 1, 0, 1], [-1, 0, 1], [ 1, 0,-1], [-1, 0,-1],
    [ 0, 1, 1], [ 0,-1, 1], [ 0, 1,-1], [ 0,-1,-1],
    [ 1, 1, 0], [ 0,-1, 1], [-1, 1, 0], [ 0,-1,-1],
], dtype=np.float64)

S = (GRAD.T @ GRAD) / 16.0
Sxx, Syy, Szz = float(S[0,0]), float(S[1,1]), float(S[2,2])

print(f"Gradient S = diag({Sxx:.4f}, {Syy:.4f}, {Szz:.4f})")
print(f"  → y-axis has higher 2nd moment ({Syy}) → dy controls y-energy in 2D slice")
print()


def smoothstep(t):
    return t*t*t * (t*(t*6.0 - 15.0) + 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Build analytic LUT
# ═══════════════════════════════════════════════════════════════════════════
def build_lut(LUT_SIZE=256, n_quad=200):
    dy_vals = np.linspace(0.0, 1.0, LUT_SIZE)
    lut = np.zeros(LUT_SIZE)
    dx = dz = (np.arange(n_quad) + 0.5) / n_quad
    DX, DZ = np.meshgrid(dx, dz)
    TX = smoothstep(DX)
    TZ = smoothstep(DZ)

    for k, dy in enumerate(dy_vals):
        t2 = smoothstep(dy)
        w000 = (1-TX)*(1-t2)*(1-TZ); w001 = (1-TX)*(1-t2)*TZ
        w010 = (1-TX)*t2*(1-TZ);     w011 = (1-TX)*t2*TZ
        w100 = TX*(1-t2)*(1-TZ);     w101 = TX*(1-t2)*TZ
        w110 = TX*t2*(1-TZ);         w111 = TX*t2*TZ

        total = (
            w000**2*(Sxx*(DX)**2   +Syy*(dy)**2   +Szz*(DZ)**2) +
            w001**2*(Sxx*(DX)**2   +Syy*(dy)**2   +Szz*(DZ-1)**2) +
            w010**2*(Sxx*(DX)**2   +Syy*(dy-1)**2 +Szz*(DZ)**2) +
            w011**2*(Sxx*(DX)**2   +Syy*(dy-1)**2 +Szz*(DZ-1)**2) +
            w100**2*(Sxx*(DX-1)**2 +Syy*(dy)**2   +Szz*(DZ)**2) +
            w101**2*(Sxx*(DX-1)**2 +Syy*(dy)**2   +Szz*(DZ-1)**2) +
            w110**2*(Sxx*(DX-1)**2 +Syy*(dy-1)**2 +Szz*(DZ)**2) +
            w111**2*(Sxx*(DX-1)**2 +Syy*(dy-1)**2 +Szz*(DZ-1)**2)
        )
        lut[k] = np.mean(total)
    return dy_vals, lut


print("Building analytic LUT...")
dy_vals, lut = build_lut()
print(f"  {len(lut)} entries: [{lut.min():.6f}, {lut.max():.6f}]")
print(f"  Max/Min = {lut.max()/lut.min():.4f}, mean = {lut.mean():.6f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Buffer layout (ContEngine struct)
# ═══════════════════════════════════════════════════════════════════════════
OFF_OA  = 24 * 257          # 6168
OFF_OB  = OFF_OA + 24 * 8   # 6360
OFF_OC  = OFF_OA + 24 * 16  # 6552
OFF_AMP = OFF_OA + 24 * 24  # 6744  (24*8*3 = 576, 6168+576=6744)
OFF_LAC = OFF_AMP + 24 * 8  # 6936
OFF_H2  = OFF_OA + 24 * 40  # 7128  (24*8*5 = 960, 6168+960=7128)
OFF_D2  = OFF_H2 + 24       # 7152
OFF_T2  = OFF_D2 + 24 * 8   # 7344


def extract(buf, oi):
    perm = np.array([buf[oi*257 + i] for i in range(257)], dtype=np.int32)
    oa = struct.unpack_from('d', buf, OFF_OA  + oi*8)[0]
    ob = struct.unpack_from('d', buf, OFF_OB  + oi*8)[0]
    oc = struct.unpack_from('d', buf, OFF_OC  + oi*8)[0]
    amp = struct.unpack_from('d', buf, OFF_AMP + oi*8)[0]
    lac = struct.unpack_from('d', buf, OFF_LAC + oi*8)[0]
    h2  = buf[OFF_H2 + oi]
    d2  = struct.unpack_from('d', buf, OFF_D2 + oi*8)[0]
    t2  = struct.unpack_from('d', buf, OFF_T2 + oi*8)[0]
    return perm, oa, ob, oc, amp, lac, h2, d2, t2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Validate LUT vs C engine (single octave) for a few dy values
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Validation: LUT vs C engine for single octave (O6 isolated)")
print("=" * 70)

seed_test = -3466959581962766165
buf = (ctypes.c_ubyte * 8192)()
_eng._lib.cont_engine_init(buf, seed_test & 0xFFFFFFFFFFFFFFFF, 0)

# Save all amplitudes, then zero all except O6
amps_orig = []
for i in range(24):
    amps_orig.append(struct.unpack_from('d', buf, OFF_AMP + i*8)[0])
    struct.pack_into('d', buf, OFF_AMP + i*8, 0.0)
# Restore O6 amplitude
struct.pack_into('d', buf, OFF_AMP + 6*8, amps_orig[6])

# Get O6 params
perm6, oa6, ob6, oc6, amp6, lac6, h26, d26, t26 = extract(buf, 6)
print(f"  O6: amp={amp6:.4f} lac={lac6:.8f} ob={ob6:.4f} (dy={ob6-np.floor(ob6):.6f})")

# Vary only the y-offset (ob) and measure empirical variance with C engine
print("  Varying ob, measuring var(cont_sample_grid) with only O6 active...")
for dy_test in [0.0, 0.25, 0.5, 0.75]:
    # Modify ob in buffer to set dy
    new_ob = 100.0 + dy_test  # floor=100, frac=dy_test
    new_h2 = 100 & 0xFF
    new_d2 = dy_test
    new_t2 = smoothstep(dy_test)
    struct.pack_into('d', buf, OFF_OB + 6*8, new_ob)
    buf[OFF_H2 + 6] = new_h2
    struct.pack_into('d', buf, OFF_D2 + 6*8, new_d2)
    struct.pack_into('d', buf, OFF_T2 + 6*8, new_t2)

    # Sample via C engine (only O6 active)
    grid = np.empty((64, 64), dtype=np.float32)
    _eng._lib.cont_sample_grid(buf, grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                -32000, -32000, 64, 64, 1000)
    emp_var = float(np.var(grid.astype(np.float64)))

    lut_idx = int(dy_test * (len(lut) - 1))
    pred_var = amp6**2 * lut[lut_idx]
    print(f"    dy={dy_test:.2f}: LUT={pred_var:.6f}  C_engine={emp_var:.6f}  ratio={emp_var/pred_var:.4f}")

# Restore original amplitudes
for i in range(24):
    struct.pack_into('d', buf, OFF_AMP + i*8, amps_orig[i])

print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Discrimination: island vs random seeds (LUT only — instant)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Discrimination: LUT-predicted O6+O15 var (island vs random)")
print("=" * 70)

with open('islands_3m.jsonl') as f:
    islands = [json.loads(line) for line in f]
island_seeds = [e['seed'] for e in islands]
random.seed(42)
random_seeds = [random.getrandbits(64) for _ in range(5000)]

N = 2000
all_seeds = island_seeds[:N] + random_seeds[:N]

lut_scores = []
for s in all_seeds:
    b = (ctypes.c_ubyte * 8192)()
    _eng._lib.cont_engine_init(b, s & 0xFFFFFFFFFFFFFFFF, 0)

    score = 0.0
    for oi in [6, 15]:
        _, _, ob_s, _, amp_s, _, _, _, _ = extract(b, oi)
        dy_s = ob_s - np.floor(ob_s)
        li = min(int(dy_s * (len(lut) - 1)), len(lut) - 1)
        score += amp_s**2 * lut[li]
    lut_scores.append(score)

lut_scores = np.array(lut_scores)

isl = lut_scores[:N]
rnd = lut_scores[N:]
mi, mr = np.mean(isl), np.mean(rnd)
pooled = np.sqrt((np.std(isl)**2 + np.std(rnd)**2) / 2)
d = (mi - mr) / (pooled + 1e-15)
print(f"  LUT O6+O15 var: isl={mi:.6f} rnd={mr:.6f} d={d:+.4f}")
print(f"  Island min/max: {isl.min():.6f} / {isl.max():.6f}")
print(f"  Random min/max: {rnd.min():.6f} / {rnd.max():.6f}")

# Also check: does var correlate with island area?
areas = np.array([e['area'] for e in islands[:N]])
r_area = np.corrcoef(isl, areas)[0, 1]
print(f"  Correlation(var, area) among islands: r={r_area:+.4f}")

# Filtering efficiency: how many randoms can we reject?
print()
print("  --- Pre-filter analysis ---")
for pct in [50, 75, 90, 95, 99]:
    thr = np.percentile(rnd, pct)
    isl_pass = np.mean(isl >= thr) * 100
    rnd_pass = np.mean(rnd >= thr) * 100
    print(f"  threshold=rnd_p{pct}={thr:.6f}: retain {isl_pass:.1f}% islands, {rnd_pass:.1f}% randoms")

print()

# ═══════════════════════════════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LUT shape (best → worst dy for variance/crunchiness)")
print("=" * 70)
for rank, i in enumerate(np.argsort(lut)[::-1][:8]):
    dy = i / 255.0
    print(f"  #{rank+1}: dy={dy:.4f}  LUT={lut[i]:.6f}")
print("  ...")
for rank, i in enumerate(np.argsort(lut)[:4]):
    dy = i / 255.0
    print(f"  #{len(lut)-rank}: dy={dy:.4f}  LUT={lut[i]:.6f}")

# Save
np.savez('variance_lut.npz', dy=dy_vals, lut=lut)
print(f"\nSaved variance_lut.npz")

print("\nDone.")

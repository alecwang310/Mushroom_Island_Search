# Mushroom Island Search — GPU-Accelerated Seed Scanner

## Objective

Find Minecraft seeds with mushroom islands ≥ 4 million blocks², **speed over accuracy**. We prioritize scanning more seeds faster over catching every island.

## Pipeline (current: hex grid O6+O15)

```
Seed range → [optional: prefilter_kernel GPU — LUT variance filter]
     → compact survivor seeds in RAM (no 800MB host seed array or disk round trip)
     → cont_batch_init_tiered (CPU, O6+O15 only) → one compact upload
     → tiered_scan GPU kernel (2 octaves, hex grid, ~9M cycles/seed)
     → download only compacted hit triples plus geometry codes
     → CPU verify (≥2 unique 1x neighbors around each triple)
     → CPU 6-octave flood (C, six octaves initialized) → if area≥3M, 24-octave flood
     → log results ≥4M to islands_4m.jsonl
```

The previous baseline was ~20,000 tiered seeds/s at G=512 and step_2x=280.
The compact initialization/transfer path must be benchmarked on the CUDA host after rebuilding.

## File Structure

```
engine/                        # CPU continentalness engine (C + Python bindings)
  continentalness.h            #   24-octave Perlin noise, MC 1.18+ exact
  continentalness.c            #   sampling, compact tier init, flood fills
  __init__.py                  #   Python ctypes bindings (ContEngine class)

gpu/
  tiered_kernel.cu             # ★ Current main kernel: hex grid, 2 octaves only
  prefilter_kernel.cu          #    GPU seed-range variance LUT pre-filter
  variance_lut.h               #    Precomputed analytic LUT + MD5/amplitude constants
  hunt_engine.cu               #    DLL host: compact init/upload/launch/download
  hunt_tiered.py               # ★ Current hunt script: GPU thread + CPU verify + flood
  gpu_monitor.py               #    nvidia-smi polling script

continentalness_pipeline.py    # Pure-Python reference implementation (verified vs cubiomes)
build_variance_lut.py          # Builds analytic variance LUT + pre-filter threshold calibration
variance_lut.npz               # Saved LUT (256 floats)
islands_4m.jsonl               # Output: seed, area, center coordinates
```

## Pre-Filter (prefilter_kernel.cu)

- **Purpose**: Reject unpromising seeds BEFORE the expensive hex grid GPU scan
- **Mechanism**: GPU-side Xoroshiro128++ RNG → extract `ob₆`, `ob₁₅` (y-offsets)
  → LUT lookup → `score = amp²·(LUT[dy₆] + LUT[dy₁₅])`
- **Theory**: Single-octave variance depends only on `dy = frac(ob)`.
  Gradient table has `Syy=0.75 > Sxx,Szz=0.625`, so dy controls y-energy
  coupling into the 2D slice. Island seeds have dy tightly clustered near 0.5.
- **LUT**: `gpu/variance_lut.h` — 256 floats, analytic integral over (dx,dz)∈[0,1]²
- **Work**: 10 Xoroshiro draws/seed, one thread/seed
- **Input/output**: seed is generated as `start_seed + tid`; only survivor seeds are copied back
- **Thresholds** (from `build_variance_lut.py`):
  | pct | threshold | islands kept | randoms rejected |
  |-----|-----------|-------------|-----------------|
  | 90  | 0.0415    | ~91%        | 90%             |
  | 95  | 0.0424    | ~69%        | 95%             |
  | 99  | 0.0432    | ~23%        | 99%             |
  | 995 | 0.0434    | ~12%        | 99.5%           |
  | 999 | 0.0435    | ~2.3%       | 99.9%           |

## GPU Kernel (tiered_kernel.cu)

- **Grid**: Hex lattice (staggered rows), D=280-block spacing, 6 neighbors
- **Octaves**: Only O6 + O15 (cont A/B first octaves). No shift distortion.
  - O6: amp=0.501, wavelength=2000 blocks. O15: amp=0.501, 1.8% detuned.
- **Threshold**: O6+O15 < -1.00 (lenient, catches island cores)
- **Threads**: 256 per block; launch bounds default to 4 blocks/SM
- **Tiling**: 32×32 cells per tile, processed as four converged one-cell waves
- **Perm storage**: each warp distributes each 256-byte table across 32 lanes
  (two packed uint32 registers per lane per octave)
- **Perm lookup**: seven adjacent-pair stages per Perlin call; each pair uses three
  `shfl.sync.idx.b32` instructions plus `prmt.b32`, with no shared-memory lookup
- **Gradient lookup**: branchless hash arithmetic replaces divergent constant-table reads
- **Detection**: 32 shared row bitmasks replace the float grid; a hit requires the center plus at least two true hex neighbors. The two selected neighbor bits and row parity are returned as a geometry code, so line, triangle, and V-shaped triples are represented without rescanning the GPU grid.
- **Fallback**: compile with `-DTIERED_USE_WARP_PERM=0` for the shared-memory reference
- **Initialization**: CPU creates only the exact O6/O15 state instead of all 24 octaves.

## CPU Pipeline (hunt_tiered.py)

- **16 workers** (ThreadPoolExecutor)
- **GPU thread**: continuously submits batches, non-blocking verify+flood
- **Verify** (`verify_triple_cpu`): a cached lookup expands the center and the two selected 2x neighbors into unique adjacent 1x-grid points. Verification passes when at least 2 sampled points are below -1.0; all continentalness octaves are used without shift.
- **Tier 1 flood** (`cont_flood_fill_6oct`): 6 essential octaves, C BFS, 37ms (3.6× faster than full)
- **Tier 2 flood** (`cont_flood_fill`): full 24-octave, only if tier 1 ≥ 3M
- **Dedup**: by (seed, area) to avoid logging the same island from overlapping triples

## Key Findings (from analysis)

1. **Octaves 6+15 contribute 80%** of island signal at center. Always negative for islands.
2. **Shift octaves (0-5) contribute 0%** to island area — purely cosmetic.
3. **O6+O15 beat period: 109K blocks** — islands repeat every ~55K blocks.
4. **Previous GPU bottleneck**: 14 serial shared-memory loads in the hash chain
   produced 43-54% bank conflicts. The default kernel now uses warp-register
   permutation storage; it still needs Nsight benchmarking on the CUDA host.
5. **CPU bottleneck**: flood fill (37-134ms). The triple filter and ≥2-of-all-neighbors verification threshold are intentionally stricter to reduce this workload while preserving clustered islands.
6. The remaining serial dependency is the seven-stage paired permutation hash chain.

## Build

```powershell
# GPU DLL
cd gpu
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o hunt_engine.dll hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu ../engine/continentalness.c -I../engine -lcudart

# Engine DLL (if continentalness.c changes)
cd engine
gcc -O3 -shared -o continentalness.dll continentalness.c -lm
```

For the shared-memory A/B baseline, add `-DTIERED_USE_WARP_PERM=0`. If ptxas
reports spills in the warp path, benchmark `-DTIERED_MIN_BLOCKS_PER_SM=3` rather
than forcing `--maxrregcount`; the launch-bounds setting is intentionally exposed.

## Run

```powershell
cd D:\Code\Seeds
python gpu\hunt_tiered.py
```

Logs islands ≥ 4M to `islands_4m.jsonl`. Ctrl+C to stop.

Pre-filter mode is controlled by `PREFT_ENABLED`, `PREFT_LO`, and `PREFT_HI`
in `gpu/hunt_tiered.py`.

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
     → CPU 0.5x connected estimate → only probable ≥4M hits reach flood fill
     → CPU 6-octave flood (C, six octaves initialized) → if area≥3M, 24-octave flood
     → log results ≥4M to islands_4m.jsonl
```

At `G=512`, `step_2x=300`, and threshold `-0.95`, the current full-tile prefix
LUT measures `263.9-265.6K survivors/s` in warmed pure-kernel runs and
`225.3-226.8K/s` through the complete DLL stages. The no-prefix control measures
`240.7-241.1K/s`, so the kernel gain is approximately `10%`. An exact
consecutive-seed comparison returned the same `2,274` sorted hit records.
Disable the prefix with `-DTIERED_USE_PREFIX_LUT=0`. A 24-worker CPU estimate
sample processed `55.2K hits/s`; no sampled estimate reached 4M, so positive
flood-fill throughput still requires a separate benchmark.

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
  benchmark_historical_hits.py #    Historical triple-filter retention benchmark
  benchmark_step_sizes.py      #    GPU spacing/retention/selectivity benchmark
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

- **Grid**: Hex lattice (staggered rows), D=300-block spacing, 6 neighbors
- **Octaves**: Only O6 + O15 (cont A/B first octaves). No shift distortion.
  - O6: amp=0.501, wavelength=2000 blocks. O15: amp=0.501, 1.8% detuned.
- **Threshold**: O6+O15 < -0.95 (lenient, catches island cores)
- **Threads**: 256 per block; launch bounds default to 3 blocks/SM
- **Tiling**: 32×32 cells per tile, processed as four converged one-cell waves
- **Perm storage**: the default keeps each 256-entry table in packed shared
  memory, storing `P[index] | (P[index + 1] << 8)` in one uint32 per index
- **Perm lookup**: seven adjacent-pair stages per Perlin call; the default uses
  one shared load per pair, while the warp-register control uses three
  `shfl.sync.idx.b32` instructions plus `prmt.b32`
- **Prefix LUT**: full tiles precompute the first three x-only pair stages once
  per lane and reuse two packed second-level pairs across four waves; partial
  edge tiles use the original path
- **Gradient lookup**: branchless hash arithmetic replaces divergent constant-table reads
- **Detection**: 32 shared row bitmasks replace the float grid; a hit requires the center plus at least two true hex neighbors. The two selected neighbor bits and row parity are returned as a geometry code, so line, triangle, and V-shaped triples are represented without rescanning the GPU grid.
- **A/B control**: compile with `-DTIERED_USE_WARP_PERM=1` for the warp-register
  path, `-DTIERED_USE_PREFIX_LUT=0` for the no-prefix shared path, or add
  `-DTIERED_SHARED_PACKED_PAIRS=0` for the original byte-table path
- **Initialization**: CPU creates only the exact O6/O15 state instead of all 24 octaves.

## CPU Pipeline (hunt_tiered.py)

- **24 workers** (ThreadPoolExecutor)
- **Bounded queue**: 2,048 verification tasks by default. The old 48-task queue
  forced the GPU thread to wait after every result chunk and prevented CPU work
  from overlapping the next CUDA launch.
- **GPU thread**: continuously submits batches, non-blocking verify+flood
- **Estimate** (`estimate_triple_area`): a cached 0.5x hex lookup samples the connected low-cell component touching the three GPU points. It uses the full shifted continentalness field and the real mushroom threshold; only connected estimates ≥4M reach flood fill.
- **Tier 1 flood** (`cont_flood_fill_6oct`): 6 essential octaves, C BFS gate before the full flood
- **Tier 2 flood** (`cont_flood_fill`): full 24-octave, only if tier 1 ≥ 3M
- **Dedup**: by (seed, area) to avoid logging the same island from overlapping triples

## Key Findings (from analysis)

1. **Octaves 6+15 contribute 80%** of island signal at center. Always negative for islands.
2. **Shift octaves (0-5) contribute 0%** to island area — purely cosmetic.
3. **O6+O15 beat period: 109K blocks** — islands repeat every ~55K blocks.
4. **GPU memory behavior**: the no-prefix baseline performs `3,670,016` pair
   loads per `G=512` seed. The full-tile prefix path performs `2,490,368`, a
   `32.1%` reduction, and measures approximately `10%` faster. The gain is
   larger than the memory-only estimate because the prefix also shortens the
   serial hash dependency chain. Nsight reports about `50.9%` shared-load
   wavefront expansion on the no-prefix capture.
5. **CPU behavior**: the estimate path is scalar FP64 Perlin work; flood fill
   adds allocation, hash-table, branch, and cache pressure. More CPU registers
   alone are unlikely to create a large speedup; higher clocks and cores help
   until the memory system or allocator saturates.
6. The selected three-block prefix build uses 80 registers/thread, 2,180 B
   shared memory, and a 4-byte spill load/store. The spill-free two-block build
   is slower because occupancy falls; four blocks are also slower despite using
   64 registers.
7. Keep `G` divisible by 32. Partial edge tiles execute a full 32x32 tile and use
   the slower validity/atomic path; for example, `G=449` measured ~181K/s while
   `G=448` measured ~207K/s.

## Build

```powershell
# GPU DLL
cd gpu
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o hunt_engine.dll hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu ../engine/continentalness.c -I../engine -lcudart

# Engine DLL (if continentalness.c changes)
cd engine
gcc -O3 -shared -o continentalness.dll continentalness.c -lm
```

The default build uses packed shared permutation pairs. For the warp-register
A/B control, add `-DTIERED_USE_WARP_PERM=1`; for the no-prefix control, add
`-DTIERED_USE_PREFIX_LUT=0`. The default three-block prefix build intentionally
keeps a 4-byte spill because the spill-free two-block build is slower.

## Run

```powershell
cd D:\Code\Seeds
python gpu\hunt_tiered.py
```

Logs islands ≥ 4M to `islands_4m.jsonl`. Ctrl+C to stop.

Pre-filter mode is controlled by `PREFT_ENABLED`, `PREFT_LO`, and `PREFT_HI`
in `gpu/hunt_tiered.py`.

## Persistent Operations Handoff

Read `PROJECT_RUNBOOK.md` before using the Windows CUDA host. It contains the
SSH/VPN procedure, safe branch synchronization, exact Visual Studio/CUDA build
commands, benchmark commands, pipeline interpretation, and historical-filter
test protocol.

Current handoff state on July 27, 2026: branch `gpt` uses ordinary packed
shared permutation pairs with the full-tile prefix LUT enabled; the 16/8-replica
transpose switches remain only for architecture-specific A/B tests. Use
`-DTIERED_USE_PREFIX_LUT=0` to reproduce the no-prefix control.
`main` / `origin/main` remains at
`b7fcd38` (`refine prefilter`); the historical `islands_3m.jsonl` file remains
available at `2b1c2f2:islands_3m.jsonl` and must stay out of `gpt` commits.
Only final full-flood results >=4M are valid evidence; estimator passes are
screening signals, not confirmed islands.

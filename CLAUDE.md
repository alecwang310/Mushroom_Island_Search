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

The July 27, 2026 RTX 5080 baseline at `G=512`, `step_2x=300`, and threshold
`-0.95` is ~157K survivors/s in the pure kernel and ~142K survivors/s including
compact CPU initialization and transfers. With a 2,048-task verification queue,
the live GPU/CPU pipeline sustains ~133K survivors/s on the deterministic test.

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

- **24 workers** (ThreadPoolExecutor)
- **Bounded queue**: 2,048 verification tasks by default. The old 48-task queue
  forced the GPU thread to wait after every result chunk and prevented CPU work
  from overlapping the next CUDA launch.
- **GPU thread**: continuously submits batches, non-blocking verify+flood
- **Estimate** (`estimate_triple_area`): a cached 0.5x hex lookup samples the connected low-cell component touching the three GPU points. It uses the full shifted continentalness field and the real mushroom threshold; only connected estimates ≥4M reach flood fill.
- **Tier 1 flood** (`cont_flood_fill_6oct`): 6 essential octaves, C BFS, 37ms (3.6× faster than full)
- **Tier 2 flood** (`cont_flood_fill`): full 24-octave, only if tier 1 ≥ 3M
- **Dedup**: by (seed, area) to avoid logging the same island from overlapping triples

## Key Findings (from analysis)

1. **Octaves 6+15 contribute 80%** of island signal at center. Always negative for islands.
2. **Shift octaves (0-5) contribute 0%** to island area — purely cosmetic.
3. **O6+O15 beat period: 109K blocks** — islands repeat every ~55K blocks.
4. **Previous GPU bottleneck**: 14 serial shared-memory loads in the hash chain
   produced 43-54% bank conflicts. The warp-register path reduces the measured
   conflicts to 30 total per profiled launch, with no shared atomic conflicts.
5. **CPU bottleneck**: flood fill (37-134ms). The triple filter and ≥2-of-all-neighbors verification threshold are intentionally stricter to reduce this workload while preserving clustered islands.
6. Nsight Compute measures 86.3% SM issue utilization, 64 registers/thread,
   65.8% achieved occupancy, 0.21% DRAM throughput, and zero spills. The remaining
   kernel bottleneck is math/shuffle execution in `perm_pair_warp` and `grad_dot`,
   especially the seven-stage paired permutation hash chain.
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

## Persistent Operations Handoff

Read `PROJECT_RUNBOOK.md` before using the Windows CUDA host. It contains the
SSH/VPN procedure, safe branch synchronization, exact Visual Studio/CUDA build
commands, benchmark commands, pipeline interpretation, and historical-filter
test protocol.

Current handoff state on 2026-07-26: branch `gpt` and `origin/gpt` are at
`442b300`; `main` / `origin/main` is at `b7fcd38` (`refine prefilter`) and
deleted `islands_3m.jsonl`, which is available at `2b1c2f2:islands_3m.jsonl`.
The current 0.5x estimator is not trusted: a
4,096-hit benchmark passed 3,419 hits to six-octave flood but produced only one
actual result >=4M. Diagnose this before treating estimator passes as evidence.

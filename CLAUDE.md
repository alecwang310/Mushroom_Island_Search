# Mushroom Island Search — GPU-Accelerated Seed Scanner

## Objective

Find Minecraft seeds with mushroom islands ≥ 3 million blocks², **speed over accuracy**. We prioritize scanning more seeds faster over catching every island.

## Pipeline (current: hex grid O6+O15)

```
Seed → cont_batch_init (CPU, 300ms) → upload (PCIe, 1ms)
     → tiered_scan GPU kernel (2 octaves, hex grid, ~9M cycles/seed)
     → download pairs (~500 per batch)
     → CPU verify (≥2 hex points, all cont octaves, 22µs/hit)
     → CPU 6-octave flood (C, 37ms) → if area≥3M, 24-octave flood (134ms)
     → log to islands_3m.jsonl
```

**Throughput: ~20,000 seeds/s at G=512, step_2x=280. CPU keeps up with 16 workers.**

## File Structure

```
engine/                        # CPU continentalness engine (C + Python bindings)
  continentalness.h            #   24-octave Perlin noise, MC 1.18+ exact
  continentalness.c            #   cont_sample, cont_flood_fill, cont_flood_fill_6oct
  __init__.py                  #   Python ctypes bindings (ContEngine class)

gpu/
  tiered_kernel.cu             # ★ Current main kernel: hex grid, 2 octaves only
  hunt_engine.cu               #    DLL host: upload, launch, download, timing
  hunt_tiered.py               # ★ Current hunt script: GPU thread + CPU verify + flood
  sparse_kernel.cu             #    Baseline kernel (all 24 octaves, K=2, square grid)
  warp_shuffle.cu              #    Experimental register-based perm table (broken)
  bench_ws.py                  #    Compare shared-mem vs warp-shuffle variants
  gpu_monitor.py               #    nvidia-smi polling script

continentalness_pipeline.py    # Pure-Python reference implementation (verified vs cubiomes)
islands_3m.jsonl               # Output: seed, area, center_1:4 coords
```

## GPU Kernel (tiered_kernel.cu)

- **Grid**: Hex lattice (staggered rows), D=280-block spacing, 6 neighbors
- **Octaves**: Only O6 + O15 (cont A/B first octaves). No shift distortion.
  - O6: amp=0.501, wavelength=2000 blocks. O15: amp=0.501, 1.8% detuned.
- **Threshold**: O6+O15 < -1.00 (lenient, catches island cores)
- **Threads**: 256 per block, 4 blocks/SM, 64 registers/thread
- **Tiling**: 32×32 cells per tile, MAXC=4 cells/thread
- **Perm storage**: uint32_t[257] per octave (zero bank conflict attempt, marginal gain)
- **Perm load**: Both perms loaded together, single __syncthreads() per tile
- **Detection**: 6-neighbor adjacency on s_grid. All pairs written via atomicAdd.
- **Timing**: clock64() measures perlin (97%) vs detection (3%)

## CPU Pipeline (hunt_tiered.py)

- **16 workers** (ThreadPoolExecutor)
- **GPU thread**: continuously submits batches, non-blocking verify+flood
- **Verify** (`verify_pair_cpu`): O6+O15 only, 5 hex points, ≥2 must be < -1.0
- **Tier 1 flood** (`cont_flood_fill_6oct`): 6 essential octaves, C BFS, 37ms (3.6× faster than full)
- **Tier 2 flood** (`cont_flood_fill`): full 24-octave, only if tier 1 ≥ 3M
- **Dedup**: by (seed, area) to avoid logging same island from adjacent pairs

## Key Findings (from analysis)

1. **Octaves 6+15 contribute 80%** of island signal at center. Always negative for islands.
2. **Shift octaves (0-5) contribute 0%** to island area — purely cosmetic.
3. **O6+O15 beat period: 109K blocks** — islands repeat every ~55K blocks.
4. **GPU bottleneck**: 14 serial shared-memory loads in hash chain. Bank conflicts 43-54%.
   Perlin math is 0.7% of time; 99% is perm loads + shared memory stalls + sync.
5. **CPU bottleneck**: flood fill (37-134ms). ≥2 verify filter keeps pass rate low enough.
6. 2-octave kernel is at performance ceiling without algorithmic change.

## Build

```powershell
# GPU DLL
cd gpu
nvcc -O3 -arch=sm_120 -shared -o hunt_engine.dll hunt_engine.cu sparse_kernel.cu warp_shuffle.cu tiered_kernel.cu ../engine/continentalness.c -I../engine -lcudart

# Engine DLL (if continentalness.c changes)
cd engine
gcc -O3 -shared -o continentalness.dll continentalness.c -lm
```

## Run

```powershell
cd D:\Code\Seeds
python gpu\hunt_tiered.py
```

Logs islands ≥ 3M to `islands_3m.jsonl`. Ctrl+C to stop.

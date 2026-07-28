# Mushroom Island Search — Remote Operations Runbook

This is the persistent handoff for operating the project across the local Mac
workspace and the Windows CUDA machine. The Mac is the source of truth for
source edits. The Windows machine is used for CUDA compilation, execution, and
Nsight profiling.

## Current State

- Branch: `gpt`.
- Current kernel: packed shared permutation pairs with the full-tile prefix LUT
  and a three-block launch bound; see `KERNEL_INVESTIGATION.md` for measured
  A/B timings and the memory analysis.
- Original branch: `main` / `origin/main` at `b7fcd38`, `refine prefilter`.
- Historical dataset: `2b1c2f2:islands_3m.jsonl`; `b7fcd38` deleted it while refining the prefilter, so keep the materialized file out of `gpt` commits.
- `continentalness_pipeline.py` is verified and is the correctness reference.
- Current hunt defaults: `step_05x=125`, `step_2x=500`, `G=512`, GPU threshold
  `-0.95`, 24 CPU workers, and prefilter band `[0.0432, 0.0434]`.
- The GPU R=2 coarse gate is `6_000_000` estimated blocks²; only final
  full-flood areas >=4M are valid results.
- Override the device-side coarse gate with `HUNT_GPU_COARSE_MIN_AREA` for
  controlled retention experiments; clipped R=2 components are always kept.
- GPU scanning and CPU verification run concurrently. The 2,048-task bounded
  queue preserves overlap without returning to the old unbounded-Future memory
  growth; current CPU pressure samples are estimate-only unless flood-positive
  hits are deliberately supplied.

## Required Change Workflow

For every substantial change:

1. Edit source and benchmark files locally.
2. Run local static checks only; this Mac does not provide the target CUDA runtime.
3. Review the diff, commit with a precise message, and push `gpt`.
4. Fetch the exact commit on Windows and verify the hash.
5. Rebuild the affected DLLs before running a benchmark or hunt.
6. Record commit, command, sample size, timings, hit counts, and retention/error counts.

Do not edit source directly on Windows. Before using `git reset --hard` there,
run `git status --short` and preserve or discard local changes deliberately.

## SSH Connection

The VPN host is represented on the Mac by a top-level folder whose basename
starts with `192.`. Find the folder under the appropriate parent directory:

```zsh
HOST="$(find /path/to/top-level -maxdepth 1 -type d -name '192.*.*.*' -print -quit | sed 's#.*/##')"
test -n "$HOST" || { echo "VPN host folder not found"; exit 1; }
ssh -tt administrator@"$HOST"
```

The password is entered interactively and must not be stored in the repository
or placed in a command. `-tt` keeps the interactive terminal and helps when
the password prompt or Windows `cmd.exe` behaves unexpectedly.

After connecting, the remote shell is normally Windows `cmd.exe`:

```bat
cd /d D:\Code\Seeds
git status --short --branch
git rev-parse --short HEAD
```

If SSH fails, check VPN connectivity and confirm that the `192.*` folder is
the current host address. Retry with `ssh -vv administrator@HOST` for transport
diagnostics. If no password prompt appears, retry with `ssh -tt`.

## Synchronize Windows

After pushing a local commit, use this safe sequence on Windows:

```bat
cd /d D:\Code\Seeds
git status --short
git fetch origin
git switch gpt
git reset --hard origin/gpt
git rev-parse --short HEAD
```

The final hash must match the commit being tested. If the working tree is not
clean, stop before the reset and inspect the changes.

To test an old checkpoint without moving the branch permanently:

```bat
git fetch origin
git log --all --oneline --decorate -20
git switch --detach <exact-commit>
git rev-parse --short HEAD
```

Return to the current branch with `git switch gpt` and the normal sync steps.

## Windows Build

The Visual Studio environment is not initialized automatically. Before
invoking `nvcc`, run:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
where nvcc
where gcc
where python
nvcc --version
nvidia-smi
```

If `engine/continentalness.c` changed, rebuild the CPU DLL first:

```bat
cd /d D:\Code\Seeds\engine
gcc -O3 -shared -o continentalness.dll continentalness.c -lm
```

Then rebuild the GPU DLL:

```bat
cd /d D:\Code\Seeds\gpu
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o hunt_engine.dll hunt_engine.cu tiered_kernel.cu coarse_verify_kernel.cu prefilter_kernel.cu ..\engine\continentalness.c -I..\engine -lcudart
```

The default build uses packed shared permutation pairs. For the warp-register
A/B control, add `-DTIERED_USE_WARP_PERM=1`; for the original byte-table shared
control, also add `-DTIERED_SHARED_PACKED_PAIRS=0`. The prefix-LUT path is on by
default with `TIERED_MIN_BLOCKS_PER_SM=3`; add `-DTIERED_USE_PREFIX_LUT=0` for
its control build. The selected prefix build intentionally retains a 4-byte
spill because the spill-free two-block build is slower. Preserve ptxas
register/spill/local-memory output with every result instead of optimizing for
spill count alone.
`C4819` code-page warnings have occurred and are not fatal when the DLL builds;
missing DLLs, unresolved symbols, `nvcc` errors, and Python load errors are
fatal and must be fixed first.

## Benchmarks and Hunt

Run the bounded benchmark from the repository root:

```bat
cd /d D:\Code\Seeds
set HUNT_CPU_WORKERS=24
set HUNT_CPU_HIT_LIMIT=4096
python gpu\benchmark_tiered.py
```

`HUNT_CPU_HIT_LIMIT=0` samples every returned GPU hit. Use a finite limit first
when measuring CPU service time. Increase the limit for a longer CPU pressure
test while keeping `HUNT_CPU_WORKERS=24`.

The benchmark reports prefilter survivors, GPU hits and throughput, CPU wall
time, aggregate estimator time, six-octave flood time, full-flood time,
pending-queue pressure, estimator passes, and actual >=4M results.

Use this deterministic raw-stage benchmark when comparing kernel parameters:

```bat
set HUNT_START_SEED=0x123456789ABCDEF0
set HUNT_PREF_BATCH=100000000
set HUNT_SCAN_LIMIT=262144
set HUNT_SKIP_CPU=1
set HUNT_STEP_05X=125
set HUNT_STEP_2X=500
set HUNT_GRID_SIZE=512
set HUNT_THRESHOLD=-0.95
python gpu\benchmark_tiered.py
```

The compressed full-scale LUT experiment is a standalone synthetic A/B and
does not modify `tiered_kernel.cu`. It packs the eight final gradient hashes
for each `(h1,h3)` key into one `uint32` per octave, so the two tables occupy
`512 KiB` in global memory. Build and run it on Windows with:

```bat
cd /d D:\Code\Seeds\gpu
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o full_lut_benchmark.dll full_lut_benchmark.cu -lcudart
cd /d D:\Code\Seeds
python gpu\benchmark_full_lut.py --blocks 256 --grid 512 --repeats 10
```

The harness reports one-seed LUT construction time, extraction-only time,
LUT extraction plus interpolation, and a synthetic shared-permutation-chain
control. The LUT is intentionally random and global-memory resident; use the
result to estimate lookup/scheduling potential, not to validate seed output.

Use this bounded live-pipeline profile to measure GPU/CPU overlap:

```bat
set HUNT_MAX_PREFILTER_BATCHES=1
set HUNT_PROFILE_STAGES=1
set HUNT_CPU_WORKERS=24
set HUNT_MAX_PENDING=2048
python gpu\hunt_tiered.py
```

The no-prefix kernel measures `240.7-241.1K survivors/s` at `G=512`. The
selected prefix-LUT build measures `263.9-265.6K/s` after clock warm-up with a
three-block launch bound, approximately `10%` faster. An exact consecutive-seed comparison
returned the same `2,274` sorted hit records. Use
`-DTIERED_USE_PREFIX_LUT=0` for the control. The
24-worker CPU sample processed `4,096` hits at `55.2K hits/s`, with peak queue
depth `48` and no blocked submissions; no estimate crossed 4M, so this does
not measure positive flood-fill service.

Run the continuous hunt with:

```bat
cd /d D:\Code\Seeds
python gpu\hunt_tiered.py
```

Stop with `Ctrl+C`. Only final full-flood results >=4M are valid; estimator
passes are not verified islands. Valid results append to `islands_4m.jsonl`,
and the best result is written to `best_4m.jsonl`.

## Pipeline Mental Model

1. Optional GPU prefilter: a 100M consecutive-seed range is scored with the variance LUT and compacted into survivor seeds. It enriches the search but does not prove an island.
2. CPU compact initialization: O6/O15 state and six-octave verifier state are created for each survivor and uploaded once per GPU scan chunk.
3. GPU tiered scan: `tiered_kernel.cu` evaluates a 2x hex grid, with 500-block spacing and `G=512`. A hit is a center plus two connected mushroom points, encoded as line, triangle, or V geometry.
4. GPU coarse verifier: `coarse_verify_kernel.cu` evaluates the fixed R=2 1x hex neighborhood at 250-block spacing, counts connected six-octave cells, and keeps estimates >=6M or clipped components.
5. CPU estimate: the cached 0.5x lookup samples at 125-block spacing and counts only the connected low-cell component touching the triple. It no longer extrapolates a perimeter shell.
6. Tier-1 flood: six-octave C flood is a cheap gate; only areas >=3M reach the full 24-octave flood.
7. Final output: only a full 24-octave area >=4M is a valid large mushroom island.

Key units are `step_05x=125`, `step_1x=250`, `step_2x=500`, `G=512`, coarse
gate `6_000_000`, final target `4_000_000`, and the tier-1 gate `3_000_000`.
The GPU O6+O15 threshold
`-0.95` is intentionally lenient; the CPU flood remains the authority.

The runtime hunt and `benchmark_tiered.py` use 24 workers by default. The live
hunt bounds pending verification tasks at 2,048 by default; override with
`HUNT_MAX_PENDING` only for controlled queue-depth benchmarks.

## Nsight Findings

The July 27, 2026 RTX 5080 capture used CUDA 13.2, Nsight Systems 2025.6.3,
Nsight Compute 2025.4.1, and `sm_120`.

- Nsight Systems: `tiered_scan` was 99.8% of GPU kernel time. H2D averaged
  about 0.26 ms/chunk and D2H was negligible.
- Nsight Compute on the no-prefix packed shared path: 64 registers/thread,
  2,180 B shared memory, zero spills, negligible DRAM traffic, and
  approximately 50.9% shared load wavefront expansion at an average 2-way
  conflict.
- The prefix LUT reduces logical pair-load calls by 32.1%—not elapsed time or
  measured shared-memory wavefronts—and measures `263.9-265.6K survivors/s`
  after warm-up, approximately 10% above the no-prefix kernel.
  The selected three-block build uses 80 registers/thread and a 4-byte spill.
- Shared conflicts remain measurable, but the packed path is faster than the
  warp-register control. The logical traffic is about `3.53 TB/s` for packed
  permutation reads and `4.28 TB/s` including row-mask reads. A fixed-buffer
  estimate puts the practical no-conflict ceiling at roughly `246-249K
  survivors/s`, not 2× current throughput.
- The remaining work is the Perlin arithmetic chain plus random-index shared
  lookup scheduling; the tested 16- and 8-replica transposes did not improve
  throughput and remain compile-time A/B switches only.
- Global coalescing is not a practical issue: source counters found only 18
  excessive sectors in the 1,024-seed line-mapped capture.
- `G` must stay divisible by 32. Partial edge tiles still run all Perlin samples
  and use the slower validity/atomic path. Measured examples were `449 -> 448`:
  ~181K/s -> ~207K/s, and `417 -> 416`: ~208K/s -> ~240K/s.

For line-correlated Nsight Compute output, rebuild with `-lineinfo` added to the
normal `nvcc` command. Keep `.ncu-rep`, `.nsys-rep`, `.sqlite`, and exported CSV
files untracked.

## Historical Retention Test

Materialize the old dataset on Windows without changing `gpt`:

```bat
cd /d D:\Code\Seeds
git show 2b1c2f2:islands_3m.jsonl > historical_islands_3m.jsonl
```

The retention benchmark must feed those historical seeds directly to the
current GPU triple kernel; do not run the random-range prefilter for this
measurement. Report seed-level retention and coordinate-local retention,
split by actual area, especially 3M+ and 4M+. For matched hits, compare the
current estimate with historical actual area. Keep the large JSONL untracked.

Run the committed tool from the repository root:

```bat
python gpu\benchmark_historical_hits.py --input historical_islands_3m.jsonl --estimate --report historical_retention.json
```

The default coordinate-local radius is two coarse steps (`560` in the current
configuration). The report also shows exact-coordinate, one-step, and
seed-level retention so the sacrificed count is not dependent on one matching
radius.

Compare GPU spacing and random-control pruning with:

```bat
python gpu\benchmark_step_sizes.py --input historical_islands_3m.jsonl --report step_sizes.json
```

This uses historical records >=4M as known positives. Random controls are
seeds not present in the historical file, so their pruning rate measures GPU
selectivity but is not proof that every control lacks a large island.

## Local Checks and Git Handoff

Before pushing source or benchmark changes:

```zsh
python3 -m py_compile gpu/hunt_tiered.py gpu/benchmark_tiered.py engine/__init__.py
git diff --check
git status --short
gcc -fsyntax-only engine/continentalness.c -Iengine
```

Use a precise commit message, for example:

```zsh
git add CLAUDE.md PROJECT_RUNBOOK.md <changed-files>
git commit -m "Document SSH, Windows build, and tiered pipeline workflow"
git push origin gpt
```

Never add `.DS_Store`, generated DLLs, temporary benchmark output, or the
historical dataset unless explicitly required.

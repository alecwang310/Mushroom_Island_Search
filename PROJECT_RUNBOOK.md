# Mushroom Island Search — Remote Operations Runbook

This is the persistent handoff for operating the project across the local Mac
workspace and the Windows CUDA machine. The Mac is the source of truth for
source edits. The Windows machine is used for CUDA compilation, execution, and
Nsight profiling.

## Current State

- Branch: `gpt`.
- Current commit: `442b300`, `Add 0.5x perimeter area estimator before flood fill`.
- Original branch: `main` / `origin/main` at `b7fcd38`, `refine prefilter`.
- Historical dataset: `2b1c2f2:islands_3m.jsonl`; `b7fcd38` deleted it while refining the prefilter, so keep the materialized file out of `gpt` commits.
- `continentalness_pipeline.py` is verified and is the correctness reference.
- The current 0.5x estimator is not trusted: one 4,096-hit benchmark passed 3,419 hits to six-octave flood, but only one final result was actually >=4M.
- Next investigation: diagnose the estimator and measure aggressive triple-filter retention against the historical dataset.

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
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o hunt_engine.dll hunt_engine.cu tiered_kernel.cu prefilter_kernel.cu ..\engine\continentalness.c -I..\engine -lcudart
```

For the shared-memory permutation baseline, add
`-DTIERED_USE_WARP_PERM=0`. If the warp-register build reports spills or high
local memory, compare `-DTIERED_MIN_BLOCKS_PER_SM=3` before imposing a register
limit. Preserve ptxas register/spill/local-memory output with the result.
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
2. CPU compact initialization: only O6 and O15 state is created for each survivor and uploaded once per GPU scan chunk.
3. GPU tiered scan: `tiered_kernel.cu` evaluates a 2x hex grid, with 280-block spacing and `G=512`. A hit is a center plus two connected mushroom points, encoded as line, triangle, or V geometry.
4. CPU estimate: the cached 0.5x lookup samples at 70-block spacing and counts only the connected low-cell component touching the triple. It no longer extrapolates a perimeter shell.
5. Tier-1 flood: six-octave C flood is a cheap gate; only areas >=3M reach the full 24-octave flood.
6. Final output: only a full 24-octave area >=4M is a valid large mushroom island.

Key units are `step_05x=70`, `step_1x=140`, `step_2x=280`, `G=512`, target
`4_000_000`, and the tier-1 gate `3_000_000`. The GPU O6+O15 threshold near
`-1.00` is intentionally lenient; the CPU flood remains the authority.

The runtime hunt uses 16 flood workers. `benchmark_tiered.py` defaults to 24
workers and should be explicitly run with `HUNT_CPU_WORKERS=24` for pressure
measurements.

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

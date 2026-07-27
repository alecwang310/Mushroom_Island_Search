# Tiered Kernel Investigation

Date: July 27, 2026

## Baseline

The current production path is `gpu/tiered_kernel.cu` with warp-register
permutation tables, `G=512`, `step_2x=300`, threshold `-0.95`, and the O6+O15
two-octave scan. Measurements use the RTX 5080 / `sm_120` Windows host with
identical deterministic survivor seeds.

The latest production warp path measured:

- Pure CUDA kernel: about `158.6k survivors/s` at `G=512`.
- Full tiered DLL call: about `143.6k survivors/s`.
- Live GPU/CPU pipeline: about `133k survivors/s` with the 2,048-task bounded
  verification queue.

## Perlin Prefix LUT Option

Each `perlin_warp` evaluation has seven permutation-pair lookups. The first
three are independent of z:

1. `P[x]`.
2. `P[P[x] + h2]`.
3. `P[P[x+1] + h2]`.

`h2`, `d2`, and `t2` are cached because the tiered scan evaluates the y=0
path. The remaining four pair lookups add the z-cell hash and therefore depend
on z.

For a full 32x32 tile, `cell_z = warp + wave * 8`, so row parity is constant
for all four waves of a warp. A register prefix LUT could therefore compute the
first three pairs once per warp per tile and reuse them across four waves. The
stored prefix would be four packed bytes per octave, or four additional
32-bit registers for both octaves. This option is set aside until the table
storage path is settled because it increases register pressure and does not
address the fact that the current warp-shuffle implementation is slower than
the shared-memory reference.

Seed-wide LUT storage is not promising: each seed executes one block, x-cell
coordinates change across tiles, and a full-seed table would consume much more
storage without cross-seed reuse. A shared tile prefix would save duplicate
prefix work but would add shared loads for every cell.

## Warp Versus Shared A/B

The existing shared-memory reference was compiled with
`-DTIERED_USE_WARP_PERM=0`. The results were:

| Path | G=512 pure kernel | G=448 pure kernel |
| --- | ---: | ---: |
| Warp-register permutation | `158.6k/s` | `206.7k/s` |
| Shared permutation | `226.5k/s` | `296.0k/s` |

The shared path is consistently about `1.43x` faster. Both builds use 64
registers/thread and have zero spills. The shared build uses about 2.2 KiB of
shared memory for the two permutation tables; the warp build uses only the row
masks and hit counter.

The shared Nsight Compute capture reports:

- `53.31%` excessive shared-load wavefronts, averaging about `2.1-way`
  conflicts.
- Only `45.98%` shared-memory throughput and `0.82%` DRAM throughput.
- `90.77%` of cycles with at least one eligible warp.
- Much less sampled math-pipe pressure than the warp path.

The warp capture reports `84.64%` eligible cycles and substantially more
math-pipe, wait, and scoreboard stalls. The warp path performs three dynamic
`shfl.sync.idx` operations plus `prmt.b32` for every permutation pair, with a
long dependent chain across seven pairs. The shared path performs two indexed
shared loads per pair; conflicts expand the load wavefronts but do not saturate
the shared pipeline.

At `G=512` and `158,638 survivors/s`, the warp path issues approximately:

- `344,064` warp-level shuffle instructions per seed.
- `54.6 billion` warp shuffle instructions per second.
- `18.2 billion` `PRMT` instructions per second.

The shared path issues approximately `229,376` shared-load instructions per
seed, or `52.0 billion` shared-load instructions per second at `226,542/s`,
before conflict wavefront expansion. These are aggregate issue rates, not
isolated instruction latencies, but they show that the shared path replaces a
larger dependent instruction sequence with an LSU workload that still has
substantial headroom.

The current 53% conflict percentage is therefore an optimization warning, not
proof that shared memory is the bottleneck. The shared path can be faster while
having more conflicts because it removes the shuffle/`PRMT` dependency chain and
lets the compiler schedule ordinary loads more effectively.

The two A/B runs differed by 15 GPU hits at `G=512` (`36,156` versus `36,171`),
so a direct hit-coordinate comparison is required before changing the default
permutation representation.

## Low-Complexity Shared-Memory Upgrade

The shared byte-table path performs two shared loads for every adjacent
permutation pair: one for `P[index]` and one for `P[index + 1]`. The values are
only used as two eight-bit results, so the table can instead store
`P[index] | (P[index + 1] << 8)` in one 32-bit shared entry. This preserves the
exact lookup sequence and reduces the shared-load instruction count by half;
it does not skip any Perlin lookup or add register-resident state. The source
keeps `TIERED_SHARED_PACKED_PAIRS=0` as an A/B switch for the original layout.

This packed layout is the first optimization to benchmark because it avoids
the register pressure and dependent shuffle/`PRMT` chain of the warp path while
also reducing the shared-memory traffic. Its remaining risk is random-index
bank conflicts, which should affect one load instruction rather than two.

## Current Upgrade

The first implementation experiment is a transposed shared permutation table.
It stores entries as `perm[index * 32 + lane]`, so every lane accesses its own
bank regardless of the permutation index. With packed pairs it uses 64 KiB for
the two tables; with byte entries it uses the same 64 KiB footprint. It keeps
the shared-load code visible to the compiler and avoids the current
random-index bank conflicts. It is compile-time disabled by default while it
is benchmarked against both existing paths.

## Test Discipline

Use the same deterministic survivor range, `G`, threshold, and scan limit for
every path. Record pure kernel time separately from DLL time and hit decoding.
Do not commit Nsight reports, CSV exports, DLLs, or generated JSONL files.

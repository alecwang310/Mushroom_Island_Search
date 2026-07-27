# Tiered Kernel Investigation

Date: July 27, 2026

## Baseline

The current production path is `gpu/tiered_kernel.cu` with ordinary packed
shared permutation pairs, `G=512`, `step_2x=300`, threshold `-0.95`, and the
O6+O15 two-octave scan. Measurements use the RTX 5080 / `sm_120` Windows host
with one materialized deterministic survivor buffer so that every A/B build
sees the same seeds and produces comparable work.

The current fixed-buffer baseline measured:

- Pure CUDA kernel: `1.0888 s` for `262,144` survivors, or `240.8k/s`.
- Hit output: exactly `36,175` sorted records across the compared builds.
- Full DLL-call throughput: approximately `203-208k survivors/s`, including
  compact CPU initialization and host timing overhead.

The old warp-register result is retained below as a historical control, not as
the production baseline. Build it with `-DTIERED_USE_WARP_PERM=1`.

## Perlin Prefix LUT Option

Each `perlin_shared` or `perlin_warp` evaluation has seven permutation-pair
lookups. The first three are independent of z:

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
32-bit registers for both octaves. This option remains set aside because it
increases register pressure and the packed shared path is already faster than
the warp-shuffle reference.

Seed-wide LUT storage is not promising: each seed executes one block, x-cell
coordinates change across tiles, and a full-seed table would consume much more
storage without cross-seed reuse. A shared tile prefix would save duplicate
prefix work but would add shared loads for every cell.

## Warp Versus Shared A/B

The fixed-buffer comparison at `G=512` is:

| Path | Pure CUDA kernel | ptxas registers/shared memory |
| --- | ---: | ---: |
| Ordinary packed shared | `240.8k/s` | `64` / `2,180 B` |
| Shared byte-table control | `225.9k/s` | `64` / `2,180 B` |
| Warp-register historical control | approximately `158k/s` | `64` / `132 B` |

The packed shared path is about `1.52x` faster than the warp-register control
on the fixed buffer. All three builds are spill-free. The warp path performs
three dynamic `shfl.sync.idx` operations plus `prmt.b32` for every permutation
pair, with a long dependent chain across seven pairs. The shared path performs
one indexed shared load per packed pair; conflicts expand the load wavefronts,
but they do not saturate the shared pipeline.

At `G=512` and the historical `158,638 survivors/s`, the warp path issued:

- `344,064` warp-level shuffle instructions per seed.
- `54.6 billion` warp shuffle instructions per second.
- `18.2 billion` `PRMT` instructions per second.

The current packed shared path issues approximately `229,376` shared-load
instructions per seed, or `55.2 billion` shared-load instructions per second
at `240,759/s`, before conflict wavefront expansion. These are aggregate issue
rates, not isolated instruction latencies, but they show that the shared path
replaces a larger dependent instruction sequence with an LSU workload that
still has measurable headroom.

The current `50.9%` conflict percentage is therefore an optimization warning, not
proof that shared memory is the bottleneck. The shared path can be faster while
having more conflicts because it removes the shuffle/`PRMT` dependency chain and
lets the compiler schedule ordinary loads more effectively.

The fixed-buffer A/B builds all returned the same `36,175` sorted hit records,
including seed, coordinates, and geometry code. This removes the earlier
confounding effect from regenerating a nondeterministically compacted
survivor prefix for each run.

## Packed Shared-Memory Upgrade

The shared byte-table path performs two shared loads for every adjacent
permutation pair: one for `P[index]` and one for `P[index + 1]`. The values are
only used as two eight-bit results, so the table can instead store
`P[index] | (P[index + 1] << 8)` in one 32-bit shared entry. This preserves the
exact lookup sequence and reduces the shared-load instruction count by half;
it does not skip any Perlin lookup or add register-resident state. The source
keeps `TIERED_SHARED_PACKED_PAIRS=0` as an A/B switch for the original layout.

This packed layout avoids the register pressure and dependent shuffle/`PRMT`
chain of the warp path while also reducing the shared-memory lookup count. The
fixed-buffer comparison is recorded in the A/B section above. It is the
production default; `TIERED_SHARED_PACKED_PAIRS=0` remains only as a control.

Nsight Compute on the packed path still reports approximately `50.9%` shared
load wavefront expansion at an average `2.0-way` conflict. This is expected
for random permutation indices, but the packed table reduces the number of
conflict-prone load instructions by half. The kernel remains spill-free and
DRAM-light, so further work should target instruction scheduling or a smaller
lookup representation rather than assuming that removing all conflicts would
double throughput.

## Logical Memory Traffic Ceiling

This is a calculation of the work performed by the current kernel, not a claim
about DRAM bandwidth. The permutation tables and row masks reside in shared
memory; Nsight reports negligible DRAM traffic. At `G=512`, one seed evaluates
`512 x 512 = 262,144` cells. Each cell evaluates two octaves, and each Perlin
call performs seven packed adjacent-pair lookups:

| Operation | Bytes per seed | Rate at `240,759 survivors/s` |
| --- | ---: | ---: |
| Packed permutation reads (`14 x 4 B/cell`) | `14,680,064 B` (`14.0 MiB`) | `3.53 TB/s` |
| Row-mask logical reads | `3,080,192 B` (`2.94 MiB`) | `0.742 TB/s` |
| Row-mask writes | `32,768 B` (`32 KiB`) | `7.89 GB/s` |
| Permutation-table initialization writes | `2,048 B` | `0.493 GB/s` |

The dominant logical traffic is therefore about `4.28 TB/s` of shared-memory
reads, overwhelmingly from the packed permutation table. Row-mask reads are
mostly warp broadcasts, and the writes are negligible compared with the pair
loads. These figures are logical request rates after multiplying by the
measured seed throughput; they are not a usable DRAM-bandwidth target.

The closest measured isolation of shared lookup cost is the packed-versus-byte
control. Their kernel times were `1.088822 s` and `1.160628 s` for the same
`262,144` survivors. Treating the removed second byte-table load as the only
changed component assigns `0.071806 s`, or `6.595%` of packed runtime, to one
packed lookup-load component. If eliminating conflicts could halve that entire
component, the upper estimate is:

```text
240,759 / (1 - 0.5 x 0.06595) = approximately 248,969 survivors/s
```

That is only about `3.4%` above the current result. If the `50.9%` Nsight
wavefront expansion is interpreted literally as `1.509x` serialized load work,
the corresponding estimate is approximately `246,237 survivors/s`; the
reasonable no-conflict range is therefore about `246-249K survivors/s`.
Removing all measured shared-load time, an intentionally unrealistic upper
bound, would be approximately `257.8K/s`. Without changing the arithmetic
chain, reducing launch waste, or changing the algorithm, bank-conflict removal
alone cannot approach a 2x speedup.

## CPU Register and Clock Assessment

The CPU verification path calls the C implementation through a 24-worker
thread pool, so the Python GIL is not the inner-loop limiter. The estimate path
initializes one exact `ContEngine`, samples a connected 0.5x lattice, and runs
scalar double-precision Perlin evaluations. The flood paths add dynamic queue
and open-addressing hash-set allocations, pointer-chasing probes, branch-heavy
neighbor tests, and repeated `cont_sample` calls.

Consequently, extra CPU architectural registers are unlikely to produce a
large standalone speedup. The compiler already keeps the short-lived Perlin
values in registers; the flood fill is primarily latency/cache/branch and
allocator bound. Higher clocks and more physical cores can improve throughput
until the worker pool reaches memory-system or allocation limits, but they do
not change the GPU lookup bottleneck. CPU-specific compilation such as `-O3`
with the target's native ISA should be measured separately from algorithmic
changes, while the CPU verification result must remain the FP32/FP64 reference.

## Partial Transpose Experiment

The tested alternative is a transposed shared permutation table. It stores
entries as `perm[index * replicas + (lane & (replicas - 1))]`.
With `replicas=32`, every lane accesses its own bank and the two packed tables
use 64 KiB. With `replicas=16`, the two tables use 32 KiB plus row-mask state
and each lookup has at most approximately a 2-way lane alias; `replicas=8`
uses 16 KiB and caps the alias at approximately 4-way. The implementation is
compile-time disabled by default. A 32-replica full transpose would exceed the
target's `49,152`-byte per-block limit, while the 16- and 8-replica variants
fit and were benchmarked against the ordinary packed path.

The fixed-survivor Windows benchmark used the same `262,144` survivor buffer
for every build and returned the same `36,175` sorted hit records:

| Build | Kernel time | Throughput | ptxas result |
| --- | ---: | ---: | --- |
| Ordinary packed shared | `1.0888 s` | `240.8k/s` | `64` registers, no spills |
| Transposed, 16 replicas, launch bound 4 | `1.1116 s` | `235.8k/s` | `64` registers, 4-byte spill |
| Transposed, 16 replicas, launch bound 3 | `1.1168 s` | `234.7k/s` | `80` registers, no spills |
| Transposed, 8 replicas, launch bound 4 | `1.0890 s` | `240.7k/s` | `64` registers, 4-byte spill |
| Transposed, 8 replicas, launch bound 3 | `1.1191 s` | `234.2k/s` | `80` registers, no spills |

The partial transpose therefore does not beat the ordinary packed table on
this GPU. The 16-replica version spends more shared memory without recovering
enough scheduler time, and reducing launch bounds to remove the tiny spill
increases register pressure and loses throughput. Keep the ordinary packed
default; retain the replica count only as a future architecture-specific A/B
switch.

A concurrent 24-worker CPU sample processed `4,096` GPU hits at `55.2k hits/s`
with a peak queue of `48` and no blocked submissions, while the GPU produced
about `28.7k hits/s` in the same benchmark. No sampled estimate exceeded 4M,
so this result measures the estimate path rather than flood-fill service. It
still confirms that adding CPU registers alone is unlikely to fix the current
GPU-limited run; a positive-heavy flood benchmark is needed before changing
the allocator or flood data structures.

## Test Discipline

Use the same deterministic survivor range, `G`, threshold, and scan limit for
every path. Record pure kernel time separately from DLL time and hit decoding.
Do not commit Nsight reports, CSV exports, DLLs, or generated JSONL files.

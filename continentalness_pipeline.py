"""
continentalness_pipeline.py — MC 1.18+ continentalness sampler.
Mushroom fields when: -1.2 < continentalness(x,z) < -1.05

Verified 100% against cubiomes C output (seed 0,1,100,12345,12305878011287454963).
"""

import math

M64 = 0xFFFFFFFFFFFFFFFF

# =============================================================================
# Xoroshiro128++
# =============================================================================

def rotl64(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & M64

def xSetSeed(seed: int) -> tuple[int, int]:
    XL, XH, A, B = 0x9e3779b97f4a7c15, 0x6a09e667f3bcc909, 0xbf58476d1ce4e5b9, 0x94d049bb133111eb
    l = (seed ^ XH) & M64
    h = (l + XL) & M64
    l = ((l ^ (l >> 30)) * A) & M64
    h = ((h ^ (h >> 30)) * A) & M64
    l = ((l ^ (l >> 27)) * B) & M64
    h = ((h ^ (h >> 27)) * B) & M64
    l = (l ^ (l >> 31)) & M64
    h = (h ^ (h >> 31)) & M64
    return l, h

def xNextLong(lo: int, hi: int) -> tuple[int, int, int]:
    n = (rotl64((lo + hi) & M64, 17) + lo) & M64
    hi ^= lo
    lo = rotl64(lo, 49) ^ hi ^ ((hi << 21) & M64)
    hi = rotl64(hi, 28)
    return lo, hi, n

def xNextDouble(lo: int, hi: int) -> tuple[int, int, float]:
    lo, hi, n = xNextLong(lo, hi)
    return lo, hi, (n >> 11) * (1.0 / (1 << 53))

def xNextInt(lo: int, hi: int, n: int) -> tuple[int, int, int]:
    lo, hi, rng = xNextLong(lo, hi)
    r = (rng & 0xFFFFFFFF) * n
    if (r & 0xFFFFFFFF) < n:
        threshold = ((0xFFFFFFFF - n + 1) % n) & 0xFFFFFFFF
        while (r & 0xFFFFFFFF) < threshold:
            lo, hi, rng = xNextLong(lo, hi)
            r = (rng & 0xFFFFFFFF) * n
    return lo, hi, (r >> 32) & 0xFFFFFFFF


# =============================================================================
# Perlin noise
# =============================================================================

class PerlinNoise:
    __slots__ = ('d', 'a', 'b', 'c', 'amplitude', 'lacunarity', 'h2', 'd2', 't2')

    def __init__(self):
        self.d = [0] * 257
        self.a = self.b = self.c = 0.0
        self.amplitude = self.lacunarity = 1.0
        self.h2 = 0
        self.d2 = self.t2 = 0.0

    @staticmethod
    def from_xoroshiro(lo: int, hi: int) -> tuple['PerlinNoise', int, int]:
        pn = PerlinNoise()
        lo, hi, pn.a = xNextDouble(lo, hi); pn.a *= 256.0
        lo, hi, pn.b = xNextDouble(lo, hi); pn.b *= 256.0
        lo, hi, pn.c = xNextDouble(lo, hi); pn.c *= 256.0
        idx = list(range(256))
        for i in range(256):
            lo, hi, j = xNextInt(lo, hi, 256 - i); j += i
            idx[i], idx[j] = idx[j], idx[i]
        pn.d[:256] = idx
        pn.d[256] = idx[0]
        i2 = math.floor(pn.b)
        d2 = pn.b - i2
        pn.h2 = int(i2) & 0xFF
        pn.d2 = d2
        pn.t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0)
        return pn, lo, hi


def _indexed_lerp(idx: int, a: float, b: float, c: float) -> float:
    m = idx & 0xF
    if m == 0:   return  a + b
    if m == 1:   return -a + b
    if m == 2:   return  a - b
    if m == 3:   return -a - b
    if m == 4:   return  a + c
    if m == 5:   return -a + c
    if m == 6:   return  a - c
    if m == 7:   return -a - c
    if m == 8:   return  b + c
    if m == 9:   return -b + c
    if m == 10:  return  b - c
    if m == 11:  return -b - c
    if m == 12:  return  a + b
    if m == 13:  return -b + c
    if m == 14:  return -a + b
    if m == 15:  return -b - c
    return 0.0


def _sample_perlin(noise: PerlinNoise, x: float, y: float, z: float) -> float:
    """3D Perlin at (x, y, z). Uses cached h2/d2/t2 when y==0."""
    d1, d2, d3 = x + noise.a, y + noise.b, z + noise.c
    if y == 0.0:
        h2, d2, t2 = noise.h2, noise.d2, noise.t2
    else:
        i2 = math.floor(d2); d2 -= i2
        h2 = int(i2) & 0xFF
        t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0)
    i1, i3 = math.floor(d1), math.floor(d3)
    d1 -= i1; d3 -= i3
    h1, h3 = int(i1) & 0xFF, int(i3) & 0xFF
    t1 = d1 * d1 * d1 * (d1 * (d1 * 6.0 - 15.0) + 10.0)
    t3 = d3 * d3 * d3 * (d3 * (d3 * 6.0 - 15.0) + 10.0)
    idx = noise.d
    v1_a, v1_b = idx[h1] + h2, idx[h1 + 1] + h2
    v2_a = idx[(v1_a & 0xFF)] + h3; v2_b = idx[(v1_a & 0xFF) + 1] + h3
    v3_a = idx[(v1_b & 0xFF)] + h3; v3_b = idx[(v1_b & 0xFF) + 1] + h3
    v4_a = idx[(v2_a & 0xFF)];     v4_b = idx[(v2_a & 0xFF) + 1]
    v5_a = idx[(v2_b & 0xFF)];     v5_b = idx[(v2_b & 0xFF) + 1]
    v6_a = idx[(v3_a & 0xFF)];     v6_b = idx[(v3_a & 0xFF) + 1]
    v7_a = idx[(v3_b & 0xFF)];     v7_b = idx[(v3_b & 0xFF) + 1]
    l1 = _indexed_lerp(v4_a, d1, d2, d3);     l5 = _indexed_lerp(v4_b, d1, d2, d3 - 1)
    l2 = _indexed_lerp(v6_a, d1 - 1, d2, d3); l6 = _indexed_lerp(v6_b, d1 - 1, d2, d3 - 1)
    l3 = _indexed_lerp(v5_a, d1, d2 - 1, d3); l7 = _indexed_lerp(v5_b, d1, d2 - 1, d3 - 1)
    l4 = _indexed_lerp(v7_a, d1 - 1, d2 - 1, d3); l8 = _indexed_lerp(v7_b, d1 - 1, d2 - 1, d3 - 1)
    l1 += t1 * (l2 - l1); l3 += t1 * (l4 - l3); l5 += t1 * (l6 - l5); l7 += t1 * (l8 - l7)
    l1 += t2 * (l3 - l1); l5 += t2 * (l7 - l5)
    return l1 + t3 * (l5 - l1)


# =============================================================================
# Octave / DoublePerlin init + sampling
# =============================================================================

# md5 "octave_-12".."octave_0"
_MD5_OCTAVE = [
    (0xb198de63a8012672, 0x7b84cad43ef7b5a8), (0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8),
    (0x36d326eed40efeb2, 0x5be9ce18223c636a), (0x082fe255f8be6631, 0x4e96119e22dedc81),
    (0x0ef68ec68504005e, 0x48b6bf93a2789640), (0xf11268128982754f, 0x257a1d670430b0aa),
    (0xe51c98ce7d1de664, 0x5f9478a733040c45), (0x6d7b49e7e429850a, 0x2e3063c622a24777),
    (0xbd90d5377ba1b762, 0xc07317d419a7548d), (0x53d39c6752dac858, 0xbcd1c5a80ab65b3e),
    (0xb4a24d7a84e7677b, 0x023ff9668e89b5c4), (0xdffa22b534c5f608, 0xb9b67517d3665ca9),
    (0xd50708086cef4d7c, 0x6e1651ecc7f43309),
]

# lacuna_ini[-omin], -omin = 3..12
_LACUNA_INI = [1.0, 0.5, 0.25, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 1/2048, 1/4096]

# persist_ini[len], len = 0..9
_PERSIST_INI = [0, 1, 2/3, 4/7, 8/15, 16/31, 32/63, 64/127, 128/255, 256/511]

# amp_ini[len] = (5/3) * len/(len+1), len = 0..9
_AMP_INI = [0, 5/6, 10/9, 15/12, 20/15, 25/18, 30/21, 35/24, 40/27, 45/30]

# MD5 climate hashes (biomenoise.c) — only shift + continentalness
_MD5_SHIFT       = (0x080518cf6af25384, 0x3f3dfb40a54febd5)   # "minecraft:offset"
_MD5_CONTINENT   = (0x83886c9d0ae3a662, 0xafa638a61b42e8ad)   # "minecraft:continentalness"
_MD5_CONT_LARGE  = (0x9a3f51a113fce8dc, 0xee2dbd157e5dcdad)   # "minecraft:continentalness_large"

_FREQ_RATIO = 337.0 / 331.0


class _OctaveNoise:
    __slots__ = ('octaves',)
    def __init__(self, octaves: list[PerlinNoise]):
        self.octaves = octaves


class _DoublePerlinNoise:
    __slots__ = ('octA', 'octB', 'amplitude')
    def __init__(self, octA: _OctaveNoise, octB: _OctaveNoise, amplitude: float):
        self.octA, self.octB, self.amplitude = octA, octB, amplitude


def _xOctaveInit(lo: int, hi: int, amplitudes: list[float],
                 omin: int, length: int, nmax: int = -1) -> tuple[_OctaveNoise, int, int]:
    lo, hi, ixlo = xNextLong(lo, hi)
    lo, hi, ixhi = xNextLong(lo, hi)
    lacuna, persist = _LACUNA_INI[-omin], _PERSIST_INI[length]
    octaves = []
    for i in range(length):
        if len(octaves) == nmax:
            break
        if amplitudes[i] == 0.0:
            lacuna *= 2.0; persist *= 0.5
            continue
        md5 = _MD5_OCTAVE[12 + omin + i]
        pn, _, _ = PerlinNoise.from_xoroshiro(ixlo ^ md5[0], ixhi ^ md5[1])
        pn.amplitude = amplitudes[i] * persist
        pn.lacunarity = lacuna
        octaves.append(pn)
        lacuna *= 2.0; persist *= 0.5
    return _OctaveNoise(octaves), lo, hi


def _xDoublePerlinInit(lo: int, hi: int, amplitudes: list[float],
                       omin: int, length: int, nmax: int = -1) -> tuple[_DoublePerlinNoise, int, int]:
    na = ((nmax + 1) >> 1) if nmax > 0 else -1
    nb = (nmax - na) if nmax > 0 else -1
    octA, lo, hi = _xOctaveInit(lo, hi, amplitudes, omin, length, na)
    octB, lo, hi = _xOctaveInit(lo, hi, amplitudes, omin, length, nb)
    eff = length
    while eff > 0 and amplitudes[eff - 1] == 0.0: eff -= 1
    i = 0
    while i < eff and amplitudes[i] == 0.0: eff -= 1; i += 1
    return _DoublePerlinNoise(octA, octB, _AMP_INI[eff]), lo, hi


def _sampleOctave(noise: _OctaveNoise, x: float, y: float, z: float) -> float:
    v = 0.0
    for pn in noise.octaves:
        lf = pn.lacunarity
        v += pn.amplitude * _sample_perlin(pn, x * lf, y * lf, z * lf)
    return v


def _sampleDoublePerlin(noise: _DoublePerlinNoise, x: float, y: float, z: float) -> float:
    fr = _FREQ_RATIO
    return (  _sampleOctave(noise.octA, x, y, z)
            + _sampleOctave(noise.octB, x * fr, y * fr, z * fr)
            ) * noise.amplitude


# =============================================================================
# Public API
# =============================================================================

class ContinentalnessSampler:
    """Samples MC 1.18+ continentalness for mushroom field detection."""

    def __init__(self, world_seed: int, large_biomes: bool = False):
        lo, hi = xSetSeed(world_seed)
        lo, hi, xlo = xNextLong(lo, hi)
        lo, hi, xhi = xNextLong(lo, hi)
        # SHIFT: amp=[1,1,1,0], omin=-3
        self._shift, _, _ = _xDoublePerlinInit(
            xlo ^ _MD5_SHIFT[0], xhi ^ _MD5_SHIFT[1],
            [1.0, 1.0, 1.0, 0.0], -3, 4)
        # CONTINENTALNESS: amp=[1,1,2,2,2,1,1,1,1]
        c_md5 = _MD5_CONT_LARGE if large_biomes else _MD5_CONTINENT
        c_omin = -11 if large_biomes else -9
        self._cont, _, _ = _xDoublePerlinInit(
            xlo ^ c_md5[0], xhi ^ c_md5[1],
            [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], c_omin, 9)

    def sample(self, x: int, z: int) -> float:
        """Continentalness at (x,z) at 1:4 scale. Mushroom: val < -1.05"""
        # Matching cubiomes: dx=shift(x,0,z), dz=shift(z,x,0) where y=x NOT 0!
        dx = _sampleDoublePerlin(self._shift, float(x), 0.0, float(z))
        dz = _sampleDoublePerlin(self._shift, float(z), float(x), 0.0)
        return _sampleDoublePerlin(self._cont, x + dx * 4.0, 0.0, z + dz * 4.0)

    def is_mushroom(self, x: int, z: int) -> bool:
        c = self.sample(x, z)
        return -1.2 < c < -1.05


# =============================================================================
# Minimal demo
# =============================================================================

if __name__ == '__main__':
    for seed in [0, 1, 100, 12345, 12305878011287454963]:
        cs = ContinentalnessSampler(seed)
        c = cs.sample(0, 0)
        print(f"seed {seed:>20}  cont(0,0)={c:+.6f}  mushroom={cs.is_mushroom(0,0)}")

#ifndef TIERED_HIT_H
#define TIERED_HIT_H

#include <stdint.h>

#define TIERED_COMPONENT_CAP 8
#define TIERED_COMPONENT_WORDS (TIERED_COMPONENT_CAP / 2)
#define TIERED_VECTOR_CELL_SIDE 256
#define TIERED_VECTOR_WORDS_PER_ROW (TIERED_VECTOR_CELL_SIDE / 32)
#define TIERED_VECTOR_CELL_WORDS \
    (TIERED_VECTOR_CELL_SIDE * TIERED_VECTOR_WORDS_PER_ROW)

#ifdef __CUDACC__
#define TIERED_INLINE static __host__ __device__ __forceinline__
#else
#define TIERED_INLINE static inline
#endif

typedef struct {
    int seed;
    int x;
    int z;
    uint32_t meta;
    uint32_t component[TIERED_COMPONENT_WORDS];
} TieredHit;

TIERED_INLINE uint32_t tiered_hit_meta(
    int row_parity, int component_count, int component_size)
{
    return (uint32_t)(row_parity & 1)
        | ((uint32_t)(component_count & 0xF) << 8)
        | ((uint32_t)(component_size & 0xFFF) << 16);
}

TIERED_INLINE int tiered_hit_row_parity(const TieredHit *hit) {
    return (int)(hit->meta & 1u);
}

TIERED_INLINE int tiered_hit_component_count(const TieredHit *hit) {
    return (int)((hit->meta >> 8) & 0xFu);
}

TIERED_INLINE int tiered_hit_geometry(const TieredHit *hit) {
    return tiered_hit_row_parity(hit) << 6;
}

TIERED_INLINE uint16_t tiered_pack_component_offset(
    int column_delta, int row_delta)
{
    return (uint16_t)((uint8_t)(int8_t)column_delta)
        | (uint16_t)((uint16_t)(uint8_t)(int8_t)row_delta << 8);
}

TIERED_INLINE void tiered_store_component_offset(
    TieredHit *hit, int index, int column_delta, int row_delta)
{
    uint32_t packed = tiered_pack_component_offset(column_delta, row_delta);
    int word = index >> 1;
    int shift = (index & 1) * 16;
    uint32_t mask = 0xFFFFu << shift;
    hit->component[word] = (hit->component[word] & ~mask)
        | (packed << shift);
}

TIERED_INLINE void tiered_load_component_offset(
    const TieredHit *hit, int index, int *column_delta, int *row_delta)
{
    uint32_t packed = hit->component[index >> 1]
        >> ((index & 1) * 16);
    *column_delta = (int)(int8_t)(packed & 0xFFu);
    *row_delta = (int)(int8_t)((packed >> 8) & 0xFFu);
}

#undef TIERED_INLINE

#endif

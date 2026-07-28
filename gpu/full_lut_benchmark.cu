/*
 * full_lut_benchmark.cu — synthetic compressed full-scale hash-LUT A/B.
 *
 * This is a benchmark-only experiment. It does not change tiered_kernel.cu.
 * Each LUT entry packs the eight final gradient hashes into eight nibbles;
 * this is exact for the current grad_dot(), which consumes hash & 15.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

#define LUT_ENTRIES 65536
#define LUT_THREADS 256

__device__ __forceinline__ float lut_fade(float value) {
    return value * value * value
        * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float lut_grad_dot(
    uint32_t hash, float x, float y, float z)
{
    uint32_t h = hash & 15u;
    float u = h < 8u ? x : y;
    float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    uint32_t u_bits = __float_as_uint(u)
        ^ ((h & 1u) * 0x80000000u);
    uint32_t v_bits = __float_as_uint(v)
        ^ (((h >> 1) & 1u) * 0x80000000u);
    return __uint_as_float(u_bits) + __uint_as_float(v_bits);
}

__device__ __forceinline__ uint32_t lut_pair(
    const uint32_t *perm, uint32_t index)
{
    return perm[index & 0xFFu];
}

__device__ __forceinline__ uint32_t pack_hash_chain(
    const uint32_t *perm, uint32_t h1, uint32_t h3, uint32_t h2)
{
    uint32_t pair = lut_pair(perm, h1);
    uint32_t va = (pair & 0xFFu) + h2;
    uint32_t vb = (pair >> 8) + h2;

    pair = lut_pair(perm, va);
    uint32_t v2a = (pair & 0xFFu) + h3;
    uint32_t v2b = (pair >> 8) + h3;
    pair = lut_pair(perm, vb);
    uint32_t v3a = (pair & 0xFFu) + h3;
    uint32_t v3b = (pair >> 8) + h3;

    pair = lut_pair(perm, v2a);
    uint32_t v4a = pair & 0x0Fu;
    uint32_t v4b = (pair >> 8) & 0x0Fu;
    pair = lut_pair(perm, v2b);
    uint32_t v5a = pair & 0x0Fu;
    uint32_t v5b = (pair >> 8) & 0x0Fu;
    pair = lut_pair(perm, v3a);
    uint32_t v6a = pair & 0x0Fu;
    uint32_t v6b = (pair >> 8) & 0x0Fu;
    pair = lut_pair(perm, v3b);
    uint32_t v7a = pair & 0x0Fu;
    uint32_t v7b = (pair >> 8) & 0x0Fu;

    return v4a | (v4b << 4) | (v5a << 8) | (v5b << 12)
        | (v6a << 16) | (v6b << 20) | (v7a << 24) | (v7b << 28);
}

__device__ __forceinline__ float evaluate_packed(
    uint32_t packed, float dx, float dz, float d2, float t2)
{
    uint32_t v4a = (packed >> 0) & 0x0Fu;
    uint32_t v4b = (packed >> 4) & 0x0Fu;
    uint32_t v5a = (packed >> 8) & 0x0Fu;
    uint32_t v5b = (packed >> 12) & 0x0Fu;
    uint32_t v6a = (packed >> 16) & 0x0Fu;
    uint32_t v6b = (packed >> 20) & 0x0Fu;
    uint32_t v7a = (packed >> 24) & 0x0Fu;
    uint32_t v7b = (packed >> 28) & 0x0Fu;

    float tx = lut_fade(dx);
    float tz = lut_fade(dz);
    float l1 = lut_grad_dot(v4a, dx, d2, dz);
    float l5 = lut_grad_dot(v4b, dx, d2, dz - 1.0f);
    float l2 = lut_grad_dot(v6a, dx - 1.0f, d2, dz);
    float l6 = lut_grad_dot(v6b, dx - 1.0f, d2, dz - 1.0f);
    float l3 = lut_grad_dot(v5a, dx, d2 - 1.0f, dz);
    float l7 = lut_grad_dot(v5b, dx, d2 - 1.0f, dz - 1.0f);
    float l4 = lut_grad_dot(v7a, dx - 1.0f, d2 - 1.0f, dz);
    float l8 = lut_grad_dot(v7b, dx - 1.0f, d2 - 1.0f, dz - 1.0f);

    l1 = fmaf(tx, l2 - l1, l1);
    l3 = fmaf(tx, l4 - l3, l3);
    l5 = fmaf(tx, l6 - l5, l5);
    l7 = fmaf(tx, l8 - l7, l7);
    l1 = fmaf(t2, l3 - l1, l1);
    l5 = fmaf(t2, l7 - l5, l5);
    return fmaf(tz, l5 - l1, l1);
}

__device__ __forceinline__ void synthetic_sample(
    uint32_t cell, uint32_t block, int log_grid,
    uint32_t *h16, uint32_t *h36, uint32_t *h115, uint32_t *h315,
    float *dx6, float *dz6, float *dx15, float *dz15)
{
    uint32_t x = cell & ((1u << log_grid) - 1u);
    uint32_t z = cell >> log_grid;
    *h16 = (x * 17u + z * 31u + block * 7u) & 0xFFu;
    *h36 = (x * 29u + z * 13u + block * 11u) & 0xFFu;
    *h115 = (x * 23u + z * 37u + block * 19u) & 0xFFu;
    *h315 = (x * 41u + z * 17u + block * 5u) & 0xFFu;
    *dx6 = ((x * 13u + z * 7u + block * 3u) & 1023u) / 1024.0f;
    *dz6 = ((x * 11u + z * 19u + block * 2u) & 1023u) / 1024.0f;
    *dx15 = ((x * 7u + z * 23u + block * 5u) & 1023u) / 1024.0f;
    *dz15 = ((x * 19u + z * 29u + block * 7u) & 1023u) / 1024.0f;
}

__global__ void build_full_lut(
    const uint32_t *perm6, const uint32_t *perm15,
    uint32_t h26, uint32_t h215,
    uint32_t *lut6, uint32_t *lut15)
{
    __shared__ uint32_t shared_perm6[256];
    __shared__ uint32_t shared_perm15[256];
    int tid = threadIdx.x;
    shared_perm6[tid] = perm6[tid];
    shared_perm15[tid] = perm15[tid];
    __syncthreads();

    for (uint32_t key = (uint32_t)tid;
         key < LUT_ENTRIES; key += blockDim.x) {
        uint32_t h1 = key >> 8;
        uint32_t h3 = key & 0xFFu;
        lut6[key] = pack_hash_chain(shared_perm6, h1, h3, h26);
        lut15[key] = pack_hash_chain(shared_perm15, h1, h3, h215);
    }
}

__global__ void lut_extract_kernel(
    const uint32_t *lut6, const uint32_t *lut15,
    int grid, uint32_t *output)
{
    uint32_t accumulator = 0u;
    uint32_t cell_count = (uint32_t)(grid * grid);
    int log_grid = __ffs(grid) - 1;
    for (uint32_t cell = (uint32_t)threadIdx.x;
         cell < cell_count; cell += blockDim.x) {
        uint32_t h16, h36, h115, h315;
        float dx6, dz6, dx15, dz15;
        synthetic_sample(cell, (uint32_t)blockIdx.x, log_grid,
                         &h16, &h36, &h115, &h315,
                         &dx6, &dz6, &dx15, &dz15);
        uint32_t packed6 = lut6[(h16 << 8) | h36];
        uint32_t packed15 = lut15[(h115 << 8) | h315];
        accumulator ^= packed6 + (packed15 * 0x9E3779B9u);
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = accumulator;
}

__global__ void lut_compute_kernel(
    const uint32_t *lut6, const uint32_t *lut15,
    int grid, float *output)
{
    float accumulator = 0.0f;
    uint32_t cell_count = (uint32_t)(grid * grid);
    int log_grid = __ffs(grid) - 1;
    for (uint32_t cell = (uint32_t)threadIdx.x;
         cell < cell_count; cell += blockDim.x) {
        uint32_t h16, h36, h115, h315;
        float dx6, dz6, dx15, dz15;
        synthetic_sample(cell, (uint32_t)blockIdx.x, log_grid,
                         &h16, &h36, &h115, &h315,
                         &dx6, &dz6, &dx15, &dz15);
        float value = 0.501f * evaluate_packed(
            lut6[(h16 << 8) | h36], dx6, dz6, 0.37f, 0.61f);
        value += 0.501f * evaluate_packed(
            lut15[(h115 << 8) | h315], dx15, dz15, 0.43f, 0.67f);
        accumulator += value;
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = accumulator;
}

__global__ void chain_compute_kernel(
    const uint32_t *perm6, const uint32_t *perm15,
    uint32_t h26, uint32_t h215,
    int grid, float *output)
{
    __shared__ uint32_t shared_perm6[256];
    __shared__ uint32_t shared_perm15[256];
    int tid = threadIdx.x;
    shared_perm6[tid] = perm6[tid];
    shared_perm15[tid] = perm15[tid];
    __syncthreads();

    float accumulator = 0.0f;
    uint32_t cell_count = (uint32_t)(grid * grid);
    int log_grid = __ffs(grid) - 1;
    for (uint32_t cell = (uint32_t)tid;
         cell < cell_count; cell += blockDim.x) {
        uint32_t h16, h36, h115, h315;
        float dx6, dz6, dx15, dz15;
        synthetic_sample(cell, (uint32_t)blockIdx.x, log_grid,
                         &h16, &h36, &h115, &h315,
                         &dx6, &dz6, &dx15, &dz15);
        float value = 0.501f * evaluate_packed(
            pack_hash_chain(shared_perm6, h16, h36, h26),
            dx6, dz6, 0.37f, 0.61f);
        value += 0.501f * evaluate_packed(
            pack_hash_chain(shared_perm15, h115, h315, h215),
            dx15, dz15, 0.43f, 0.67f);
        accumulator += value;
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = accumulator;
}

static void make_random_pairs(std::array<uint32_t, 256> *output,
                              uint64_t seed)
{
    std::array<uint8_t, 256> values{};
    for (int i = 0; i < 256; ++i) values[i] = (uint8_t)i;
    std::mt19937_64 generator(seed);
    std::shuffle(values.begin(), values.end(), generator);
    for (int i = 0; i < 256; ++i) {
        (*output)[i] = (uint32_t)values[i]
            | ((uint32_t)values[(i + 1) & 255] << 8);
    }
}

static int check_cuda(cudaError_t error, const char *what)
{
    if (error == cudaSuccess) return 0;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    return -1;
}

template <typename Launch>
static int time_launch(Launch launch, int repeats, double *milliseconds)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)")) return -1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        cudaEventDestroy(start);
        return -1;
    }
    if (check_cuda(cudaEventRecord(start), "cudaEventRecord(start)")) return -1;
    for (int i = 0; i < repeats; ++i) {
        launch();
        if (check_cuda(cudaGetLastError(), "kernel launch")) return -1;
    }
    if (check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)")) return -1;
    if (check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)"))
        return -1;
    float elapsed = 0.0f;
    if (check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
                   "cudaEventElapsedTime")) return -1;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    *milliseconds = (double)elapsed / (double)repeats;
    return 0;
}

extern "C" __declspec(dllexport) int full_lut_benchmark(
    int blocks, int grid, int repeats, double *output, int output_count)
{
    if (!output || output_count < 8 || blocks <= 0 || repeats <= 0
        || grid <= 0 || (grid & (grid - 1)) != 0 || grid > 1024)
        return -2;

    std::array<uint32_t, 256> host_perm6{};
    std::array<uint32_t, 256> host_perm15{};
    make_random_pairs(&host_perm6, 0x6A09E667F3BCC909ULL);
    make_random_pairs(&host_perm15, 0xBB67AE8584CAA73BULL);

    uint32_t *device_perm6 = nullptr;
    uint32_t *device_perm15 = nullptr;
    uint32_t *device_lut6 = nullptr;
    uint32_t *device_lut15 = nullptr;
    uint32_t *device_extract = nullptr;
    float *device_compute = nullptr;
    size_t output_size = (size_t)blocks * LUT_THREADS * sizeof(float);
    size_t extract_size = (size_t)blocks * LUT_THREADS * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (device_perm6) cudaFree(device_perm6);
        if (device_perm15) cudaFree(device_perm15);
        if (device_lut6) cudaFree(device_lut6);
        if (device_lut15) cudaFree(device_lut15);
        if (device_extract) cudaFree(device_extract);
        if (device_compute) cudaFree(device_compute);
    };

    if (check_cuda(cudaMalloc(&device_perm6, sizeof(host_perm6)),
                   "cudaMalloc(perm6)")) { cleanup(); return -1; }
    if (check_cuda(cudaMalloc(&device_perm15, sizeof(host_perm15)),
                   "cudaMalloc(perm15)")) { cleanup(); return -1; }
    if (check_cuda(cudaMalloc(&device_lut6, LUT_ENTRIES * sizeof(uint32_t)),
                   "cudaMalloc(lut6)")) { cleanup(); return -1; }
    if (check_cuda(cudaMalloc(&device_lut15, LUT_ENTRIES * sizeof(uint32_t)),
                   "cudaMalloc(lut15)")) { cleanup(); return -1; }
    if (check_cuda(cudaMalloc(&device_extract, extract_size),
                   "cudaMalloc(extract)")) { cleanup(); return -1; }
    if (check_cuda(cudaMalloc(&device_compute, output_size),
                   "cudaMalloc(compute)")) { cleanup(); return -1; }
    if (check_cuda(cudaMemcpy(device_perm6, host_perm6.data(),
                              sizeof(host_perm6), cudaMemcpyHostToDevice),
                   "cudaMemcpy(perm6)")) { cleanup(); return -1; }
    if (check_cuda(cudaMemcpy(device_perm15, host_perm15.data(),
                              sizeof(host_perm15), cudaMemcpyHostToDevice),
                   "cudaMemcpy(perm15)")) { cleanup(); return -1; }

    auto launch_build = [&]() {
        build_full_lut<<<1, LUT_THREADS>>>(
            device_perm6, device_perm15, 0x37u, 0x43u,
            device_lut6, device_lut15);
    };
    auto launch_extract = [&]() {
        lut_extract_kernel<<<blocks, LUT_THREADS>>>(
            device_lut6, device_lut15, grid, device_extract);
    };
    auto launch_lut = [&]() {
        lut_compute_kernel<<<blocks, LUT_THREADS>>>(
            device_lut6, device_lut15, grid, device_compute);
    };
    auto launch_chain = [&]() {
        chain_compute_kernel<<<blocks, LUT_THREADS>>>(
            device_perm6, device_perm15, 0x37u, 0x43u,
            grid, device_compute);
    };

    launch_build();
    if (check_cuda(cudaGetLastError(), "build warmup")) {
        cleanup(); return -1;
    }
    if (check_cuda(cudaDeviceSynchronize(), "build warmup sync")) {
        cleanup(); return -1;
    }

    launch_extract();
    launch_lut();
    launch_chain();
    if (check_cuda(cudaGetLastError(), "warmup launch")) {
        cleanup(); return -1;
    }
    if (check_cuda(cudaDeviceSynchronize(), "warmup sync")) {
        cleanup(); return -1;
    }

    double build_ms = 0.0;
    double extract_ms = 0.0;
    double lut_ms = 0.0;
    double chain_ms = 0.0;
    if (time_launch(launch_build, repeats, &build_ms)
        || time_launch(launch_extract, repeats, &extract_ms)
        || time_launch(launch_lut, repeats, &lut_ms)
        || time_launch(launch_chain, repeats, &chain_ms)) {
        cleanup(); return -1;
    }

    double cells = (double)blocks * (double)grid * (double)grid;
    output[0] = build_ms;
    output[1] = extract_ms;
    output[2] = lut_ms;
    output[3] = chain_ms;
    output[4] = cells / (lut_ms * 1.0e-3);
    output[5] = cells / (chain_ms * 1.0e-3);
    output[6] = chain_ms / lut_ms;
    output[7] = (2.0 * (double)LUT_ENTRIES) / (build_ms * 1.0e-3);

    cleanup();
    return 0;
}

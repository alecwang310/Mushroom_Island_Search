/*
 * Minimal test: 1 seed, 1 thread, 1 grid cell.
 * Checks that perlin init + GPU sampling doesn't crash.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define PERM_SIZE 257
__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ float perlin_sample(
    const uint8_t *perm, float oa, float oc,
    uint8_t h2, float d2, float t2, float x, float z)
{
    float d1 = x + oa, d3 = z + oc;
    float i1 = floorf(d1), i3 = floorf(d3);
    d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF, h3 = ((int)i3) & 0xFF;
    float t1 = d1*d1*d1 * (d1*(d1*6.0f-15.0f)+10.0f);
    float t3 = d3*d3*d3 * (d3*(d3*6.0f-15.0f)+10.0f);

    int va = perm[h1]+h2, vb = perm[h1+1]+h2;
    int v2a=perm[va&0xFF]+h3, v2b=perm[(va&0xFF)+1]+h3;
    int v3a=perm[vb&0xFF]+h3, v3b=perm[(vb&0xFF)+1]+h3;
    int v4a=perm[v2a&0xFF], v4b=perm[(v2a&0xFF)+1];
    int v5a=perm[v2b&0xFF], v5b=perm[(v2b&0xFF)+1];
    int v6a=perm[v3a&0xFF], v6b=perm[(v3a&0xFF)+1];
    int v7a=perm[v3b&0xFF], v7b=perm[(v3b&0xFF)+1];

    #define L(i,a,b,c) (c_grad[(i)&0xF][0]*(a)+c_grad[(i)&0xF][1]*(b)+c_grad[(i)&0xF][2]*(c))
    float l1=L(v4a,d1,d2,d3);   float l5=L(v4b,d1,d2,d3-1);
    float l2=L(v6a,d1-1,d2,d3); float l6=L(v6b,d1-1,d2,d3-1);
    float l3=L(v5a,d1,d2-1,d3); float l7=L(v5b,d1,d2-1,d3-1);
    float l4=L(v7a,d1-1,d2-1,d3); float l8=L(v7b,d1-1,d2-1,d3-1);
    #undef L

    l1+=t1*(l2-l1); l3+=t1*(l4-l3); l5+=t1*(l6-l5); l7+=t1*(l8-l7);
    l1+=t2*(l3-l1); l5+=t2*(l7-l5);
    return l1+t3*(l5-l1);
}

// Minimal kernel: 1 thread, just test that perlin_sample doesn't crash
__global__ void test_perlin(
    const uint8_t *perm, float oa, float oc,
    uint8_t h2, float d2, float t2,
    float x, float z, float *out)
{
    *out = perlin_sample(perm, oa, oc, h2, d2, t2, x, z);
}

// ---- Perlin init (matching C engine) ----
typedef struct { uint64_t lo, hi; } Xoroshiro;
static uint64_t rotl64(uint64_t x, int k) { return ((x<<k)|(x>>(64-k))); }
static void xSetSeed(Xoroshiro *xr, uint64_t v) {
    const uint64_t XL=0x9e3779b97f4a7c15ULL, XH=0x6a09e667f3bcc909ULL;
    const uint64_t A=0xbf58476d1ce4e5b9ULL, B=0x94d049bb133111ebULL;
    uint64_t l=v^XH, h=l+XL;
    l=(l^(l>>30))*A; h=(h^(h>>30))*A;
    l=(l^(l>>27))*B; h=(h^(h>>27))*B;
    l^=(l>>31); h^=(h>>31);
    xr->lo=l; xr->hi=h;
}
static uint64_t xNextLong(Xoroshiro *xr) {
    uint64_t l=xr->lo, h=xr->hi, n=rotl64(l+h,17)+l;
    h^=l; xr->lo=rotl64(l,49)^h^(h<<21); xr->hi=rotl64(h,28);
    return n;
}
static double xNextDouble(Xoroshiro *xr) { return (xNextLong(xr)>>11)*(1.0/(1ULL<<53)); }
static int xNextInt(Xoroshiro *xr, uint32_t n) {
    uint64_t r=(xNextLong(xr)&0xFFFFFFFF)*n;
    if((uint32_t)r<n){uint32_t t=(~n+1)%n; while((uint32_t)r<t) r=(xNextLong(xr)&0xFFFFFFFF)*n;}
    return (int)(r>>32);
}

int main() {
    uint64_t seed = 0;
    printf("=== GPU Perlin Verification Test ===\n");
    printf("Seed: %llu\n", (unsigned long long)seed);

    // Step 1: Create one PerlinNoise exactly like C engine does
    Xoroshiro xr;
    xSetSeed(&xr, seed);
    uint64_t xlo = xNextLong(&xr), xhi = xNextLong(&xr);
    printf("xlo=%016llx xhi=%016llx\n", (unsigned long long)xlo, (unsigned long long)xhi);

    // Init SHIFT → get the first octave of shift_octA
    Xoroshiro pxr = {xlo ^ 0x080518cf6af25384ULL, xhi ^ 0x3f3dfb40a54febd5ULL};
    uint64_t axlo = xNextLong(&pxr), axhi = xNextLong(&pxr);
    printf("shift internal: axlo=%016llx axhi=%016llx\n", (unsigned long long)axlo, (unsigned long long)axhi);

    // Octave -3 (index 9 in md5_oct)
    uint64_t md5_octave[13][2] = {
        {0xb198de63a8012672ULL,0x7b84cad43ef7b5a8ULL},{0x0fd787bfbc403ec3ULL,0x74a4a31ca21b48b8ULL},
        {0x36d326eed40efeb2ULL,0x5be9ce18223c636aULL},{0x082fe255f8be6631ULL,0x4e96119e22dedc81ULL},
        {0x0ef68ec68504005eULL,0x48b6bf93a2789640ULL},{0xf11268128982754fULL,0x257a1d670430b0aaULL},
        {0xe51c98ce7d1de664ULL,0x5f9478a733040c45ULL},{0x6d7b49e7e429850aULL,0x2e3063c622a24777ULL},
        {0xbd90d5377ba1b762ULL,0xc07317d419a7548dULL},{0x53d39c6752dac858ULL,0xbcd1c5a80ab65b3eULL},
        {0xb4a24d7a84e7677bULL,0x023ff9668e89b5c4ULL},{0xdffa22b534c5f608ULL,0xb9b67517d3665ca9ULL},
        {0xd50708086cef4d7cULL,0x6e1651ecc7f43309ULL},
    };
    uint64_t sub_lo = axlo ^ md5_octave[9][0];
    uint64_t sub_hi = axhi ^ md5_octave[9][1];
    printf("sub_lo=%016llx sub_hi=%016llx\n", (unsigned long long)sub_lo, (unsigned long long)sub_hi);

    // Create PerlinNoise from sub_lo/sub_hi
    Xoroshiro pnxr = {sub_lo, sub_hi};
    float oa = (float)(xNextDouble(&pnxr) * 256.0);
    float ob = (float)(xNextDouble(&pnxr) * 256.0);
    float oc = (float)(xNextDouble(&pnxr) * 256.0);

    uint8_t perm[PERM_SIZE];
    for (int i=0; i<256; i++) perm[i] = (uint8_t)i;
    for (int i=0; i<256; i++) {
        int j = xNextInt(&pnxr, 256-i) + i;
        uint8_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    perm[256] = perm[0];

    double ib = floor(ob), db = ob - ib;
    uint8_t h2 = (uint8_t)((int)ib & 0xFF);
    float d2 = (float)db;
    float t2 = (float)(db*db*db * (db*(db*6.0-15.0)+10.0));

    printf("Perlin: oa=%.4f ob=%.4f oc=%.4f h2=%d d2=%.4f t2=%.4f\n", oa, ob, oc, h2, d2, t2);
    printf("First 10 perm: ");
    for (int i=0; i<10; i++) printf("%d ", perm[i]);
    printf("\n");

    // Step 2: Upload to GPU and run 1-thread kernel
    uint8_t *d_perm;
    float *d_out;
    CHECK(cudaMalloc(&d_perm, PERM_SIZE));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_perm, perm, PERM_SIZE, cudaMemcpyHostToDevice));

    test_perlin<<<1, 1>>>(
        d_perm, oa, oc, h2, d2, t2,
        0.0f, 0.0f,  // sample at origin
        d_out);

    CHECK(cudaDeviceSynchronize());
    printf("Kernel completed successfully!\n");

    float result;
    CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("GPU perlin_sample(0, 0) = %.10f\n", result);

    // Step 3: Compare with CPU
    // Use the same perlin function on CPU
    float cpu_result;
    {
        float x=0.0f, z=0.0f;
        float d1=x+oa, d3=z+oc;
        float i1=floorf(d1), i3=floorf(d3);
        d1-=i1; d3-=i3;
        int h1=((int)i1)&0xFF, h3=((int)i3)&0xFF;
        float t1=d1*d1*d1*(d1*(d1*6.0f-15.0f)+10.0f);
        float t3=d3*d3*d3*(d3*(d3*6.0f-15.0f)+10.0f);
        int va=perm[h1]+h2, vb=perm[h1+1]+h2;
        int v2a=perm[va&0xFF]+h3, v2b=perm[(va&0xFF)+1]+h3;
        int v3a=perm[vb&0xFF]+h3, v3b=perm[(vb&0xFF)+1]+h3;
        int v4a=perm[v2a&0xFF], v4b=perm[(v2a&0xFF)+1];
        int v5a=perm[v2b&0xFF], v5b=perm[(v2b&0xFF)+1];
        int v6a=perm[v3a&0xFF], v6b=perm[(v3a&0xFF)+1];
        int v7a=perm[v3b&0xFF], v7b=perm[(v3b&0xFF)+1];
        float g[16][3]={{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},{1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}};
        #define L(i,a,b,c) (g[(i)&0xF][0]*(a)+g[(i)&0xF][1]*(b)+g[(i)&0xF][2]*(c))
        float l1=L(v4a,d1,d2,d3); float l5=L(v4b,d1,d2,d3-1);
        float l2=L(v6a,d1-1,d2,d3); float l6=L(v6b,d1-1,d2,d3-1);
        float l3=L(v5a,d1,d2-1,d3); float l7=L(v5b,d1,d2-1,d3-1);
        float l4=L(v7a,d1-1,d2-1,d3); float l8=L(v7b,d1-1,d2-1,d3-1);
        #undef L
        l1+=t1*(l2-l1); l3+=t1*(l4-l3); l5+=t1*(l6-l5); l7+=t1*(l8-l7);
        l1+=t2*(l3-l1); l5+=t2*(l7-l5);
        cpu_result = l1+t3*(l5-l1);
    }
    printf("CPU perlin_sample(0, 0) = %.10f\n", cpu_result);
    printf("Match: %s (diff = %.6e)\n",
           fabsf(result - cpu_result) < 1e-6 ? "YES" : "NO",
           fabsf(result - cpu_result));

    // Step 4: Compare against C engine via Python
    // (the user can run this separately)
    printf("\n=== To verify against C engine ===\n");
    printf("Run: python -c \"from engine import ContEngine; e=ContEngine(0); print(e.sample(0,0))\"\n");

    CHECK(cudaFree(d_perm));
    CHECK(cudaFree(d_out));
    return 0;
}

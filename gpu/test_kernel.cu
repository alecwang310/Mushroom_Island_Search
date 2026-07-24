/*
 * test_kernel.cu — Minimal test of sparse_scan kernel.
 * Compile: nvcc -arch=sm_120 -o test_kernel.exe test_kernel.cu -lcudart
 * Run: test_kernel.exe
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Copy all the kernel code from sparse_kernel.cu
#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define THREADS 256
#define MAX_GRIDS 16

__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ __forceinline__ float perlin(
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

extern "C" __global__ void sparse_scan(
    const uint8_t *perm, const float *oa, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, int num_seeds,
    const int *grid_offsets, int num_grids, int G, int step, int K,
    int *hit_flags, int *hit_gx, int *hit_gz, int *hit_grid)
{
    int seed = blockIdx.x, tid = threadIdx.x;
    float fr = 337.0f/331.0f;

    __shared__ uint8_t s_perm[MAX_OCTAVES][PERM_SIZE];
    __shared__ float   s_oa[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float   s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float   s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    int s_ranges[8]; float s_dbl[2];

    int oct=0, b=tid;
    while (oct<MAX_OCTAVES && b<PERM_SIZE) {
        int src=(seed*MAX_OCTAVES+oct)*PERM_SIZE+b;
        s_perm[oct][b]=perm[src];
        oct+=(b+THREADS)/PERM_SIZE; b=(b+THREADS)%PERM_SIZE;
    }
    if (tid<MAX_OCTAVES) {
        int src=seed*MAX_OCTAVES+tid;
        s_oa[tid]=oa[src]; s_oc[tid]=oc[src];
        s_amp[tid]=amp[src]; s_lac[tid]=lac[src];
        s_h2[tid]=h2[src]; s_d2[tid]=d2[src]; s_t2[tid]=t2[src];
    }
    if (tid<8) s_ranges[tid]=ranges[seed*8+tid];
    if (tid<2) s_dbl[tid]=dbl_amps[seed*2+tid];
    __syncthreads();

    int sh_as=s_ranges[0],sh_ac=s_ranges[1],sh_bs=s_ranges[2],sh_bc=s_ranges[3];
    int ct_as=s_ranges[4],ct_ac=s_ranges[5],ct_bs=s_ranges[6],ct_bc=s_ranges[7];
    float sh_amp=s_dbl[0], ct_amp=s_dbl[1];
    __shared__ float s_grid[MAX_GRIDS * MAX_GRIDS];
    __shared__ int s_found;

    for (int gid = 0; gid < num_grids; gid++) {
        if (tid == 0) s_found = 0;
        __syncthreads();
        int ox = grid_offsets[gid*2], oz = grid_offsets[gid*2+1];
        int total = G * G;

        // Sample grid
        for (int idx = tid; idx < total; idx += THREADS) {
            int cx = idx % G, cz = idx / G;
            float x = (float)(ox + cx * step);
            float z = (float)(oz + cz * step);
            float dx = 0, dz = 0;
            for (int i=sh_as; i<sh_as+sh_ac; i++) {
                float lf=s_lac[i];
                dx += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],x*lf,z*lf);
                dz += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],z*lf,x*lf);
            }
            for (int i=sh_bs; i<sh_bs+sh_bc; i++) {
                float lf=s_lac[i];
                dx += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],x*lf*fr,z*lf*fr);
                dz += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],z*lf*fr,x*lf*fr);
            }
            dx *= sh_amp; dz *= sh_amp;
            float px = x + dx*4.0f, pz = z + dz*4.0f;
            float cont = 0;
            for (int i=ct_as; i<ct_as+ct_ac; i++) {
                float lf=s_lac[i];
                cont += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf,pz*lf);
            }
            for (int i=ct_bs; i<ct_bs+ct_bc; i++) {
                float lf=s_lac[i];
                cont += s_amp[i]*perlin(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf*fr,pz*lf*fr);
            }
            s_grid[idx] = cont * ct_amp;
        }
        __syncthreads();

        // K×K detection
        int bpa = G - K + 1;
        int total_blocks = bpa * bpa;
        for (int idx = tid; idx < total_blocks; idx += THREADS) {
            int bx = idx % bpa, bz = idx / bpa;
            int all_mushroom = 1;
            for (int dz2=0; dz2<K && all_mushroom; dz2++)
                for (int dx2=0; dx2<K; dx2++)
                    if (s_grid[(bz+dz2)*G + (bx+dx2)] >= -1.05f)
                        { all_mushroom=0; break; }
            if (all_mushroom) {
                hit_flags[seed] = 1;
                hit_gx[seed] = ox + (bx + K/2) * step;
                hit_gz[seed] = oz + (bz + K/2) * step;
                hit_grid[seed] = gid;
            }
        }
        __syncthreads();
        if (tid == 0 && hit_flags[seed]) break;
    }
}

// ---- Self-contained Xoroshiro + Perlin init (minimal for test) ----
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

static void perlin_init(uint8_t perm[257], float *a, float *b, float *c,
                        uint8_t *h2, float *d2, float *t2, uint64_t lo, uint64_t hi) {
    Xoroshiro xr={lo,hi};
    *a=(float)(xNextDouble(&xr)*256.0); *b=(float)(xNextDouble(&xr)*256.0); *c=(float)(xNextDouble(&xr)*256.0);
    for(int i=0;i<256;i++) perm[i]=(uint8_t)i;
    for(int i=0;i<256;i++){int j=xNextInt(&xr,256-i)+i; uint8_t t=perm[i]; perm[i]=perm[j]; perm[j]=t;}
    perm[256]=perm[0];
    double ib=floor(*b), db=*b-ib;
    *h2=(uint8_t)((int)ib&0xFF); *d2=(float)db;
    *t2=(float)(db*db*db*(db*(db*6.0-15.0)+10.0));
}

// MD5 octave hashes
static const uint64_t md5_oct[13][2]={
    {0xb198de63a8012672ULL,0x7b84cad43ef7b5a8ULL},{0x0fd787bfbc403ec3ULL,0x74a4a31ca21b48b8ULL},
    {0x36d326eed40efeb2ULL,0x5be9ce18223c636aULL},{0x082fe255f8be6631ULL,0x4e96119e22dedc81ULL},
    {0x0ef68ec68504005eULL,0x48b6bf93a2789640ULL},{0xf11268128982754fULL,0x257a1d670430b0aaULL},
    {0xe51c98ce7d1de664ULL,0x5f9478a733040c45ULL},{0x6d7b49e7e429850aULL,0x2e3063c622a24777ULL},
    {0xbd90d5377ba1b762ULL,0xc07317d419a7548dULL},{0x53d39c6752dac858ULL,0xbcd1c5a80ab65b3eULL},
    {0xb4a24d7a84e7677bULL,0x023ff9668e89b5c4ULL},{0xdffa22b534c5f608ULL,0xb9b67517d3665ca9ULL},
    {0xd50708086cef4d7cULL,0x6e1651ecc7f43309ULL},
};
static const double lac_ini[]={1.0,0.5,0.25,1./8,1./16,1./32,1./64,1./128,1./256,1./512,1./1024,1./2048,1./4096};
static const double per_ini[]={0,1,2./3,4./7,8./15,16./31,32./63,64./127,128./255,256./511};
static const double amp_ini[]={0,5./6,10./9,15./12,20./15,25./18,30./21,35./24,40./27,45./30};

int main() {
    // Init one seed (seed=12345, non-large)
    uint64_t seed = 12345;
    Xoroshiro xr;
    xSetSeed(&xr, seed);
    uint64_t xlo = xNextLong(&xr), xhi = xNextLong(&xr);

    // Allocate host data
    uint8_t (*h_perm)[PERM_SIZE] = (uint8_t(*)[PERM_SIZE])calloc(MAX_OCTAVES*PERM_SIZE,1);
    float *h_oa=(float*)calloc(MAX_OCTAVES,sizeof(float));
    float *h_oc=(float*)calloc(MAX_OCTAVES,sizeof(float));
    float *h_amp=(float*)calloc(MAX_OCTAVES,sizeof(float));
    float *h_lac=(float*)calloc(MAX_OCTAVES,sizeof(float));
    uint8_t *h_h2=(uint8_t*)calloc(MAX_OCTAVES,1);
    float *h_d2=(float*)calloc(MAX_OCTAVES,sizeof(float));
    float *h_t2=(float*)calloc(MAX_OCTAVES,sizeof(float));
    int h_ranges[8]={0};
    float h_dbl[2]={0};
    int n_perlin=0;

    // Init SHIFT
    {
        Xoroshiro pxr = {xlo^0x080518cf6af25384ULL, xhi^0x3f3dfb40a54febd5ULL};
        double amp[]={1,1,1,0};
        uint64_t axlo=xNextLong(&pxr), axhi=xNextLong(&pxr);
        double lac=lac_ini[3], per=per_ini[4]; // omin=-3
        int octA_s=n_perlin, octA_c=0;
        for(int i=0;i<4;i++){
            lac*=2;per*=0.5;
            if(amp[i]==0)continue;
            perlin_init(h_perm[n_perlin],&h_oa[n_perlin],&h_oc[n_perlin],&h_oc[n_perlin],&h_h2[n_perlin],&h_d2[n_perlin],&h_t2[n_perlin], axlo^md5_oct[12-3+i][0], axhi^md5_oct[12-3+i][1]);
            h_amp[n_perlin]=(float)(amp[i]*per); h_lac[n_perlin]=(float)lac;
            n_perlin++; octA_c++;
        }
        uint64_t bxlo=xNextLong(&pxr), bxhi=xNextLong(&pxr);
        lac=lac_ini[3]; per=per_ini[4];
        int octB_s=n_perlin, octB_c=0;
        for(int i=0;i<4;i++){
            lac*=2;per*=0.5;
            if(amp[i]==0)continue;
            perlin_init(h_perm[n_perlin],&h_oa[n_perlin],&h_oc[n_perlin],&h_oc[n_perlin],&h_h2[n_perlin],&h_d2[n_perlin],&h_t2[n_perlin], bxlo^md5_oct[12-3+i][0], bxhi^md5_oct[12-3+i][1]);
            h_amp[n_perlin]=(float)(amp[i]*per); h_lac[n_perlin]=(float)lac;
            n_perlin++; octB_c++;
        }
        int eff=3; while(eff>0&&amp[eff-1]==0)eff--;
        h_ranges[0]=octA_s; h_ranges[1]=octA_c; h_ranges[2]=octB_s; h_ranges[3]=octB_c;
        h_dbl[0]=(float)amp_ini[eff];
    }

    // Init CONTINENTALNESS
    {
        Xoroshiro pxr = {xlo^0x83886c9d0ae3a662ULL, xhi^0xafa638a61b42e8adULL};
        double amp[]={1,1,2,2,2,1,1,1,1};
        uint64_t axlo=xNextLong(&pxr), axhi=xNextLong(&pxr);
        double lac=lac_ini[9], per=per_ini[9]; // omin=-9
        int octA_s=n_perlin, octA_c=0;
        for(int i=0;i<9;i++){
            lac*=2;per*=0.5;
            if(amp[i]==0)continue;
            perlin_init(h_perm[n_perlin],&h_oa[n_perlin],&h_oc[n_perlin],&h_oc[n_perlin],&h_h2[n_perlin],&h_d2[n_perlin],&h_t2[n_perlin], axlo^md5_oct[12-9+i][0], axhi^md5_oct[12-9+i][1]);
            h_amp[n_perlin]=(float)(amp[i]*per); h_lac[n_perlin]=(float)lac;
            n_perlin++; octA_c++;
        }
        uint64_t bxlo=xNextLong(&pxr), bxhi=xNextLong(&pxr);
        lac=lac_ini[9]; per=per_ini[9];
        int octB_s=n_perlin, octB_c=0;
        for(int i=0;i<9;i++){
            lac*=2;per*=0.5;
            if(amp[i]==0)continue;
            perlin_init(h_perm[n_perlin],&h_oa[n_perlin],&h_oc[n_perlin],&h_oc[n_perlin],&h_h2[n_perlin],&h_d2[n_perlin],&h_t2[n_perlin], bxlo^md5_oct[12-9+i][0], bxhi^md5_oct[12-9+i][1]);
            h_amp[n_perlin]=(float)(amp[i]*per); h_lac[n_perlin]=(float)lac;
            n_perlin++; octB_c++;
        }
        int eff=9;
        h_ranges[4]=octA_s; h_ranges[5]=octA_c; h_ranges[6]=octB_s; h_ranges[7]=octB_c;
        h_dbl[1]=(float)amp_ini[eff];
    }
    printf("Init complete: %d perlin instances (expected 24)\n", n_perlin);
    printf("Ranges: shift A[%d,%d] B[%d,%d] cont A[%d,%d] B[%d,%d]\n",
           h_ranges[0],h_ranges[1],h_ranges[2],h_ranges[3],
           h_ranges[4],h_ranges[5],h_ranges[6],h_ranges[7]);

    // Allocate device memory
    uint8_t *d_perm; float *d_oa, *d_oc, *d_amp, *d_lac; uint8_t *d_h2; float *d_d2, *d_t2;
    int *d_ranges; float *d_dbl;
    cudaMalloc(&d_perm, MAX_OCTAVES*PERM_SIZE);
    cudaMalloc(&d_oa, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_oc, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_amp, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_lac, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_h2, MAX_OCTAVES);
    cudaMalloc(&d_d2, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_t2, MAX_OCTAVES*sizeof(float));
    cudaMalloc(&d_ranges, 8*sizeof(int));
    cudaMalloc(&d_dbl, 2*sizeof(float));
    cudaMemcpy(d_perm, h_perm, MAX_OCTAVES*PERM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oa, h_oa, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oc, h_oc, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amp, h_amp, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lac, h_lac, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, h_h2, MAX_OCTAVES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2, h_d2, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t2, h_t2, MAX_OCTAVES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ranges, h_ranges, 8*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbl, h_dbl, 2*sizeof(float), cudaMemcpyHostToDevice);

    // Grid params
    int G=16, K=4, step=80, num_grids=8;
    int grid_offsets[16] = {0,0, 1280,0, 0,1280, 1280,1280,
                            2560,0, 0,2560, 2560,2560, 3840,0};
    int *d_offsets;
    cudaMalloc(&d_offsets, 16*sizeof(int));
    cudaMemcpy(d_offsets, grid_offsets, 16*sizeof(int), cudaMemcpyHostToDevice);

    // Output
    int h_hit=0, h_hx=0, h_hz=0, h_hg=0;
    int *d_hit, *d_hx, *d_hz, *d_hg;
    cudaMalloc(&d_hit, sizeof(int)); cudaMalloc(&d_hx, sizeof(int));
    cudaMalloc(&d_hz, sizeof(int)); cudaMalloc(&d_hg, sizeof(int));
    cudaMemset(d_hit, 0, sizeof(int));

    // Launch
    printf("Launching kernel (1 block, 256 threads)...\n");
    sparse_scan<<<1, THREADS>>>(
        d_perm, d_oa, d_oc, d_amp, d_lac, d_h2, d_d2, d_t2,
        d_ranges, d_dbl, 1,
        d_offsets, num_grids, G, step, K,
        d_hit, d_hx, d_hz, d_hg);

    cudaError_t err = cudaDeviceSynchronize();
    printf("Kernel sync: %s\n", cudaGetErrorString(err));

    cudaMemcpy(&h_hit, d_hit, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_hx, d_hx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_hz, d_hz, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: hit=%d x=%d z=%d\n", h_hit, h_hx, h_hz);

    // Compare with CPU continentalness at the hit location
    if (h_hit) {
        printf("Mushroom island candidate at (%d, %d)\n", h_hx, h_hz);
        printf("(CPU verification would require linking C engine)\n");
    } else {
        printf("No KxK mushroom block found for seed %llu\n", (unsigned long long)seed);
    }

    // Free
    cudaFree(d_perm); cudaFree(d_oa); cudaFree(d_oc); cudaFree(d_amp);
    cudaFree(d_lac); cudaFree(d_h2); cudaFree(d_d2); cudaFree(d_t2);
    cudaFree(d_ranges); cudaFree(d_dbl); cudaFree(d_offsets);
    cudaFree(d_hit); cudaFree(d_hx); cudaFree(d_hz); cudaFree(d_hg);
    free(h_perm); free(h_oa); free(h_oc); free(h_amp); free(h_lac);
    free(h_h2); free(h_d2); free(h_t2);
    return 0;
}

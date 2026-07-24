/*
 * test_value.cu — Sample continentalness on GPU, return the float value.
 * Compares against CPU C engine output.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define THREADS 256

__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ float perlin_(
    const uint8_t *perm, float oa, float oc,
    uint8_t h2, float d2, float t2, float x, float z)
{
    float d1 = x + oa, d3 = z + oc;
    float i1 = floorf(d1), i3 = floorf(d3);
    d1 -= i1; d3 -= i3;
    int h1 = ((int)i1) & 0xFF, h3 = ((int)i3) & 0xFF;
    float t1 = d1*d1*d1*(d1*(d1*6.0f-15.0f)+10.0f);
    float t3 = d3*d3*d3*(d3*(d3*6.0f-15.0f)+10.0f);

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

__global__ void sample_cont(
    const uint8_t *perm, const float *oa, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps,
    int x, int z, float *out)
{
    int tid = threadIdx.x;
    float fr = 337.0f/331.0f;

    __shared__ uint8_t s_perm[MAX_OCTAVES][PERM_SIZE];
    __shared__ float s_oa[MAX_OCTAVES], s_oc[MAX_OCTAVES];
    __shared__ float s_amp[MAX_OCTAVES], s_lac[MAX_OCTAVES];
    __shared__ uint8_t s_h2[MAX_OCTAVES];
    __shared__ float s_d2[MAX_OCTAVES], s_t2[MAX_OCTAVES];
    int s_ranges[8]; float s_dbl[2];

    int oct=0, b=tid;
    while (oct<MAX_OCTAVES && b<PERM_SIZE) {
        s_perm[oct][b]=perm[oct*PERM_SIZE+b];
        oct+=(b+THREADS)/PERM_SIZE; b=(b+THREADS)%PERM_SIZE;
    }
    if (tid<MAX_OCTAVES) {
        s_oa[tid]=oa[tid]; s_oc[tid]=oc[tid];
        s_amp[tid]=amp[tid]; s_lac[tid]=lac[tid];
        s_h2[tid]=h2[tid]; s_d2[tid]=d2[tid]; s_t2[tid]=t2[tid];
    }
    if (tid<8) s_ranges[tid]=ranges[tid];
    if (tid<2) s_dbl[tid]=dbl_amps[tid];
    __syncthreads();

    if (tid != 0) return;

    int sh_as=s_ranges[0],sh_ac=s_ranges[1],sh_bs=s_ranges[2],sh_bc=s_ranges[3];
    int ct_as=s_ranges[4],ct_ac=s_ranges[5],ct_bs=s_ranges[6],ct_bc=s_ranges[7];
    float sh_amp=s_dbl[0], ct_amp=s_dbl[1];

    float px=(float)x, pz=(float)z;
    // Shift
    float dx=0, dz=0;
    for (int i=sh_as; i<sh_as+sh_ac; i++) {
        float lf=s_lac[i];
        dx+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf,pz*lf);
        dz+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],pz*lf,px*lf);
    }
    for (int i=sh_bs; i<sh_bs+sh_bc; i++) {
        float lf=s_lac[i];
        dx+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf*fr,pz*lf*fr);
        dz+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],pz*lf*fr,px*lf*fr);
    }
    dx*=sh_amp; dz*=sh_amp;
    px+=dx*4.0f; pz+=dz*4.0f;

    // Cont
    float cont=0;
    for (int i=ct_as; i<ct_as+ct_ac; i++) {
        float lf=s_lac[i];
        cont+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf,pz*lf);
    }
    for (int i=ct_bs; i<ct_bs+ct_bc; i++) {
        float lf=s_lac[i];
        cont+=s_amp[i]*perlin_(s_perm[i],s_oa[i],s_oc[i],s_h2[i],s_d2[i],s_t2[i],px*lf*fr,pz*lf*fr);
    }
    *out = cont * ct_amp;
}

// ---- Test harness ----
typedef struct { uint64_t lo, hi; } Xoroshiro;
static uint64_t rotl64(uint64_t x, int k) { return ((x<<k)|(x>>(64-k))); }
static void xSetSeed(Xoroshiro *xr, uint64_t v) {
    const uint64_t XL=0x9e3779b97f4a7c15ULL,XH=0x6a09e667f3bcc909ULL,A=0xbf58476d1ce4e5b9ULL,B=0x94d049bb133111ebULL;
    uint64_t l=v^XH,h=l+XL;l=(l^(l>>30))*A;h=(h^(h>>30))*A;l=(l^(l>>27))*B;h=(h^(h>>27))*B;l^=(l>>31);h^=(h>>31);xr->lo=l;xr->hi=h;
}
static uint64_t xNextLong(Xoroshiro *xr){uint64_t l=xr->lo,h=xr->hi,n=rotl64(l+h,17)+l;h^=l;xr->lo=rotl64(l,49)^h^(h<<21);xr->hi=rotl64(h,28);return n;}
static double xNextDouble(Xoroshiro *xr){return(xNextLong(xr)>>11)*(1.0/(1ULL<<53));}
static int xNextInt(Xoroshiro *xr, uint32_t n){uint64_t r=(xNextLong(xr)&0xFFFFFFFF)*n;if((uint32_t)r<n){uint32_t t=(~n+1)%n;while((uint32_t)r<t)r=(xNextLong(xr)&0xFFFFFFFF)*n;}return(int)(r>>32);}
static void perlin_init(uint8_t perm[257],float*a,float*b,float*c,uint8_t*h2,float*d2,float*t2,uint64_t lo,uint64_t hi){
    Xoroshiro xr={lo,hi};*a=(float)(xNextDouble(&xr)*256);*b=(float)(xNextDouble(&xr)*256);*c=(float)(xNextDouble(&xr)*256);
    for(int i=0;i<256;i++)perm[i]=(uint8_t)i;
    for(int i=0;i<256;i++){int j=xNextInt(&xr,256-i)+i;uint8_t t=perm[i];perm[i]=perm[j];perm[j]=t;}perm[256]=perm[0];
    double ib=floor(*b),db=*b-ib;*h2=(uint8_t)((int)ib&0xFF);*d2=(float)db;*t2=(float)(db*db*db*(db*(db*6.0-15.0)+10.0));
}
static const uint64_t md5_oct[13][2]={
    {0xb198de63a8012672ULL,0x7b84cad43ef7b5a8ULL},{0x0fd787bfbc403ec3ULL,0x74a4a31ca21b48b8ULL},{0x36d326eed40efeb2ULL,0x5be9ce18223c636aULL},{0x082fe255f8be6631ULL,0x4e96119e22dedc81ULL},{0x0ef68ec68504005eULL,0x48b6bf93a2789640ULL},{0xf11268128982754fULL,0x257a1d670430b0aaULL},{0xe51c98ce7d1de664ULL,0x5f9478a733040c45ULL},{0x6d7b49e7e429850aULL,0x2e3063c622a24777ULL},{0xbd90d5377ba1b762ULL,0xc07317d419a7548dULL},{0x53d39c6752dac858ULL,0xbcd1c5a80ab65b3eULL},{0xb4a24d7a84e7677bULL,0x023ff9668e89b5c4ULL},{0xdffa22b534c5f608ULL,0xb9b67517d3665ca9ULL},{0xd50708086cef4d7cULL,0x6e1651ecc7f43309ULL}};
static const double lac_ini[]={1.0,0.5,0.25,1./8,1./16,1./32,1./64,1./128,1./256,1./512,1./1024,1./2048,1./4096};
static const double per_ini[]={0,1,2./3,4./7,8./15,16./31,32./63,64./127,128./255,256./511};
static const double amp_ini[]={0,5./6,10./9,15./12,20./15,25./18,30./21,35./24,40./27,45./30};

int main(){
    uint64_t seed=0;
    Xoroshiro xr;xSetSeed(&xr,seed);
    uint64_t xlo=xNextLong(&xr),xhi=xNextLong(&xr);
    uint8_t h_perm[MAX_OCTAVES][PERM_SIZE];float h_oa[MAX_OCTAVES],h_ob[MAX_OCTAVES],h_oc[MAX_OCTAVES];
    float h_amp[MAX_OCTAVES],h_lac[MAX_OCTAVES];uint8_t h_h2[MAX_OCTAVES];float h_d2[MAX_OCTAVES],h_t2[MAX_OCTAVES];
    int h_ranges[8];float h_dbl[2];int n=0;

    // SHIFT
    {Xoroshiro pxr={xlo^0x080518cf6af25384ULL,xhi^0x3f3dfb40a54febd5ULL};
    double amp[]={1,1,1,0};uint64_t alo=xNextLong(&pxr),ahi=xNextLong(&pxr);
    double lac=lac_ini[3],per=per_ini[4];int as=n,ac=0;
    for(int i=0;i<4;i++){lac*=2;per*=0.5;if(amp[i]==0)continue;
    perlin_init(h_perm[n],&h_oa[n],&h_ob[n],&h_oc[n],&h_h2[n],&h_d2[n],&h_t2[n],alo^md5_oct[9+i][0],ahi^md5_oct[9+i][1]);
    h_amp[n]=(float)(amp[i]*per);h_lac[n]=(float)lac;n++;ac++;}
    uint64_t blo=xNextLong(&pxr),bhi=xNextLong(&pxr);lac=lac_ini[3];per=per_ini[4];int bs=n,bc=0;
    for(int i=0;i<4;i++){lac*=2;per*=0.5;if(amp[i]==0)continue;
    perlin_init(h_perm[n],&h_oa[n],&h_ob[n],&h_oc[n],&h_h2[n],&h_d2[n],&h_t2[n],blo^md5_oct[9+i][0],bhi^md5_oct[9+i][1]);
    h_amp[n]=(float)(amp[i]*per);h_lac[n]=(float)lac;n++;bc++;}
    h_ranges[0]=as;h_ranges[1]=ac;h_ranges[2]=bs;h_ranges[3]=bc;h_dbl[0]=(float)amp_ini[3];}

    // CONT
    {Xoroshiro pxr={xlo^0x83886c9d0ae3a662ULL,xhi^0xafa638a61b42e8adULL};
    double amp[]={1,1,2,2,2,1,1,1,1};uint64_t alo=xNextLong(&pxr),ahi=xNextLong(&pxr);
    double lac=lac_ini[9],per=per_ini[9];int as=n,ac=0;
    for(int i=0;i<9;i++){lac*=2;per*=0.5;if(amp[i]==0)continue;
    perlin_init(h_perm[n],&h_oa[n],&h_ob[n],&h_oc[n],&h_h2[n],&h_d2[n],&h_t2[n],alo^md5_oct[3+i][0],ahi^md5_oct[3+i][1]);
    h_amp[n]=(float)(amp[i]*per);h_lac[n]=(float)lac;n++;ac++;}
    uint64_t blo=xNextLong(&pxr),bhi=xNextLong(&pxr);lac=lac_ini[9];per=per_ini[9];int bs=n,bc=0;
    for(int i=0;i<9;i++){lac*=2;per*=0.5;if(amp[i]==0)continue;
    perlin_init(h_perm[n],&h_oa[n],&h_ob[n],&h_oc[n],&h_h2[n],&h_d2[n],&h_t2[n],blo^md5_oct[3+i][0],bhi^md5_oct[3+i][1]);
    h_amp[n]=(float)(amp[i]*per);h_lac[n]=(float)lac;n++;bc++;}
    h_ranges[4]=as;h_ranges[5]=ac;h_ranges[6]=bs;h_ranges[7]=bc;h_dbl[1]=(float)amp_ini[9];}

    printf("Init %d perlin instances\n",n);

    uint8_t *dp;float *doa,*doc,*da,*dl;uint8_t *dh;float *dd,*dt2;int *dr;float *ddb;float *dout;
    cudaMalloc(&dp,MAX_OCTAVES*PERM_SIZE);cudaMalloc(&doa,MAX_OCTAVES*sizeof(float));cudaMalloc(&doc,MAX_OCTAVES*sizeof(float));
    cudaMalloc(&da,MAX_OCTAVES*sizeof(float));cudaMalloc(&dl,MAX_OCTAVES*sizeof(float));cudaMalloc(&dh,MAX_OCTAVES);
    cudaMalloc(&dd,MAX_OCTAVES*sizeof(float));cudaMalloc(&dt2,MAX_OCTAVES*sizeof(float));
    cudaMalloc(&dr,8*sizeof(int));cudaMalloc(&ddb,2*sizeof(float));cudaMalloc(&dout,sizeof(float));
    cudaMemcpy(dp,h_perm,MAX_OCTAVES*PERM_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(doa,h_oa,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(doc,h_oc,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(da,h_amp,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dl,h_lac,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dh,h_h2,MAX_OCTAVES,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,h_d2,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dt2,h_t2,MAX_OCTAVES*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dr,h_ranges,8*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ddb,h_dbl,2*sizeof(float),cudaMemcpyHostToDevice);

    for (int tx=0;tx<5;tx++){
        for(int tz=0;tz<5;tz++){
            sample_cont<<<1,256>>>(dp,doa,doc,da,dl,dh,dd,dt2,dr,ddb,tx,tz,dout);
            float gpu_val;cudaMemcpy(&gpu_val,dout,sizeof(float),cudaMemcpyDeviceToHost);
            printf("GPU cont(%d,%d) = %.6f\n",tx,tz,gpu_val);
        }
    }

    // CPU comparison values (from C engine)
    printf("\nExpected from C engine:");
    printf("\n  cont(0,0) = -0.008172");
    printf("\n  cont(0,1) should be close to GPU\n");

    cudaFree(dp);cudaFree(doa);cudaFree(doc);cudaFree(da);cudaFree(dl);
    cudaFree(dh);cudaFree(dd);cudaFree(dt2);cudaFree(dr);cudaFree(ddb);cudaFree(dout);
    return 0;
}

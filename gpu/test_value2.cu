/* Minimal GPU continentalness test with printf debugging */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

__constant__ float c_grad[16][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, { 0,-1, 1}, {-1, 1, 0}, { 0,-1,-1},
};

__device__ float p(const uint8_t *perm, float oa, float oc,
    uint8_t h2, float d2, float t2, float x, float z)
{
    float d1=x+oa, d3=z+oc;
    float i1=floorf(d1), i3=floorf(d3);
    d1-=i1; d3-=i3;
    int h1=((int)i1)&0xFF, h3=((int)i3)&0xFF;
    float t1=d1*d1*d1*(d1*(d1*6-15)+10);
    float t3=d3*d3*d3*(d3*(d3*6-15)+10);
    int va=perm[h1]+h2, vb=perm[h1+1]+h2;
    int v2a=perm[va&0xFF]+h3; int v2b=perm[(va&0xFF)+1]+h3;
    int v3a=perm[vb&0xFF]+h3; int v3b=perm[(vb&0xFF)+1]+h3;
    int v4a=perm[v2a&0xFF];   int v4b=perm[(v2a&0xFF)+1];
    int v5a=perm[v2b&0xFF];   int v5b=perm[(v2b&0xFF)+1];
    int v6a=perm[v3a&0xFF];   int v6b=perm[(v3a&0xFF)+1];
    int v7a=perm[v3b&0xFF];   int v7b=perm[(v3b&0xFF)+1];
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

__global__ void test(const uint8_t *perm, const float *oa, const float *oc,
    const float *amp, const float *lac, const uint8_t *h2,
    const float *d2, const float *t2,
    const int *ranges, const float *dbl_amps, float *out)
{
    // Just thread 0, directly from global memory (no shared)
    if (threadIdx.x != 0) return;

    int sh_as=ranges[0],sh_ac=ranges[1],sh_bs=ranges[2],sh_bc=ranges[3];
    int ct_as=ranges[4],ct_ac=ranges[5],ct_bs=ranges[6],ct_bc=ranges[7];
    float sh_amp=dbl_amps[0], ct_amp=dbl_amps[1];
    float fr=337.0f/331.0f;

    printf("ranges: %d %d %d %d %d %d %d %d\n",sh_as,sh_ac,sh_bs,sh_bc,ct_as,ct_ac,ct_bs,ct_bc);
    printf("amps: %.4f %.4f\n",sh_amp,ct_amp);

    float px=0, pz=0;
    float dx=0, dz=0;
    for (int i=sh_as; i<sh_as+sh_ac; i++) {
        float lf=lac[i];
        printf("  shiftA[%d]: lac=%.6f amp=%.6f oa=%.4f oc=%.4f\n",i,lac[i],amp[i],oa[i],oc[i]);
        float v = p(&perm[i*257], oa[i], oc[i], h2[i], d2[i], t2[i], px*lf, pz*lf);
        printf("    perlin=%.6f\n",v);
        dx+=amp[i]*v;
        dz+=amp[i]*p(&perm[i*257], oa[i], oc[i], h2[i], d2[i], t2[i], pz*lf, px*lf);
    }
    dx*=sh_amp; dz*=sh_amp;
    px+=dx*4; pz+=dz*4;

    float cont=0;
    for (int i=ct_as; i<ct_as+ct_ac; i++) {
        float lf=lac[i];
        cont+=amp[i]*p(&perm[i*257], oa[i], oc[i], h2[i], d2[i], t2[i], px*lf, pz*lf);
    }
    *out = cont * ct_amp;
    printf("final cont=%.6f\n",*out);
}

// RNG code (same as before, omitted for brevity — copied from test_value.cu)
typedef struct { uint64_t lo, hi; } Xoroshiro;
static uint64_t rotl64(uint64_t x, int k) { return ((x<<k)|(x>>(64-k))); }
static void xSetSeed(Xoroshiro *xr, uint64_t v) {
    const uint64_t XL=0x9e3779b97f4a7c15ULL,XH=0x6a09e667f3bcc909ULL,A=0xbf58476d1ce4e5b9ULL,B=0x94d049bb133111ebULL;
    uint64_t l=v^XH,h=l+XL;l=(l^(l>>30))*A;h=(h^(h>>30))*A;l=(l^(l>>27))*B;h=(h^(h>>27))*B;l^=(l>>31);h^=(h>>31);xr->lo=l;xr->hi=h;
}
static uint64_t xNextLong(Xoroshiro *xr){uint64_t l=xr->lo,h=xr->hi,n=rotl64(l+h,17)+l;h^=l;xr->lo=rotl64(l,49)^h^(h<<21);xr->hi=rotl64(h,28);return n;}
static double xNextDouble(Xoroshiro *xr){return(xNextLong(xr)>>11)*(1.0/(1ULL<<53));}
static int xNextInt(Xoroshiro *xr, uint32_t n){uint64_t r=(xNextLong(xr)&0xFFFFFFFF)*n;if((uint32_t)r<n){uint32_t t=(~n+1)%n;while((uint32_t)r<t)r=(xNextLong(xr)&0xFFFFFFFF)*n;}return(int)(r>>32);}
static const uint64_t md5_oct[13][2]={
    {0xb198de63a8012672ULL,0x7b84cad43ef7b5a8ULL},{0x0fd787bfbc403ec3ULL,0x74a4a31ca21b48b8ULL},{0x36d326eed40efeb2ULL,0x5be9ce18223c636aULL},{0x082fe255f8be6631ULL,0x4e96119e22dedc81ULL},{0x0ef68ec68504005eULL,0x48b6bf93a2789640ULL},{0xf11268128982754fULL,0x257a1d670430b0aaULL},{0xe51c98ce7d1de664ULL,0x5f9478a733040c45ULL},{0x6d7b49e7e429850aULL,0x2e3063c622a24777ULL},{0xbd90d5377ba1b762ULL,0xc07317d419a7548dULL},{0x53d39c6752dac858ULL,0xbcd1c5a80ab65b3eULL},{0xb4a24d7a84e7677bULL,0x023ff9668e89b5c4ULL},{0xdffa22b534c5f608ULL,0xb9b67517d3665ca9ULL},{0xd50708086cef4d7cULL,0x6e1651ecc7f43309ULL}};
static const double lac_ini[]={1.0,0.5,0.25,1./8,1./16,1./32,1./64,1./128,1./256,1./512,1./1024,1./2048,1./4096};
static const double per_ini[]={0,1,2./3,4./7,8./15,16./31,32./63,64./127,128./255,256./511};
static const double amp_ini[]={0,5./6,10./9,15./12,20./15,25./18,30./21,35./24,40./27,45./30};

int main(){
    uint64_t seed=0;
    Xoroshiro xr;xSetSeed(&xr,seed);
    uint64_t xlo=xNextLong(&xr),xhi=xNextLong(&xr);
    printf("xlo=%016llx xhi=%016llx\n",(unsigned long long)xlo,(unsigned long long)xhi);

    #define MAX_OCT 24
    #define PSZ 257
    uint8_t perm[MAX_OCT*PSZ]; float oa[MAX_OCT],ob[MAX_OCT],oc[MAX_OCT];
    float amp[MAX_OCT],lac[MAX_OCT]; uint8_t h2[MAX_OCT]; float d2[MAX_OCT],t2[MAX_OCT];
    int ranges[8]; float dbl[2]; int n=0;

    // SHIFT init
    {Xoroshiro pxr={xlo^0x080518cf6af25384ULL,xhi^0x3f3dfb40a54febd5ULL};
    double amps[]={1,1,1,0}; uint64_t alo=xNextLong(&pxr),ahi=xNextLong(&pxr);
    double lc=lac_ini[3],pr=per_ini[4]; int as=n,ac=0;
    for(int i=0;i<4;i++){lc*=2;pr*=0.5;if(amps[i]==0)continue;
        uint64_t slo=alo^md5_oct[9+i][0],shi=ahi^md5_oct[9+i][1];
        Xoroshiro px={slo,shi};
        oa[n]=(float)(xNextDouble(&px)*256); ob[n]=(float)(xNextDouble(&px)*256); oc[n]=(float)(xNextDouble(&px)*256);
        for(int j=0;j<256;j++)perm[n*PSZ+j]=(uint8_t)j;
        for(int j=0;j<256;j++){int k=xNextInt(&px,256-j)+j;uint8_t t=perm[n*PSZ+j];perm[n*PSZ+j]=perm[n*PSZ+k];perm[n*PSZ+k]=t;}
        perm[n*PSZ+256]=perm[n*PSZ];
        double ib=floor(ob[n]),db=ob[n]-ib;h2[n]=(uint8_t)((int)ib&0xFF);d2[n]=(float)db;t2[n]=(float)(db*db*db*(db*(db*6-15)+10));
        amp[n]=(float)(amps[i]*pr);lac[n]=(float)lc;n++;ac++;}
    uint64_t blo=xNextLong(&pxr),bhi=xNextLong(&pxr);lc=lac_ini[3];pr=per_ini[4];int bs=n,bc=0;
    for(int i=0;i<4;i++){lc*=2;pr*=0.5;if(amps[i]==0)continue;
        uint64_t slo=blo^md5_oct[9+i][0],shi=bhi^md5_oct[9+i][1];
        Xoroshiro px={slo,shi};
        oa[n]=(float)(xNextDouble(&px)*256);ob[n]=(float)(xNextDouble(&px)*256);oc[n]=(float)(xNextDouble(&px)*256);
        for(int j=0;j<256;j++)perm[n*PSZ+j]=(uint8_t)j;
        for(int j=0;j<256;j++){int k=xNextInt(&px,256-j)+j;uint8_t t=perm[n*PSZ+j];perm[n*PSZ+j]=perm[n*PSZ+k];perm[n*PSZ+k]=t;}
        perm[n*PSZ+256]=perm[n*PSZ];
        double ib=floor(ob[n]),db=ob[n]-ib;h2[n]=(uint8_t)((int)ib&0xFF);d2[n]=(float)db;t2[n]=(float)(db*db*db*(db*(db*6-15)+10));
        amp[n]=(float)(amps[i]*pr);lac[n]=(float)lc;n++;bc++;}
    ranges[0]=as;ranges[1]=ac;ranges[2]=bs;ranges[3]=bc;dbl[0]=(float)amp_ini[3];}
    printf("After shift: n=%d\n",n);

    // CONT init
    {Xoroshiro pxr={xlo^0x83886c9d0ae3a662ULL,xhi^0xafa638a61b42e8adULL};
    double amps[]={1,1,2,2,2,1,1,1,1}; uint64_t alo=xNextLong(&pxr),ahi=xNextLong(&pxr);
    double lc=lac_ini[9],pr=per_ini[9]; int as=n,ac=0;
    for(int i=0;i<9;i++){lc*=2;pr*=0.5;if(amps[i]==0)continue;
        uint64_t slo=alo^md5_oct[3+i][0],shi=ahi^md5_oct[3+i][1];
        Xoroshiro px={slo,shi};
        oa[n]=(float)(xNextDouble(&px)*256);ob[n]=(float)(xNextDouble(&px)*256);oc[n]=(float)(xNextDouble(&px)*256);
        for(int j=0;j<256;j++)perm[n*PSZ+j]=(uint8_t)j;
        for(int j=0;j<256;j++){int k=xNextInt(&px,256-j)+j;uint8_t t=perm[n*PSZ+j];perm[n*PSZ+j]=perm[n*PSZ+k];perm[n*PSZ+k]=t;}
        perm[n*PSZ+256]=perm[n*PSZ];
        double ib=floor(ob[n]),db=ob[n]-ib;h2[n]=(uint8_t)((int)ib&0xFF);d2[n]=(float)db;t2[n]=(float)(db*db*db*(db*(db*6-15)+10));
        amp[n]=(float)(amps[i]*pr);lac[n]=(float)lc;n++;ac++;}
    uint64_t blo=xNextLong(&pxr),bhi=xNextLong(&pxr);lc=lac_ini[9];pr=per_ini[9];int bs=n,bc=0;
    for(int i=0;i<9;i++){lc*=2;pr*=0.5;if(amps[i]==0)continue;
        uint64_t slo=blo^md5_oct[3+i][0],shi=bhi^md5_oct[3+i][1];
        Xoroshiro px={slo,shi};
        oa[n]=(float)(xNextDouble(&px)*256);ob[n]=(float)(xNextDouble(&px)*256);oc[n]=(float)(xNextDouble(&px)*256);
        for(int j=0;j<256;j++)perm[n*PSZ+j]=(uint8_t)j;
        for(int j=0;j<256;j++){int k=xNextInt(&px,256-j)+j;uint8_t t=perm[n*PSZ+j];perm[n*PSZ+j]=perm[n*PSZ+k];perm[n*PSZ+k]=t;}
        perm[n*PSZ+256]=perm[n*PSZ];
        double ib=floor(ob[n]),db=ob[n]-ib;h2[n]=(uint8_t)((int)ib&0xFF);d2[n]=(float)db;t2[n]=(float)(db*db*db*(db*(db*6-15)+10));
        amp[n]=(float)(amps[i]*pr);lac[n]=(float)lc;n++;bc++;}
    ranges[4]=as;ranges[5]=ac;ranges[6]=bs;ranges[7]=bc;dbl[1]=(float)amp_ini[9];}
    printf("After cont: n=%d\n",n);
    printf("Ranges: %d %d %d %d %d %d %d %d\n",ranges[0],ranges[1],ranges[2],ranges[3],ranges[4],ranges[5],ranges[6],ranges[7]);

    // GPU
    uint8_t *dp;float *doa,*doc,*da,*dl;uint8_t *dh;float *dd,*dt;int *dr;float *ddb,*dout;
    cudaMalloc(&dp,MAX_OCT*PSZ);cudaMalloc(&doa,MAX_OCT*4);cudaMalloc(&doc,MAX_OCT*4);
    cudaMalloc(&da,MAX_OCT*4);cudaMalloc(&dl,MAX_OCT*4);cudaMalloc(&dh,MAX_OCT);
    cudaMalloc(&dd,MAX_OCT*4);cudaMalloc(&dt,MAX_OCT*4);
    cudaMalloc(&dr,8*4);cudaMalloc(&ddb,2*4);cudaMalloc(&dout,4);
    cudaMemcpy(dp,perm,MAX_OCT*PSZ,cudaMemcpyHostToDevice);
    cudaMemcpy(doa,oa,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(doc,oc,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(da,amp,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dl,lac,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dh,h2,MAX_OCT,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,d2,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dt,t2,MAX_OCT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dr,ranges,8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(ddb,dbl,2*4,cudaMemcpyHostToDevice);

    test<<<1,1>>>(dp,doa,doc,da,dl,dh,dd,dt,dr,ddb,dout);
    cudaDeviceSynchronize();
    float result;cudaMemcpy(&result,dout,4,cudaMemcpyDeviceToHost);
    printf("GPU result: %.6f\n",result);
    // CPU expected: -0.008172 for seed 0 at (0,0)

    return 0;
}

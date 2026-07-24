#include <cuda_runtime.h>
#include <stdint.h>
#define MAX_OCTAVES 24
#define PERM_SIZE 257
#define THREADS 256
__constant__ float c_grad[16][3]={{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},{1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}};

__device__ float perlin(const uint8_t *perm,float oa,float ob,float oc,uint8_t h2,float d2,float t2,float x,float y,float z){
    float d1=x+oa,d3=z+oc;
    if(y!=0.0f){float i2=floorf(y+ob);float dy=y+ob-i2;h2=(uint8_t)((int)i2&0xFF);d2=dy;t2=dy*dy*dy*(dy*(dy*6.0f-15.0f)+10.0f);}
    float i1=floorf(d1),i3=floorf(d3);d1-=i1;d3-=i3;
    int h1=((int)i1)&0xFF,h3=((int)i3)&0xFF;
    float t1=d1*d1*d1*(d1*(d1*6.0f-15.0f)+10.0f),t3=d3*d3*d3*(d3*(d3*6.0f-15.0f)+10.0f);
    int va=perm[h1]+h2,vb=perm[h1+1]+h2;
    int v2a=perm[va&0xFF]+h3,v2b=perm[(va&0xFF)+1]+h3,v3a=perm[vb&0xFF]+h3,v3b=perm[(vb&0xFF)+1]+h3;
    int v4a=perm[v2a&0xFF],v4b=perm[(v2a&0xFF)+1],v5a=perm[v2b&0xFF],v5b=perm[(v2b&0xFF)+1];
    int v6a=perm[v3a&0xFF],v6b=perm[(v3a&0xFF)+1],v7a=perm[v3b&0xFF],v7b=perm[(v3b&0xFF)+1];
    #define L(i,a,b,c) (c_grad[(i)&0xF][0]*(a)+c_grad[(i)&0xF][1]*(b)+c_grad[(i)&0xF][2]*(c))
    float l1=L(v4a,d1,d2,d3),l5=L(v4b,d1,d2,d3-1),l2=L(v6a,d1-1,d2,d3),l6=L(v6b,d1-1,d2,d3-1);
    float l3=L(v5a,d1,d2-1,d3),l7=L(v5b,d1,d2-1,d3-1),l4=L(v7a,d1-1,d2-1,d3),l8=L(v7b,d1-1,d2-1,d3-1);
    #undef L
    l1+=t1*(l2-l1);l3+=t1*(l4-l3);l5+=t1*(l6-l5);l7+=t1*(l8-l7);l1+=t2*(l3-l1);l5+=t2*(l7-l5);
    return l1+t3*(l5-l1);
}

extern "C" __global__ void sample_one(
    const uint8_t *perm,const float *oa,const float *ob,const float *oc,
    const float *amp,const float *lac,const uint8_t *h2,const float *d2,const float *t2,
    const int *ranges,const float *dbl_amps,
    int x,int z,float *out)
{
    if(threadIdx.x!=0)return;
    float fr=337.0f/331.0f;
    uint8_t p[MAX_OCTAVES][PERM_SIZE];float a[MAX_OCTAVES],b[MAX_OCTAVES],c[MAX_OCTAVES];
    float am[MAX_OCTAVES],l[MAX_OCTAVES];uint8_t h[MAX_OCTAVES];float dd[MAX_OCTAVES],tt[MAX_OCTAVES];
    for(int i=0;i<MAX_OCTAVES*PERM_SIZE;i++) ((uint8_t*)p)[i]=perm[i];
    for(int i=0;i<MAX_OCTAVES;i++){a[i]=oa[i];b[i]=ob[i];c[i]=oc[i];am[i]=amp[i];l[i]=lac[i];h[i]=h2[i];dd[i]=d2[i];tt[i]=t2[i];}
    int sh_as=ranges[0],sh_ac=ranges[1],sh_bs=ranges[2],sh_bc=ranges[3];
    int ct_as=ranges[4],ct_ac=ranges[5],ct_bs=ranges[6],ct_bc=ranges[7];
    float sh_amp=dbl_amps[0],ct_amp=dbl_amps[1];
    float dx=0,dz=0;
    for(int i=sh_as;i<sh_as+sh_ac;i++){float lf=l[i];dx+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],x*lf,0.0f,z*lf);dz+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],z*lf,x*lf,0.0f);}
    for(int i=sh_bs;i<sh_bs+sh_bc;i++){float lf=l[i];dx+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],x*lf*fr,0.0f,z*lf*fr);dz+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],z*lf*fr,x*lf*fr,0.0f);}
    dx*=sh_amp;dz*=sh_amp;float px=x+dx*4.0f,pz=z+dz*4.0f;
    float cont=0;
    for(int i=ct_as;i<ct_as+ct_ac;i++){float lf=l[i];cont+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],px*lf,0.0f,pz*lf);}
    for(int i=ct_bs;i<ct_bs+ct_bc;i++){float lf=l[i];cont+=am[i]*perlin(p[i],a[i],b[i],c[i],h[i],dd[i],tt[i],px*lf*fr,0.0f,pz*lf*fr);}
    *out=cont*ct_amp;
}

extern "C" __declspec(dllexport) int gpu_sample(int n,void *d_perm,void *d_oa,void *d_ob,void *d_oc,void *d_amp,void *d_lac,void *d_h2,void *d_d2,void *d_t2,void *d_ranges,void *d_dbl,int x,int z,void *d_out){
    sample_one<<<1,1>>>(d_perm,d_oa,d_ob,d_oc,d_amp,d_lac,d_h2,d_d2,d_t2,d_ranges,d_dbl,x,z,d_out);
    return cudaDeviceSynchronize();
}

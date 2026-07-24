"""hunt_1m.py — GPU hunt for 1M+ mushroom islands."""
import sys, os, time, ctypes, json, math
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as _eng

cuda = ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll')
cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; cuda.cudaMalloc.restype = ctypes.c_int
cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]; cuda.cudaMemcpy.restype = ctypes.c_int
cuda.cudaFree.argtypes = [ctypes.c_void_p]; cuda.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cuda.cudaDeviceSynchronize.restype = ctypes.c_int
H2D, D2H = 1, 2

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'sparse_kernel.dll'))
lib.gpu_scan_seeds.argtypes = [ctypes.c_void_p]*21; lib.gpu_scan_seeds.restype = ctypes.c_int

MAX_OCT, PERM, ENG_SZ = 24, 257, 8192

def extract(buf):
    off=0
    perm=np.frombuffer(buf,dtype=np.uint8,count=MAX_OCT*PERM,offset=off);off+=MAX_OCT*PERM
    oa=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    off+=MAX_OCT*8  # skip ob
    oc=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    amp=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    lac=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    h2=np.frombuffer(buf,dtype=np.uint8,count=MAX_OCT,offset=off);off+=MAX_OCT
    d2=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    t2=np.frombuffer(buf,dtype=np.float64,count=MAX_OCT,offset=off).astype(np.float32);off+=MAX_OCT*8
    rng=np.frombuffer(buf,dtype=np.int32,count=8,offset=off);off+=32
    dbl=np.frombuffer(buf,dtype=np.float64,count=2,offset=off).astype(np.float32)
    return perm.reshape(MAX_OCT,PERM),oa,oc,amp,lac,h2,d2,t2,rng,dbl

def m(arr): p=ctypes.c_void_p();cuda.cudaMalloc(ctypes.byref(p),arr.nbytes);cuda.cudaMemcpy(p,arr.ctypes.data,arr.nbytes,H2D);return p
def mz(sz): p=ctypes.c_void_p();cuda.cudaMalloc(ctypes.byref(p),sz);cuda.cudaMemset(p,0,sz);return p
def to_host(dp,arr): cuda.cudaMemcpy(arr.ctypes.data,dp,arr.nbytes,D2H)

def gpu_scan(seeds,step,K,G,offsets):
    n=len(seeds);ng=len(offsets)//2
    d = _eng.batch_init(seeds)
    perms=d['perm'];oas=d['oa'];obs=d['ob'];ocs=d['oc'];amps=d['amp'];lacs=d['lac']
    h2s=d['h2'];d2s=d['d2'];t2s=d['t2'];rngs=d['rng'];dbls=d['dbl']
    dp=m(perms.ravel());doa=m(oas.ravel());dob=m(obs.ravel());doc=m(ocs.ravel())
    da=m(amps.ravel());dl=m(lacs.ravel());dh=m(h2s.ravel());dd=m(d2s.ravel());dt=m(t2s.ravel())
    dr=m(rngs.ravel());ddb=m(dbls.ravel());doff=m(offsets)
    hf=np.zeros(n,dtype=np.int32);hx=np.zeros(n,dtype=np.int32);hz=np.zeros(n,dtype=np.int32);hg=np.zeros(n,dtype=np.int32)
    df=mz(n*4);dx=mz(n*4);dz=mz(n*4);dg=mz(n*4)
    lib.gpu_scan_seeds(dp,doa,dob,doc,da,dl,dh,dd,dt,dr,ddb,n,doff,ng,G,step,K,df,dx,dz,dg)
    cuda.cudaDeviceSynchronize()
    to_host(df,hf);to_host(dx,hx);to_host(dz,hz)
    for p in[dp,doa,dob,doc,da,dl,dh,dd,dt,dr,ddb,doff,df,dx,dz,dg]:cuda.cudaFree(p)
    return[(int(seeds[i]),int(hx[i]),int(hz[i]))for i in range(n)if hf[i]]

def flood_fill(seed,cx,cz,max_cells=2_000_000):
    """Flood fill at 1:4 scale. cx,cz are 1:4 coordinates."""
    buf=(ctypes.c_ubyte*ENG_SZ)();_eng._lib.cont_engine_init(buf,seed&0xFFFFFFFFFFFFFFFF,0)
    def s(x,z):return _eng._lib.cont_sample(buf,x,z)
    sx,sz=int(cx),int(cz)  # already at 1:4 scale
    if s(sx,sz)>=-1.05:return 0
    q=deque([(sx,sz)]);seen={(sx,sz)};cache={(sx,sz):s(sx,sz)};cells=0
    while q and cells<max_cells:
        x,z=q.popleft();cells+=1
        for dx,dz in[(1,0),(-1,0),(0,1),(0,-1)]:
            nx,nz=x+dx,z+dz
            if(nx,nz)in seen:continue
            seen.add((nx,nz));cache[(nx,nz)]=s(nx,nz)
            if cache[(nx,nz)]<-1.05:q.append((nx,nz))
    return cells*16  # each 1:4 cell = 4x4 = 16 blocks at 1:1

if __name__=='__main__':
    S=5_000_000;D=math.sqrt(S)       # D=2236 blocks at 1:1
    D4=D/4                            # D=559 cells at 1:4
    step=int(D4/2.5)//4*4             # step=220, K=2, spans 2.5 cells
    K=2;G=4;ng=2048//(G*G)            # G=4 → 128 grids
    radius=int(8000)                  # 8000 cells at 1:4 = 32K blocks
    batch=1024
    print(f'S={S:,} D4={D4:.0f} step={step} K={K} G={G} grids={ng} radius={radius:,}')
    best,scanned,t0=None,0,time.perf_counter()
    try:
        while True:
            off=np.array([v for _ in range(ng) for v in
                (np.random.randint(-radius,radius)//step*step,
                 np.random.randint(-radius,radius)//step*step)],dtype=np.int32)
            seeds=np.random.randint(0,2**63,size=batch,dtype=np.uint64)
            t1=time.perf_counter()
            hits=gpu_scan(seeds,step,K,G,off)
            scanned+=batch
            # Only flood fill top candidates (flood fill is the bottleneck)
            for seed,hx,hz in hits[:5]:
                area=flood_fill(seed,hx,hz)
                if best is None or area>best['area']:
                    best={'seed':seed,'area':area,'center':(hx,hz)}
                    print(f'\n  *** NEW BEST: seed {seed}, {area:,} blocks^2 at ({hx},{hz}) ***')
                    with open('best_1m.json','w')as f:json.dump(best,f)
            elapsed=time.perf_counter()-t0;rate=scanned/elapsed
            ba=best['area']if best else 0
            print(f'\r  {scanned:,} seeds ({rate:.0f}/s) | {len(hits)} hits | best {ba:,}  ',end='')
    except KeyboardInterrupt:pass
    print(f'\n\nScanned {scanned:,} seeds in {time.perf_counter()-t0:.0f}s')
    if best:
        print(f"BEST: seed {best['seed']}, {best['area']:,} blocks^2 at {best['center']}")

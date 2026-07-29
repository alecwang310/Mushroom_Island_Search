@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\Code\Seeds\gpu
nvcc -O3 -arch=sm_120 -Xptxas=-v,-warn-spills,-warn-lmem-usage -shared -o hunt_engine.dll hunt_engine.cu tiered_kernel.cu coarse_verify_kernel.cu prefilter_kernel.cu ..\engine\continentalness.c -I..\engine -lcudart

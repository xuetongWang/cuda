#include <iostream>
#include <cuda_runtime.h>
#include "cudaUtility.h"

__global__ void add(int a, int b, int *c){
    *c = a+b;
}
int main()
{
    int c;
    int *dev_c;

    cudaDeviceProp prop;
    int count;

    CUDA_FAILED(cudaMalloc((void **)&dev_c,sizeof(int)));
    add<<<1,1>>>(2,7,dev_c);

    CUDA_FAILED(cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost));

    printf("c value is %d\n",c);

    cudaFree(dev_c);

    CUDA_FAILED(cudaGetDeviceCount(&count));

    for(int i = 0; i< count; i++){
        CUDA_FAILED(cudaGetDeviceProperties(&prop ,i));
        printf("---------device iNFO num = %d----\n",i);
        printf("name = %s----\n",prop.name);
        printf("compute capability:%d.%d----\n",prop.major, prop.minor);
        printf("colock rate::%d----\n",prop.clockRate);

        printf("------device copy overlap:----\n");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("Kernel execition timeout : ");
        if(prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
         printf("---------Memory iNFO for device = %d----\n",i);
         printf("total global mem : %ld----\n",prop.totalGlobalMem);
         printf("total const mem : %ld----\n",prop.totalConstMem);
         printf("max  mem pitch : %ld----\n",prop.memPitch);
         printf("Texture  Alignment : %ld----\n",prop.textureAlignment);
         printf("---------Mp iNFO for device = %d----\n",i);
         printf("Multiprocessor count  : %d----\n",prop.multiProcessorCount);
         printf("Shared mem per  mp : %ld----\n",prop.sharedMemPerBlock);
         printf("Registers pre mp  : %ld----\n",prop.regsPerBlock);
         printf("Threads in warp  : %ld----\n",prop.warpSize);
         printf("Max threads  pre block  : %ld----\n",prop.maxThreadsPerBlock);
         printf("Max threads  dimensions  : (%d, %d, %d)\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
         printf("Max grid  dimensions  : (%d, %d, %d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    }
    return 0;
}

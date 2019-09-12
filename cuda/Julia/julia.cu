#include <iostream>
#include <cuda_runtime.h>
#include "../include/cpu_bitmap.h"
#include "../include/cudaUtility.h"

#define DIM 1000
/*
 * CPU code
*/
struct cuComplex_cpu
{
    float r;
    float i;
    cuComplex_cpu(float a, float b):r(a),i(b){}

    float magnitude2(void){return r*r +i*i;}

    cuComplex_cpu operator*(const cuComplex_cpu& a){

        return cuComplex_cpu(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    cuComplex_cpu operator+(const cuComplex_cpu& a){
        return cuComplex_cpu(r+a.r, i+a.i);
    }
};

struct cuComplex_gpu
{
    float r;
    float i;
    __device__ cuComplex_gpu(float a, float b):r(a),i(b){}

    __device__ float magnitude2(void){return r*r +i*i;}

    __device__ cuComplex_gpu operator*(const cuComplex_gpu& a){

        return cuComplex_gpu(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ cuComplex_gpu operator+(const cuComplex_gpu& a){
        return cuComplex_gpu(r+a.r, i+a.i);
    }
};

int julia_cpu(int x, int y){
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex_cpu c(-0.8, 0.156);
    cuComplex_cpu a(jx, jy);

    for(int i=0; i<200; i++){
        a = a*a + c;
        if(a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__device__ int julia_gpu(int x, int y){
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex_gpu c(-0.8, 0.156);
    cuComplex_gpu a(jx, jy);

    for(int i=0; i<200; i++){
        a = a*a + c;
        if(a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

void kernle_cpu(unsigned char *ptr){
    for(int y = 0; y<DIM;y++){
        for(int x =0; x<DIM; x++){
            int offset = x + y * DIM;
            int juliaValue = julia_cpu(x, y);

            ptr[offset*4 + 0] = 255* juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}

__global__ void kernle_gpu(unsigned char * ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x+ y * gridDim.x;

    int juliaValue = julia_gpu(x, y);

    ptr[offset*4 + 0] = 255* juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;

}
/*
 * int main (void){

    CPUBitmap bitmap (DIM,DIM);

    void *ptr =  bitmap.get_ptr();

    kernel_cpu((unsigned char *)ptr);

    bitmap.display_and_exit();
}
*/
int main (void){

    CPUBitmap bitmap (DIM,DIM);

    unsigned char *dev_bitmap;

    CUDA_FAILED(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernle_gpu<<<grid, 1>>>(dev_bitmap);

    CUDA_FAILED(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}

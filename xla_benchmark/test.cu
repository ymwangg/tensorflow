#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void compute(float*x, float*y, float*z, int64_t N, int64_t n) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = 0; i < n; i++) {
        int global_id = tid * n + i;
        if (global_id < N) {
            float tmp = x[tid] / y[tid];
            tmp += x[tid];
            tmp = __expf(tmp);
            z[tid] = tmp;
        }

    }
}
int main(){
    int64_t N = 12288*30522;
    float *x, *y, *z;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    z = (float*)malloc(N*sizeof(float));
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(float));
    // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    /*
        HloModule Module
        ENTRY main {
          %x = f32[12288,30522]{1,0} parameter(0)
          %y = f32[12288,30522]{1,0} parameter(1)
          %z = f32[12288,30522]{1,0} divide(f32[12288,30522]{1,0} %x, f32[12288,30522]{1,0} %y)
          %w = f32[12288,30522]{1,0} add(f32[12288,30522]{1,0} %z, f32[12288,30522]{1,0} %x)
          ROOT %a = f32[12288,30522]{1,0} exponential(f32[12288,30522]{1,0} %w)
        }
    */
    int num = 32*2;
    for (int i = 0; i < 100; i++) {
      int b = 163840/num;
      compute<<<b,num>>>(d_x, d_y, d_z, N, 2290);
    }
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyDeviceToHost);

}

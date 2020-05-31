#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>
#include "util/matrix.h"
#include "util/timer.h"

using namespace cutlass;

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 4096;
  float alpha = 1.0;
  float beta = 0.0;
  typedef float value_t;
  typedef float accum_t;
  int g_timing_iterations = 10;
  cudaStream_t stream = 0;
  matrix<value_t> A(m, k);
  matrix<value_t> B(k, n);
  matrix<accum_t> C(m, n);
  matrix<accum_t> C2(m, n);
  A.random();
  B.random();
  C.fill_ramp(0,0);
  C2.fill_ramp(0,0);
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    CUDA_PERROR(cublasSgemm(
                            g_cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            B.d_data(),
                            n,
                            A.d_data(),
                            k,
                            &beta,
                            C.d_data(),
                            n));
  }
  timer.stop();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;
  typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    gemm::dispatch<epilogue_op_t>(
        m,
        n,
        k,
        alpha,
        beta,
        A.d_data(),
        B.d_data(),
        C2.d_data(),
        stream,
        false);      
     }
  timer.stop();
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C.get(i,j) - C2.get(i,j));
    }
  }
  printf("error: %lf\n", err/n/m);
  cublasDestroy(g_cublas_handle);
}

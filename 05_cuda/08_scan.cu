#include <cstdio>
#include <cstdlib>

__global__ void scan(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=1; j<N; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    if(i>=j) a[i] += b[i-j];
    __syncthreads();
  }
}

int main() {
  const int N=8;
  int *a, *b;
  cudaMallocManaged(&a, N*sizeof(int));
  cudaMallocManaged(&b, N*sizeof(int));
  for(int i=0; i<N; i++) {
    a[i] = rand() & 3;
    printf("%*d ",2,a[i]);
  }
  printf("\n");
  scan<<<1,N>>>(a, b, N);
  cudaDeviceSynchronize();
  for(int i=0; i<N; i++)
    printf("%*d ",2,a[i]);
  printf("\n");
  cudaFree(a);
  cudaFree(b);
}

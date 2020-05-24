//Assignment of CUDA
// 19M18085 Lian Tongda

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_set_zero(int *bucket, int range)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i > range) return;
    bucket[i] = 0;
    __syncthreads();
}

__global__ void bucket_set(int *key, int *bucket, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= n) return;
    atomicAdd(&bucket[key[i]], 1);
    __syncthreads();
}

__global__ void bucket_sort(int *key, int *bucket, int n, int range)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= range) return;
    
    int begin = 0;
//    int end   = 0;
    
    for( int j  = 0 ; j < i ; j++)
    {
        begin += bucket[(j+n)%n];
//        end += bucket[(j+1+n)%n] - 1;
    }
    
    __syncthreads();

//    for( int j  = 0 ; j <= i ; j++)
//    {
//        end += bucket[(j+n)%n] - 1;
//    }  
//    __syncthreads();
    
    for(int k = begin ; k < begin + bucket[i] ; k++)
    {
        key[k] = i;
    }
    __syncthreads();
//    for(; bucket[i] > 0 ; bucket[i]--)
//    {
//        key[j++] = i;
//    }
}


int main() {
  int n = 50;
  int range = 5;
    
    int sizeN = n * sizeof(int);
    int sizeRANGE = range * sizeof(int);
    
//  std::vector<int> key(n);
    
//    int *key, *key_gpu;
//    key = (int*)malloc(sizeN);
//
//  for (int i=0; i<n; i++)
//  {
//    key[i] = rand() % range;
//    printf("%d ",key[i]);
//  }
//  printf("\n");
    
    int *key;
    cudaMallocManaged(&key, sizeN);
    
    for (int i=0; i<n; i++)
      {
        key[i] = rand() % range;
        printf("%d ",key[i]);
      }
      printf("\n");

//  std::vector<int> bucket(range);

//  for (int i=0; i<range; i++)
//  {
//    bucket[i] = 0;
//  }

    int *bucket;
    cudaMallocManaged(&bucket, sizeRANGE);
    bucket_set_zero<<<1, range>>>( bucket, range);
    cudaDeviceSynchronize();
    
//  for (int i=0; i<n; i++)
//  {
//    bucket[key[i]]++;
//  }
    bucket_set<<<1, n>>>(key, bucket, n);
    cudaDeviceSynchronize();
    
//    for(int i = 0 ; i < range ; i++)
//    {
//        printf("%d ", bucket[i]);
//    }
//    printf("\n");
    
//  for (int i=0, j=0; i<range; i++) {
//    for (; bucket[i]>0; bucket[i]--) {
//      key[j++] = i;
//    }
//  }
    
    bucket_sort<<<1, range>>>(key, bucket,  n,  range);
    cudaDeviceSynchronize();
    
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
    
  printf("\n");
    return 0;
}

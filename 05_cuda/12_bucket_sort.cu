#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __managed__ int bucket[5];

__global__ void initialization(int * bucket){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[i] = 0;
}
__global__ void reduction(int *bucket,int *key){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]],1);
}

__global__ void  bucketsort(int *bucket, int *key){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=0, k=0;k<=i;j++){
  key[i]=j;
  k+=bucket[j];  
}
}
__global__ void fillkey(int* key, int *bucket){
   int i = threadIdx.x;
   int j = bucket[i];
   for (int k=1;k<8;k<<=1){
       int n= __shfl_up_sync(0xffffffff, j, k);
       if(i>=k) j+=n;
}
   j -= bucket[i];
   for(;bucket[i];bucket[i]--)
      key[j++]=i;
} 


int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  initialization<<<1,range>>>(bucket);
  cudaDeviceSynchronize();
  reduction<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();
<<<<<<< HEAD
  for (int i=0; i<range; i++) {
    printf("%d ", bucket[i]);
  }
  printf("\n");
//  bucketsort<<<1,n>>>(bucket, key, range);
  fillkey<<<1,range>>>(key,bucket);
  cudaDeviceSynchronize();
/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
//get the offset of j
=======
  bucketsort<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();
>>>>>>> 2026757663db05c3bde302abdbd82ad559e7c0b7

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fillBucket(int* key, int *bucket) {
  int i = threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void fillKey(int *key, int *bucket) {
  int i = threadIdx.x;
  int j = bucket[i];
  for (int k=1; k<8; k<<=1) {
    int n = __shfl_up_sync(0xffffffff, j, k);
    if (i >= k) j += n;
  }
  j -= bucket[i];
  for (; bucket[i]>0; bucket[i]--)
    key[j++] = i;
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  for (int i=0; i<range; i++)
    bucket[i] = 0;
  fillBucket<<<1,n>>>(key, bucket);
  fillKey<<<1,range>>>(key, bucket);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++)
    printf("%d ",key[i]);
  printf("\n");
}

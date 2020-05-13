#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  printf("initial:");
#pragma omp parallel
{ 

  for (int i=0; i<n; i++) {
#pragma omp single
{
    key[i] = rand() % range;
    printf("%d ",key[i]);
}
  }
#pragma omp single
  printf("\n");
}
  std::vector<int> bucket(range);
#pragma omp parallel
{ 
#pragma omp for 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
}
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++) {
#pragma omp atomic update
    bucket[key[i]]++;
  }
#pragma omp parallel
{
#pragma omp sections
{
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
}
} 
#pragma omp parallel
{
#pragma omp single 
  printf("sorted:");
  
  for (int i=0; i<n; i++) {
#pragma omp single
    printf("%d ",key[i]);
  }
#pragma omp single
  printf("\n");
}
}

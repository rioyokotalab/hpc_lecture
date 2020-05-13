#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  
//    19M18085 Lian Tongda
#pragma omp parallel for
  for (int i=0; i<n; i++)
  {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range); 
//    19M18085 Lian Tongda
#pragma omp parallel for
  for (int i=0; i<range; i++) 
  {
    bucket[i] = 0;
  }
  
//    19M18085 Lian Tongda
#pragma omp parallel for
  for (int i=0; i<n; i++) 
  {
    //    19M18085 Lian Tongda
    #pragma omp atomic update
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  
//    19M18085 Lian Tongda
#pragma omp parallel
  for (int i=0; i<n; i++) {
//    19M18085 Lian Tongda
#pragma omp single
    printf("%d ",key[i]);
  }
  printf("\n");
}

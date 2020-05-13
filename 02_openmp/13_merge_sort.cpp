#include <cstdio>
#include <cstdlib>
#include <vector>

template<class T>
void merge(std::vector<T>& vec, int begin, int mid, int end) {
  std::vector<T> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}

template<class T>
void merge_sort(std::vector<T>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    //   19M18085 Lian Tongda
    #pragma omp task shared(vec)
    //   19M18085 Lian Tongda
    merge_sort(vec, begin, mid);
    merge_sort(vec, mid+1, end);
    //   19M18085 Lian Tongda
    #pragma omp taskwait
    //   19M18085 Lian Tongda
    merge(vec, begin, mid, end);
  }
}

int main() {
  int n = 20;
  std::vector<int> vec(n);
  
 // 19M18085 Lian Tongda
  #pragma omp parallel
  {
    #pragma omp for
//     19M18085 Lian Tongda
  for (int i=0; i<n; i++) 
  {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  }
  printf("\n");

  merge_sort(vec, 0, n-1);

//   19M18085 Lian Tongda
  #pragma omp parallel
  for (int i=0; i<n; i++) 
  {
    #pragma omp single
//   19M18085 Lian Tongda
    printf("%d ",vec[i]);
  }
  printf("\n");
}

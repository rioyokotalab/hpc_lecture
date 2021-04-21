#include <cstdio>
#include<omp.h>

int main() {
  int n = 10;
  double dx = 1. / n;
  double pi = 0;
  #pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    //printf("%d",omp_get_thread_num());
    pi += 4.0 / (1.0 + x * x) * dx;
    //printf("%d\n",omp_get_thread_num());
  }
  printf("%17.15f\n",pi);
}

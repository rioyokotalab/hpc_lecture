#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
        int n = 50;
        int range = 5;
        std::vector<int> key(n);
        for (int i=0; i<n; i++) {
                key[i] = rand() % range;
                printf("%d ",key[i]);
        }
        printf("\n");

        std::vector<int> bucket(range);
        #pragma omp parallel
        {
          #pragma omp for
            for (int i=0; i<range; i++) {
                    bucket[i] = 0;
                    omp_init_lock(&bucket_locks[i]);
            }

          #pragma omp for
          for (int i=0; i<n; i++) {
                  omp_set_lock(&bucket_locks[key[i]])
                  bucket[key[i]]++;
                  omp_unset_lock(&bucket_locks[key[i]])
          }

          #pragma omp for
          for (int i=0; i<range; i++) {
                  int j = 0;
                  #pragma omp for reduction(+:j)
                  for (int k = 0; k < bucket[i]; k++) {
                          key[j + k] = i;
                          j++;
                  }
          }

          #pragma omp for
          for (int i=0; i<range; i++) {
                  omp_destroy_lock(&bucket_locks[i]);
          }
        }

        for (int i=0; i<n; i++) {
                printf("%d ",key[i]);
        }
        printf("\n");
}

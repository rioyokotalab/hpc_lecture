#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

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
        std::vector<omp_lock_t> bucket_locks(range);
        std::vector<int> starting_index(range);
        std::vector<int> ending_index(range);
        int b[range];
        ending_index[range-1] = n;
        int j = 0;

        // double time = omp_get_wtime();
        #pragma omp parallel
        {
          // initialize variables
          #pragma omp for
            for (int i=0; i<range; i++) {
                    bucket[i] = 0;
                    omp_init_lock(&bucket_locks[i]);
            }

            // update bucket in parallel
          #pragma omp for
          for (int i=0; i<n; i++) {
                  omp_set_lock(&bucket_locks[key[i]]);
                  bucket[key[i]]++;
                  omp_unset_lock(&bucket_locks[key[i]]);
          }

          // prefix sum for starting indices
          for(int j=1; j<range; j<<=1) {
          #pragma omp for
            for(int i=0; i<range; i++)
            b[i] = bucket[i];
          #pragma omp for
            for(int i=j; i<range; i++)
            starting_index[i] += b[i-j];
          }

          // initialize ending indices
          #pragma omp for
          for (int i=0; i < range - 1; i++) {
              ending_index[i] = starting_index[i+1];
          }

          // Change key value to the corresponding bucket id
          #pragma omp for
          for (int i = 0; i < range; i++) {
            // printf("%d %d\n",starting_index[i], ending_index[i]);
            for(int j = starting_index[i]; j < ending_index[i]; j++) {
              // printf("%d %d \n",j, omp_get_thread_num());
                  key[j] = i;
                }
          }

          // remove locks
          #pragma omp for
          for (int i=0; i<range; i++) {
                  omp_destroy_lock(&bucket_locks[i]);
          }
        }
        // printf("%f\n", omp_get_wtime() - time);

        printf("\n");
        for (int i=0; i<n; i++) {
                printf("%d ",key[i]);
        }
        printf("\n");
}

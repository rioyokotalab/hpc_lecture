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

        #pragma omp parallel
        {
          // Initialize variables.
          #pragma omp for
            for (int i=0; i<range; i++) {
                    bucket[i] = 0;
                    omp_init_lock(&bucket_locks[i]);
            }

          // Update bucket in parallel.
          // The larger the range, the more unlikely it is that two thread write
          // to the same bucket. Thus we expect the locks to not be accessed often.
          // In case there is a simultaneous acces, the locks are placed.
          #pragma omp for
          for (int i=0; i<n; i++) {
                  omp_set_lock(&bucket_locks[key[i]]);
                  bucket[key[i]]++;
                  omp_unset_lock(&bucket_locks[key[i]]);
          }

          // Prefix sum for starting indices.
          // The starting index is the sum of the number of elements in all
          // the buckets with a smaller index.
          for(int j=1; j<range; j<<=1) {
          #pragma omp for
            for(int i=0; i<range; i++)
            b[i] = bucket[i] + starting_index[i];
          #pragma omp for
            for(int i=j; i<range; i++)
            starting_index[i] += b[i-j];
          }

          // Initialize ending indices
          #pragma omp for
          for (int i=0; i < range - 1; i++) {
              ending_index[i] = starting_index[i+1];
          }

          // Change key value to the corresponding bucket id, as done in the
          // initial code. Since the indices for the keys are non-overlapping,
          // we can assign the values in parallel.
          #pragma omp for
          for (int i = 0; i < range; i++) {
            for(int j = starting_index[i]; j < ending_index[i]; j++) {
                  key[j] = i;
                }
          }

          // Remove locks
          #pragma omp for
          for (int i=0; i<range; i++) {
                  omp_destroy_lock(&bucket_locks[i]);
          }
        }

        printf("\n");
        for (int i=0; i<n; i++) {
                printf("%d ",key[i]);
        }
        printf("\n");
}

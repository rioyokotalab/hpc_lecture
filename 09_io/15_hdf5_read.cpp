#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
using namespace std;

int main (int argc, char** argv) {
  hid_t file = H5Fopen("data.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t dataset = H5Dopen(file, "dataset", H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  hsize_t N[ndim];
  H5Sget_simple_extent_dims(dataspace, N, NULL);
  int NX = N[0], NY = N[1];
  vector<int> buffer(NX*NY);
  auto tic = chrono::steady_clock::now();
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer[0]);
  auto toc = chrono::steady_clock::now();
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Fclose(file);
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<NX*NY; i++) {
    sum += buffer[i];
  }
  printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  printf("sum=%d\n",sum);
}

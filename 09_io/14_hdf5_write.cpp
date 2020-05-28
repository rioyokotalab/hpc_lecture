#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
using namespace std;

int main (int argc, char** argv) {
  const int NX = 10000, NY = 10000;
  vector<int> buffer(NX*NY,1);
  hid_t file = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t N[2] = {NX, NY};
  hid_t dataspace = H5Screate_simple(2, N, NULL);
  hid_t dataset = H5Dcreate(file, "dataset", H5T_NATIVE_INT, dataspace,
			    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  auto tic = chrono::steady_clock::now();
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer[0]);
  auto toc = chrono::steady_clock::now();
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Fclose(file);
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
}

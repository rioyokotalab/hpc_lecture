#include <mpi.h>
#include <cstdio>
#include <chrono>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
  const int N = 100000000;
  int mpisize, mpirank;
  MPI_File file;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int Nlocal = N / mpisize;
  int offset = Nlocal * mpirank;
  vector<int> buffer(Nlocal,1);
  MPI_File_open(MPI_COMM_WORLD, "data.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  auto tic = chrono::steady_clock::now();
  MPI_File_write_at(file, offset, &buffer[0], Nlocal, MPI_INT, MPI_STATUS_IGNORE);
  auto toc = chrono::steady_clock::now();
  MPI_File_close(&file);
  double time = chrono::duration<double>(toc - tic).count();
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  MPI_Finalize();
}

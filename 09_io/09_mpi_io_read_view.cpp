#include <mpi.h>
#include <cstdio>
#include <chrono>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
  int mpisize, mpirank;
  MPI_File file;
  MPI_Offset filesize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_File_open(MPI_COMM_WORLD, "data.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_get_size(file, &filesize);
  int N = filesize / sizeof(int);
  int Nlocal = N / mpisize;
  int offset = Nlocal * mpirank;
  vector<int> buffer(Nlocal);
  MPI_File_set_view(file, offset, MPI_INT, MPI_INT, "navite", MPI_INFO_NULL);
  auto tic = chrono::steady_clock::now();
  MPI_File_read(file, &buffer[0], Nlocal, MPI_INT, MPI_STATUS_IGNORE);
  auto toc = chrono::steady_clock::now();
  MPI_File_close(&file);
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<Nlocal; i++)
    sum += buffer[i];
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("rank=%d sum=%d\n",mpirank,sum);
  MPI_Finalize();
}

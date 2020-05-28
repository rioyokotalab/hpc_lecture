#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
  int dim[2] = {2, 2};
  int mpisize, mpirank;
  MPI_File file;
  MPI_Offset filesize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_File_open(MPI_COMM_WORLD, "data.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_get_size(file, &filesize);
  int NX = sqrt(filesize / sizeof(int));
  int NY = NX;
  int N[2] = {NX, NY};
  int Nlocal[2] = {NX/dim[0], NY/dim[1]};
  int offset[2] = {mpirank / dim[0], mpirank % dim[0]};
  for(int i=0; i<2; i++) offset[i] *= Nlocal[i];
  vector<int> buffer(Nlocal[0]*Nlocal[1]);
  MPI_Datatype MPI_SUBARRAY;
  MPI_Type_create_subarray(2, N, Nlocal, offset,
			   MPI_ORDER_C, MPI_INT, &MPI_SUBARRAY);
  MPI_Type_commit(&MPI_SUBARRAY);
  MPI_File_set_view(file, 0, MPI_INT, MPI_SUBARRAY, "navite", MPI_INFO_NULL);
  auto tic = chrono::steady_clock::now();
  MPI_File_read_all(file, &buffer[0], Nlocal[0]*Nlocal[1], MPI_INT, MPI_STATUS_IGNORE);
  auto toc = chrono::steady_clock::now();
  MPI_File_close(&file);
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<Nlocal[0]*Nlocal[1]; i++)
    sum += buffer[i];
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  printf("rank=%d sum=%d\n",mpirank,sum);
  MPI_Finalize();
}

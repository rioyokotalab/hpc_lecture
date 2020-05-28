#include "mpi.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int main(int argc, char ** argv) {
  struct timeval tic, toc;
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int size = atoi(argv[1]);
  int * send = new int [size];
  int * recv = new int [size];
  for(int i=0; i<size; i++)
    send[i] = mpirank+size*i;
  int send_rank = (mpirank + 1) % mpisize;
  int recv_rank = (mpirank - 1 + mpisize) % mpisize;
  gettimeofday(&tic, NULL);
  MPI_Send(send, size, MPI_INT, send_rank, 0, MPI_COMM_WORLD);
  MPI_Recv(recv, size, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  gettimeofday(&toc, NULL);
  printf("%lf s\n",toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  delete[] send;
  delete[] recv;
  MPI_Finalize();
}

#include <cstdio>
#include <fstream>
#include <chrono>
#include <vector>
using namespace std;

int main () {
  ifstream file("data.dat", ios::binary | ios::ate);
  int N = file.tellg()/sizeof(int);
  vector<int> buffer(N);
  file.seekg (0, ios::beg);
  auto tic = chrono::steady_clock::now();
  file.read((char*)&buffer[0], N*sizeof(int));
  auto toc = chrono::steady_clock::now();
  file.close();
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<N; i++)
    sum += buffer[i];
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("sum=%d\n",sum);
}

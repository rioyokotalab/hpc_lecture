#include <cstdio>
#include <fstream>
#include <chrono>
#include <vector>
using namespace std;

int main () {
  vector<int> buffer;
  ifstream file("data.dat");
  int a;
  auto tic = chrono::steady_clock::now();
  while(file >> a)
    buffer.push_back(a);
  auto toc = chrono::steady_clock::now();
  file.close();
  double time = chrono::duration<double>(toc - tic).count();
  int N = buffer.size();
  int sum = 0;
  for (int i=0; i<N; i++)
    sum += buffer[i];
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("sum=%d\n",sum);
}

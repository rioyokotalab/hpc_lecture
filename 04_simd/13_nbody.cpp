// Assignment of High Performance Computation
// 19M18085 Lian Tongda
 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main()
{
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
//      double x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
    
//    Vectorization of x, y and m
    
    __m256 xv = _mm256_load_ps(x);
    __m256 yv = _mm256_load_ps(y);
    __m256 mv = _mm256_load_ps(m);
    
  for(int i=0; i<N; i++)
  {
      //    Vectorization of x, y of i-index particles
//      All values of iv will be i
      __m256 iv  = _mm256_set1_ps(i);
      __m256 xiv = _mm256_set1_ps(x[i]);
      __m256 yiv = _mm256_set1_ps(y[i]);
      float index[N];
      
    for(int j=0; j<N; j++)
    {
        
//    =====    if(i != j) start     =====
        index[j] = j;
        __m256 jv = _mm256_load_ps(index);
        
//        If j != i, value of mask[j] will be 1
//        Otherwise, if j == i, value of mask[j] will be 0
        __m256 mask = _mm256_cmp_ps(jv, iv, _CMP_NEQ_OQ);
        
//        All values are set to 0
        __m256 xjv = _mm256_setzero_ps();
        __m256 yjv = _mm256_setzero_ps();
        __m256 mjv = _mm256_setzero_ps();
        
//        If j == i, value of xjv[j], yjv[j] and mjv[j] will be set as 0
        xjv = _mm256_blendv_ps(xjv, xv, mask);
        yjv = _mm256_blendv_ps(yjv, yv, mask);
        mjv = _mm256_blendv_ps(mjv, mv, mask);
        
//    =====     if(i != j) end     =====
        
//        Calculation of rx, ry, r^2 and r^3
        __m256 rxv  = _mm256_sub_ps(xiv, xjv);
        __m256 rx2v = _mm256_mul_ps(rxv, rxv);
        
        __m256 ryv  = _mm256_sub_ps(yiv, yjv);
        __m256 ry2v = _mm256_mul_ps(ryv, ryv);
        
        __m256 r2v = _mm256_add_ps(rx2v, ry2v);
        __m256 rv  = _mm256_sqrt_ps(r2v);
        __m256 r3v = _mm256_mul_ps(rv, r2v);
        
//        Calculation of m * rx and m * ry
        __m256 mrxv = _mm256_mul_ps(rxv, mjv);
        __m256 mryv = _mm256_mul_ps(ryv, mjv);
        
//        Calculation of force between particle i and j
        __m256 fxiv = - _mm256_div_ps(mrxv, r3v);
        __m256 fyiv = - _mm256_div_ps(mryv, r3v);
        
        //        Reduction of fx applied by all other particles
        __m256 fxv = _mm256_permute2f128_ps(fxiv, fxiv, 1);
        fxv = _mm256_add_ps(fxv, fxiv);
        fxv = _mm256_hadd_ps(fxv, fxv);
        fxv = _mm256_hadd_ps(fxv, fxv);
        _mm256_store_ps(fx, fxv);
        
        //        Reduction of fy applied by all other particles
        __m256 fyv = _mm256_permute2f128_ps(fyiv, fyiv, 1);
        fyv = _mm256_add_ps(fyv, fyiv);
        fyv = _mm256_hadd_ps(fyv, fyv);
        fyv = _mm256_hadd_ps(fyv, fyv);
        _mm256_store_ps(fy, fyv);
        
//      if(i != j)
//      {
//        double rx = x[i] - x[j];
//        double ry = y[i] - y[j];
//        double r = std::sqrt(rx * rx + ry * ry);
//        fx[i] -= rx * m[j] / (r * r * r);
//        fy[i] -= ry * m[j] / (r * r * r);
//      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

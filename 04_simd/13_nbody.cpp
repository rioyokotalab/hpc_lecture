#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
<<<<<<< HEAD
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    //load forced particle variable
    __m256 ivec= _mm256_set1_ps(i);
    __m256 ixvec= _mm256_set1_ps(x[i]);
    __m256 iyvec= _mm256_set1_ps(y[i]);
    float a[N];
    for(int j=0; j<N; j++)
      a[j] = j;
     //load force particle variable
     __m256 jvec  = _mm256_load_ps(a);
     __m256 mask  = _mm256_cmp_ps(jvec, ivec, _CMP_NEQ_OQ);

     __m256 jxvec = _mm256_setzero_ps();
     __m256 jyvec = _mm256_setzero_ps();
     __m256 jmvec = _mm256_setzero_ps();

     jxvec = _mm256_blendv_ps(jxvec, xvec, mask);
     jyvec = _mm256_blendv_ps(jyvec, yvec, mask);
     jmvec = _mm256_blendv_ps(jmvec, mvec, mask);

     __m256 rxvec = _mm256_sub_ps(ixvec, jxvec);
     __m256 ryvec = _mm256_sub_ps(iyvec, jyvec);
     __m256 rx2vec = _mm256_mul_ps(rxvec, rxvec);
     __m256 ry2vec = _mm256_mul_ps(ryvec, ryvec);
     __m256 rprevec = _mm256_add_ps(rx2vec, ry2vec);
     __m256 rvec  = _mm256_sqrt_ps(rprevec);//slower than rsqrt 10 times

     __m256 r2vec = _mm256_mul_ps(rvec, rvec);
     __m256 r3vec = _mm256_mul_ps(rvec, r2vec);

     __m256 mrxvec = _mm256_mul_ps(rxvec, jmvec);
     __m256 mryvec = _mm256_mul_ps(ryvec, jmvec);

     __m256 ifxvec =- _mm256_div_ps(mrxvec, r3vec);
     __m256 ifyvec =- _mm256_div_ps(mryvec, r3vec);

     __m256 fxvec = _mm256_permute2f128_ps(ifxvec,ifxvec,1);
     fxvec = _mm256_add_ps(fxvec,ifxvec);
     fxvec = _mm256_hadd_ps(fxvec,fxvec);
     fxvec = _mm256_hadd_ps(fxvec,fxvec);
     __m256 fyvec = _mm256_permute2f128_ps(ifyvec,ifyvec,1);
     fyvec = _mm256_add_ps(fyvec,ifyvec);
     fyvec = _mm256_hadd_ps(fyvec,fyvec);
     fyvec = _mm256_hadd_ps(fyvec,fyvec);

    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
=======
  __m256 zero = _mm256_setzero_ps();
  for(int i=0; i<N; i+=8) {
    __m256 xi = _mm256_load_ps(x+i);
    __m256 yi = _mm256_load_ps(y+i);
    __m256 fxi = zero;
    __m256 fyi = zero;
    for(int j=0; j<N; j++) {
      __m256 dx = _mm256_set1_ps(x[j]);
      __m256 dy = _mm256_set1_ps(y[j]);
      __m256 mj = _mm256_set1_ps(m[j]);
      __m256 r2 = zero;
      dx = _mm256_sub_ps(xi, dx);
      dy = _mm256_sub_ps(yi, dy);
      r2 = _mm256_fmadd_ps(dx, dx, r2);
      r2 = _mm256_fmadd_ps(dy, dy, r2);
      __m256 mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ);
      __m256 invR = _mm256_rsqrt_ps(r2);
      invR = _mm256_blendv_ps(zero, invR, mask);
      mj = _mm256_mul_ps(mj, invR);
      invR = _mm256_mul_ps(invR, invR);
      mj = _mm256_mul_ps(mj, invR);
      fxi = _mm256_fmadd_ps(dx, mj, fxi);
      fyi = _mm256_fmadd_ps(dy, mj, fyi);
    }
    _mm256_store_ps(fx, fxi);
    _mm256_store_ps(fy, fyi);
>>>>>>> ebc183ec04fbd943b6eda12f4665b266c3a700da
  }
  for(int i=0; i<N; i++)
    printf("%d %g %g\n",i,fx[i],fy[i]);
}

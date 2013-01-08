#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  double *a;
  double *b;
  double *out;
  int     m;
  int     n;
  int     k;
} vm_args_t;

double *make_array(int m, int n) {
  double *array = (double*)malloc(m * n * sizeof(double));
  int i;
  for (i = 0; i < m * n; ++i) {
    array[i] = ((double)m) / n;
  }
  return array;
}

void free_array(double *array) {
  free(array);
}

void vm(int start, int end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int l3bLen, l2aLen, l2bLen, l1aLen, l1bLen;
  int i, j, l;
  int aOff, bOff, oOff;

  l3bLen = tile_sizes[0];
  l2aLen = tile_sizes[1];
  l2bLen = tile_sizes[2];
  l1aLen = tile_sizes[3];
  l1bLen = tile_sizes[4];
  int l3b, l2a, l2b, l2as, l2bs, l1a, l1b, l1as, l1bs;
  int is, js;
  // l3a tiling amounts to size of chunk
  for (l3b = 0; l3b < n; l3b += l3bLen) {
    for (l2a = start; l2a < end; l2a += l2aLen) {
      l2bs = l3b + l3bLen;
      if (l2bs > n) l2bs = n;
      for (l2b = l3b; l2b < l2bs; l2b += l2bLen) {
        l1as = l2a + l2aLen;
        if (l1as > m) l1as = m;
        for (l1a = l2a; l1a < l1as; l1a += l1aLen) {
          l1bs = l2b + l2bLen;
          if (l1bs > n) l1bs = n;
          for (l1b = l2b; l1b < l1bs; l1b += l1bLen) {
            is = l1a + l1aLen;
            if (is > m) is = m;
            for (i = l1a; i < is; ++i) {
              aOff = i * k;
              oOff = i * n;
              js = l1b + l1bLen;
              if (js > n) js = n;
              for (j = l1b; j < js; ++j) {
                bOff = j * k;
                O[oOff + j] = 0.0;
                for (l = 0; l < k; ++l) {
                  O[oOff + j] += A[aOff + l] * B[bOff + l];
                }
              }
            }
          }
        }
      }
    }
  }
}

static inline int min(a, b) {
  return a < b ? a : b;
}

void vm2(int start, int end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int i, j, l;
  int aOff, bOff, oOff;

  int l2b, l1a, l1b, l1as, l1bs;
  int l2bLen, l1aLen, l1bLen;
  l2bLen = tile_sizes[0];
  l1aLen = tile_sizes[1];
  l1bLen = tile_sizes[2];
  int is, js;
  for (l2b = 0; l2b < n; l2b += l2bLen) {
    for (l1a = start; l1a < end; l1a += l1aLen) {
      l1bs = l2b + l2bLen;
      if (l1bs > n) l1bs = n;
      for (l1b = l2b; l1b < l1bs; l1b += l1bLen) {
        is = l1a + l1aLen;
        if (is > m) is = m;
        for (i = l1a; i < is; ++i) {
          aOff = i * k;
          oOff = i * n;
          js = l1b + l1bLen;
          if (js > n) js = n;
          for (j = l1b; j < js; ++j) {
            bOff = j * k;
            O[oOff + j] = 0.0;
            for (l = 0; l < k; ++l) {
              O[oOff + j] += A[aOff + l] * B[bOff + l];
            }
          }
        }
      }
    }
  }
}

void vm3(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;

  int64_t l1aLen = tile_sizes[0];
  int64_t l1bLen = tile_sizes[1];
  int64_t l1cLen = tile_sizes[2];
  int64_t l1a;
  int64_t l1b;
  int64_t l1c;
  int64_t is, js, ls;
  for (l1a = start; l1a < end; l1a += l1aLen) {
    is = l1a + l1aLen;
    if (is > end) is = end;
    for (l1b = 0; l1b < n; l1b += l1bLen) {
      js = l1b + l1bLen;
      if (js > n) js = n;

      // This is the zeroing out of the tiled reduce's initial value.
      for (i = l1a; i < is; ++i) {
        for (j = l1b; j < js; ++j) {
          O[i*n+j] = 0.0;
        }
      }
      for (l1c = 0; l1c < k; l1c += l1cLen) {
        ls = l1c + l1cLen;
        if (ls > k) ls = k;
        for (i = l1a; i < is; ++i) {
          aOff = i * k;
          oOff = i * n;
          for (j = l1b; j < js; ++j) {
            bOff = j * k;
            for (l = l1c; l < ls; ++l) {
              O[oOff + j] += A[aOff + l] * B[bOff + l];
            }
          }
        }
      }
    }
  }
}

void vm3_unrolled(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;

  int64_t l1aLen = tile_sizes[0];
  int64_t l1bLen = tile_sizes[1];
  int64_t l1cLen = tile_sizes[2];
  int64_t l1a;
  int64_t l1b;
  int64_t l1c;
  int64_t is, js, ls;
  for (l1a = start; l1a < end; l1a += l1aLen) {
    is = l1a + l1aLen;
    if (is > end) is = end;
    for (l1b = 0; l1b < n; l1b += l1bLen) {
      js = l1b + l1bLen;
      if (js > n) js = n;

      // This is the zeroing out of the tiled reduce's initial value.
      for (i = l1a; i < is; ++i) {
        for (j = l1b; j < js; ++j) {
          O[i*n+j] = 0.0;
        }
      }
      for (l1c = 0; l1c < k; l1c += l1cLen) {
        ls = l1c + l1cLen;
        if (ls > k) ls = k;
        for (i = l1a; i < is; ++i) {
          aOff = i * k;
          oOff = i * n;
          for (j = l1b; j < js; ++j) {
            bOff = j * k;
            double out = 0.0;
            for (l = l1c; l < ls-4; l += 5) {
              out += A[aOff + l] * B[bOff + l];
              out += A[aOff + l + 1] * B[bOff + l + 1];
              out += A[aOff + l + 2] * B[bOff + l + 2];
              out += A[aOff + l + 3] * B[bOff + l + 3];
              out += A[aOff + l + 4] * B[bOff + l + 4];
              //out += A[aOff + l + 5] * B[bOff + l + 5];
              //out += A[aOff + l + 6] * B[bOff + l + 6];
              //out += A[aOff + l + 7] * B[bOff + l + 7];
            }
            for (; l < ls; ++l) {
              out += A[aOff + l] * B[bOff + l];
            }
            O[oOff + j] += out;
          }
        }
      }
    }
  }
}

void vm3_double_unrolled(int64_t start, int64_t end, void *args,
                         int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;

  int64_t l1aLen = tile_sizes[0];
  int64_t l1bLen = tile_sizes[1];
  int64_t l1cLen = tile_sizes[2];
  int64_t l1a;
  int64_t l1b;
  int64_t l1c;
  int64_t is, js, ls;
  for (l1a = start; l1a < end; l1a += l1aLen) {
    is = l1a + l1aLen;
    if (is > end) is = end;
    for (l1b = 0; l1b < n; l1b += l1bLen) {
      js = l1b + l1bLen;
      if (js > n) js = n;

      // This is the zeroing out of the tiled reduce's initial value.
      for (i = l1a; i < is; ++i) {
        for (j = l1b; j < js; ++j) {
          O[i*n+j] = 0.0;
        }
      }
      int64_t l1cLen2 = 2*l1cLen;
      for (l1c = 0; l1c < k; l1c += l1cLen2) {
        ls = l1c + l1cLen;
        if (ls > k) ls = k;
        for (i = l1a; i < is; ++i) {
          aOff = i * k;
          oOff = i * n;
          for (j = l1b; j < js; ++j) {
            bOff = j * k;
            double out = 0.0;
            for (l = l1c; l < ls; l += 5) {
              out += A[aOff + l] * B[bOff + l];
              out += A[aOff + l + 1] * B[bOff + l + 1];
              out += A[aOff + l + 2] * B[bOff + l + 2];
              out += A[aOff + l + 3] * B[bOff + l + 3];
              out += A[aOff + l + 4] * B[bOff + l + 4];
            }
            if (l > ls) {
              for (l = l-5; l < k; ++l) {
                out += A[aOff + l] * B[bOff + l];
              }
            }
            O[oOff + j] += out;
          }
        }
        ls = l1c + l1cLen2;
        if (ls > k) ls = k;
        for (i = l1a; i < is; ++i) {
          aOff = i * k;
          oOff = i * n;
          for (j = l1b; j < js; ++j) {
            bOff = j * k;
            double out = 0.0;
            for (l = l1c+l1cLen; l < ls; l += 5) {
              out += A[aOff + l] * B[bOff + l];
              out += A[aOff + l + 1] * B[bOff + l + 1];
              out += A[aOff + l + 2] * B[bOff + l + 2];
              out += A[aOff + l + 3] * B[bOff + l + 3];
              out += A[aOff + l + 4] * B[bOff + l + 4];
            }
            if (l > ls) {
              for (l = l-5; l < k; ++l) {
                out += A[aOff + l] * B[bOff + l];
              }
            }
            O[oOff + j] += out;
          }
        }
      }
    }
  }
}
void vm3_unrolled2(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;

  int64_t l1aLen = tile_sizes[0];
  int64_t l1bLen = tile_sizes[1];
  int64_t l1a;
  int64_t l1b;
  int64_t is, js;
  for (l1a = start; l1a < end; l1a += l1aLen) {
    for (l1b = 0; l1b < n; l1b += l1bLen) {
      is = l1a + l1aLen;
      if (is > end) is = end;
      for (i = l1a; i < is; ++i) {
        aOff = i * k;
        oOff = i * n;
        js = l1b + l1bLen;
        if (js > n) js = n;
        for (j = l1b; j < js; ++j) {
          bOff = j * k;
          double out = 0.0;
          for (l = 0; l < k-4; l += 5) {
            out += A[aOff + l] * B[bOff + l];
            out += A[aOff + l + 1] * B[bOff + l + 1];
            out += A[aOff + l + 2] * B[bOff + l + 2];
            out += A[aOff + l + 3] * B[bOff + l + 3];
            out += A[aOff + l + 4] * B[bOff + l + 4];
          }
          for (; l < k; ++l) {
            out += A[aOff + l] * B[bOff + l];
          }
          O[oOff + j] = out;
        }
      }
    }
  }
}

// mu = 6, nu = 1, no unrolling of ku loop
void vm4(int start, int end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int aOff, bOff, oOff;

  int l1bLen = tile_sizes[0];
  int l1cLen = tile_sizes[1];
  int ku = tile_sizes[2];
  int j, k, i2, j2, k2, k3;
  int it, jt;
  for (j = 0; j < n; j += l1bLen) {
    int j2End = min(j + l1bLen, n);
    for (it = start; it < end; ++it) {
      for (jt = j; jt < j2End; ++jt) {
        O[it*n + jt] = 0.0;
      }
    }
    for (k = 0; k < kLen; k += l1cLen) {
      int k2End = min(k + l1cLen, kLen);
      for (i2 = start; i2 < end; ++i2) {
        aOff = i2*kLen;
        oOff = i2*n;
        for (j2 = j; j2 < j2End; j2 += 6) {
          bOff = j2*kLen;
          double c0, c1, c2, c3, c4, c5;
          c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
          for (k3 = k; k3 < k2End; ++k3) {
            double a0;
            double b0, b1, b2, b3, b4, b5;
//            int k3End = min(k2 + ku, kLen);
//            for (k3 = k2; k3 < k3End; ++k3) {
              a0 = A[aOff + k3];
              b0 = B[bOff + k3];
              b1 = B[bOff + kLen + k3];
              b2 = B[bOff + 2 * kLen + k3];
              b3 = B[bOff + 3 * kLen + k3];
              b4 = B[bOff + 4 * kLen + k3];
              b5 = B[bOff + 5 * kLen + k3];
              c0 += a0 * b0;
              c1 += a0 * b1;
              c2 += a0 * b2;
              c3 += a0 * b3;
              c4 += a0 * b4;
              c5 += a0 * b5;
//            }
          }
          O[oOff + j2] += c0;
          O[oOff + j2 + 1] += c1;
          O[oOff + j2 + 2] += c2;
          O[oOff + j2 + 3] += c3;
          O[oOff + j2 + 4] += c4;
          O[oOff + j2 + 5] += c5;
        }
      }
    }
  }
}

// mu = 6, nu = 1, no unrolling of ku loop
void vm5(int start, int end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int aOff, bOff, oOff;

  int l1bLen = tile_sizes[0];
  int l1cLen = tile_sizes[1];
  int ku = tile_sizes[2];
  int j, k, i2, j2, k2, k3;
  int it, jt;
  for (j = 0; j < n; j += l1bLen) {
    int j2End = min(j + l1bLen, n);
    for (i2 = start; i2 < end; ++i2) {
      aOff = i2*kLen;
      oOff = i2*n;
      for (j2 = j; j2 < j2End; j2 += 6) {
        bOff = j2*kLen;
        double c0, c1, c2, c3, c4, c5;
        c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
        for (k3 = 0; k3 < kLen; ++k3) {
          double a0;
          double b0, b1, b2, b3, b4, b5;
          a0 = A[aOff + k3];
          b0 = B[bOff + k3];
          b1 = B[bOff + kLen + k3];
          b2 = B[bOff + 2 * kLen + k3];
          b3 = B[bOff + 3 * kLen + k3];
          b4 = B[bOff + 4 * kLen + k3];
          b5 = B[bOff + 5 * kLen + k3];
          c0 += a0 * b0;
          c1 += a0 * b1;
          c2 += a0 * b2;
          c3 += a0 * b3;
          c4 += a0 * b4;
          c5 += a0 * b5;
        }
        O[oOff + j2] = c0;
        O[oOff + j2 + 1] = c1;
        O[oOff + j2 + 2] = c2;
        O[oOff + j2 + 3] = c3;
        O[oOff + j2 + 4] = c4;
        O[oOff + j2 + 5] = c5;
      }
    }
  }
}

void vm_a1_b5_k0(int start, int end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int aOff, bOff, oOff;

  int l1bLen = tile_sizes[0];
  int l1cLen = tile_sizes[1];

  // A L1 tile is implicit as the start/end of the chunk.
  int j;
  for (j = 0; j < n; j += l1bLen) {
    int j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int k;
    for (k = 0; k < kLen; k += l1cLen) {
      int k3End = min(k + l1cLen, kLen);
      int i2;
      // A's reg tile size set to 1.
      for (i2 = start; i2 < end; ++i2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int j2;
        // B's reg tile size set to 5.
        for (j2 = j; j2 < j2End - 4; j2 += 5) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4;
          c0 = c1 = c2 = c3 = c4 = 0.0;
          int k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            double b4 = Brow[4*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
            c3 = c3 + (a0 * b3);
            c4 = c4 + (a0 * b4);
          }
          Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
          Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
          Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
          Ocol[0*n + 3] = Ocol[0*n + 3] + c3;
          Ocol[0*n + 4] = Ocol[0*n + 4] + c4;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0;
          c0 = 0.0;
          int k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
          }
          Ocol[0*n + 0] += c0;
        }
      }
    }
  }
}

void vm_a1_b6_k0(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int64_t aOff, bOff, oOff;

  int64_t l1bLen = tile_sizes[1];
  int64_t l1cLen = tile_sizes[2];

  // A L1 tile is implicit as the start/end of the chunk.
  int64_t j;
  for (j = 0; j < n; j += l1bLen) {
    int64_t j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int64_t jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int64_t k;
    for (k = 0; k < kLen; k += l1cLen) {
      int64_t k3End = min(k + l1cLen, kLen);
      int64_t i2;
      // A's reg tile size set to 1.
      for (i2 = start; i2 < end; ++i2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 6.
        for (j2 = j; j2 < j2End - 5; j2 += 6) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5;
          c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            double b4 = Brow[4*kLen + k3];
            double b5 = Brow[5*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
            c3 = c3 + (a0 * b3);
            c4 = c4 + (a0 * b4);
            c5 = c5 + (a0 * b5);
          }
          Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
          Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
          Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
          Ocol[0*n + 3] = Ocol[0*n + 3] + c3;
          Ocol[0*n + 4] = Ocol[0*n + 4] + c4;
          Ocol[0*n + 5] = Ocol[0*n + 5] + c5;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0;
          c0 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
          }
          Ocol[0*n + 0] += c0;
        }
      }
    }
  }
}

void vm_a2_b3_k0(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int64_t aOff, bOff, oOff;

  int64_t l1bLen = tile_sizes[0];
  int64_t l1cLen = tile_sizes[1];

  // A L1 tile is implicit as the start/end of the chunk.
  int64_t j;
  for (j = 0; j < n; j += l1bLen) {
    int64_t j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int64_t jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int64_t k;
    for (k = 0; k < kLen; k += l1cLen) {
      int64_t k3End = min(k + l1cLen, kLen);
      int64_t i2;
      // A's reg tile size set to 2.
      for (i2 = start; i2 < end - 1; i2 += 2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 3.
        for (j2 = j; j2 < j2End - 2; j2 += 3) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5;
          c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
            c3 = c3 + (a1 * b0);
            c4 = c4 + (a1 * b1);
            c5 = c5 + (a1 * b2);
          }
          Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
          Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
          Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
          Ocol[1*n + 0] = Ocol[1*n + 0] + c3;
          Ocol[1*n + 1] = Ocol[1*n + 1] + c4;
          Ocol[1*n + 2] = Ocol[1*n + 2] + c5;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1;
          c0 = c1 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
            c1 += a1 * b0;
          }
          Ocol[0*n + 0] += c0;
          Ocol[1*n + 0] += c1;
        }
      }
      for (; i2 < end; ++i2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 3.
        for (j2 = j; j2 < j2End - 2; j2 += 3) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2;
          c0 = c1 = c2 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
          }
          Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
          Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
          Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0;
          c0 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
          }
          Ocol[0*n + 0] += c0;
        }
      }
    }
  }
}

void vm_a2_b4_k0(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int64_t aOff, bOff, oOff;

  int64_t l1bLen = tile_sizes[0];
  int64_t l1cLen = tile_sizes[1];

  // A L1 tile is implicit as the start/end of the chunk.
  int64_t j;
  for (j = 0; j < n; j += l1bLen) {
    int64_t j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int64_t jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int64_t k;
    for (k = 0; k < kLen; k += l1cLen) {
      int64_t k3End = min(k + l1cLen, kLen);
      int64_t i2;
      // A's reg tile size set to 2.
      for (i2 = start; i2 < end - 1; i2 += 2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 3.
        for (j2 = j; j2 < j2End - 3; j2 += 4) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5, c6, c7;
          c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            c0 += (a0 * b0);
            c1 += (a0 * b1);
            c2 += (a0 * b2);
            c3 += (a0 * b3);
            c4 += (a1 * b0);
            c5 += (a1 * b1);
            c6 += (a1 * b2);
            c7 += (a1 * b3);
          }
          Ocol[0*n + 0] += c0;
          Ocol[0*n + 1] += c1;
          Ocol[0*n + 2] += c2;
          Ocol[0*n + 3] += c3;
          Ocol[1*n + 0] += c4;
          Ocol[1*n + 1] += c5;
          Ocol[1*n + 2] += c6;
          Ocol[1*n + 3] += c7;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1;
          c0 = c1 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
            c1 += a1 * b0;
          }
          Ocol[0*n + 0] += c0;
          Ocol[1*n + 0] += c1;
        }
      }
      for (; i2 < end; ++i2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 4.
        for (j2 = j; j2 < j2End - 3; j2 += 4) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3;
          c0 = c1 = c2 = c3 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            c0 += (a0 * b0);
            c1 += (a0 * b1);
            c2 += (a0 * b2);
            c3 += (a0 * b3);
          }
          Ocol[0*n + 0] += c0;
          Ocol[0*n + 1] += c1;
          Ocol[0*n + 2] += c2;
          Ocol[0*n + 3] += c3;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0;
          c0 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
          }
          Ocol[0*n + 0] += c0;
        }
      }
    }
  }
}

void vm_a4_b4_k0(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int64_t aOff, bOff, oOff;

  int64_t l1bLen = tile_sizes[0];
  int64_t l1cLen = tile_sizes[1];

  // A L1 tile is implicit as the start/end of the chunk.
  int64_t j;
  for (j = 0; j < n; j += l1bLen) {
    int64_t j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int64_t jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int64_t k;
    for (k = 0; k < kLen; k += l1cLen) {
      int64_t k3End = min(k + l1cLen, kLen);
      int64_t i2;
      // A's reg tile size set to 4.
      for (i2 = start; i2 < end - 3; i2 += 4) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 4.
        for (j2 = j; j2 < j2End - 3; j2 += 4) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5, c6, c7;
          double c8, c9, c10, c11, c12, c13, c14, c15;
          c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0.0;
          c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double a2 = Arow[2*kLen + k3];
            double a3 = Arow[3*kLen + k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            c0  += (a0 * b0);
            c1  += (a0 * b1);
            c2  += (a0 * b2);
            c3  += (a0 * b3);
            c4  += (a1 * b0);
            c5  += (a1 * b1);
            c6  += (a1 * b2);
            c7  += (a1 * b3);
            c8  += (a2 * b0);
            c9  += (a2 * b1);
            c10 += (a2 * b2);
            c11 += (a2 * b3);
            c12 += (a3 * b0);
            c13 += (a3 * b1);
            c14 += (a3 * b2);
            c15 += (a3 * b3);
          }
          Ocol[0*n + 0] += c0;
          Ocol[0*n + 1] += c1;
          Ocol[0*n + 2] += c2;
          Ocol[0*n + 3] += c3;
          Ocol[1*n + 0] += c4;
          Ocol[1*n + 1] += c5;
          Ocol[1*n + 2] += c6;
          Ocol[1*n + 3] += c7;
          Ocol[2*n + 0] += c8;
          Ocol[2*n + 1] += c9;
          Ocol[2*n + 2] += c10;
          Ocol[2*n + 3] += c11;
          Ocol[3*n + 0] += c12;
          Ocol[3*n + 1] += c13;
          Ocol[3*n + 2] += c14;
          Ocol[3*n + 3] += c15;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3;
          c0 = c1 = c2 = c3 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double a2 = Arow[2*kLen + k3];
            double a3 = Arow[3*kLen + k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
            c1 += a1 * b0;
            c2 += a2 * b0;
            c3 += a3 * b0;
          }
          Ocol[0*n + 0] += c0;
          Ocol[1*n + 0] += c1;
          Ocol[2*n + 0] += c1;
          Ocol[3*n + 0] += c1;
        }
      }
      for (; i2 < end; ++i2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 4.
        for (j2 = j; j2 < j2End - 3; j2 += 4) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3;
          c0 = c1 = c2 = c3 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            c0 += (a0 * b0);
            c1 += (a0 * b1);
            c2 += (a0 * b2);
            c3 += (a0 * b3);
          }
          Ocol[0*n + 0] += c0;
          Ocol[0*n + 1] += c1;
          Ocol[0*n + 2] += c2;
          Ocol[0*n + 3] += c3;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0;
          c0 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
          }
          Ocol[0*n + 0] += c0;
        }
      }
    }
  }
}

void vm_a2_b6_k0(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int kLen = my_args->k;
  int64_t aOff, bOff, oOff;

  int64_t l1bLen = tile_sizes[0];
  int64_t l1cLen = tile_sizes[1];

  // A L1 tile is implicit as the start/end of the chunk.
  int64_t j;
  for (j = 0; j < n; j += l1bLen) {
    int64_t j2End = min(j + l1bLen, n);
    double *Btile = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile = O + it*n;
      int64_t jt;
      for (jt = j; jt < j2End; ++jt) {
        Otile[jt] = 0.0;
      }
    }
    int64_t k;
    for (k = 0; k < kLen; k += l1cLen) {
      int64_t k3End = min(k + l1cLen, kLen);
      int64_t i2;
      // A's reg tile size set to 2.
      for (i2 = start; i2 < end; i2 += 2) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 6.
        for (j2 = j; j2 < j2End; j2 += 6) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5;
          double c6, c7, c8, c9, c10, c11;
          c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
          c6 = c7 = c8 = c9 = c10 = c11 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            double b3 = Brow[3*kLen + k3];
            double b4 = Brow[4*kLen + k3];
            double b5 = Brow[5*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
            c3 = c3 + (a0 * b3);
            c4 = c4 + (a0 * b4);
            c5 = c5 + (a0 * b5);
            c6 = c6 + (a1 * b0);
            c7 = c7 + (a1 * b1);
            c8 = c8 + (a1 * b2);
            c9 = c9 + (a1 * b3);
            c10 = c10 + (a1 * b4);
            c11 = c11 + (a1 * b5);
          }
          Ocol[0*n + 0] += c0;
          Ocol[0*n + 1] += c1;
          Ocol[0*n + 2] += c2;
          Ocol[0*n + 3] += c3;
          Ocol[0*n + 4] += c4;
          Ocol[0*n + 5] += c5;
          Ocol[1*n + 0] += c6;
          Ocol[1*n + 1] += c7;
          Ocol[1*n + 2] += c8;
          Ocol[1*n + 3] += c9;
          Ocol[1*n + 4] += c10;
          Ocol[1*n + 5] += c11;
        }
        // Cleanup stragglers of j2 register tile
        if (j2 != j2End) {
          for (j2 = j2 - 5; j2 < j2End; ++j2) {
            double *Brow = B + j2*kLen;
            double *Ocol = Orow + j2;
            double c0, c1;
            c0 = c1 = 0.0;
            int64_t k3;
            for (k3 = k; k3 < k3End; ++k3) {
              double a0 = Arow[k3];
              double a1 = Arow[kLen + k3];
              double b0 = Brow[k3];
              c0 += a0 * b0;
              c1 += a1 * b0;
            }
            Ocol[0*n + 0] += c0;
            Ocol[1*n + 0] += c1;
          }
        }
      }
      // Cleanup stragglers of i2 register tile
      if (i2 != end) {
        for (i2 = i2 - 2; i2 < end; ++i2) {
          double *Arow = A + i2*kLen;
          double *Orow = O + i2*n;
          int64_t j2;
          // B's reg tile size set to 6.
          for (j2 = j; j2 < j2End; j2 += 6) {
            double *Brow = B + j2*kLen;
            double *Ocol = Orow + j2;
            double c0, c1, c2, c3, c4, c5;
            c0 = c1 = c2 = c3 = c4 = c5 = 0.0;
            int64_t k3;
            for (k3 = k; k3 < k3End; ++k3) {
              double a0 = Arow[k3];
              double b0 = Brow[k3];
              double b1 = Brow[kLen + k3];
              double b2 = Brow[2*kLen + k3];
              double b3 = Brow[3*kLen + k3];
              double b4 = Brow[4*kLen + k3];
              double b5 = Brow[5*kLen + k3];
              c0 = c0 + (a0 * b0);
              c1 = c1 + (a0 * b1);
              c2 = c2 + (a0 * b2);
              c3 = c3 + (a0 * b3);
              c4 = c4 + (a0 * b4);
              c5 = c5 + (a0 * b5);
            }
            Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
            Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
            Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
            Ocol[0*n + 3] = Ocol[0*n + 3] + c3;
            Ocol[0*n + 4] = Ocol[0*n + 4] + c4;
            Ocol[0*n + 5] = Ocol[0*n + 5] + c5;
          }
          // Cleanup stragglers of j2 register tile
          if (j2 != j2End) {
            for (j2 = j2 - 6; j2 < j2End; ++j2) {
              double *Brow = B + j2*kLen;
              double *Ocol = Orow + j2;
              double c0, c1;
              c0 = c1 = 0.0;
              int64_t k3;
              for (k3 = k; k3 < k3End; ++k3) {
                double a0 = Arow[k3];
                double a1 = Arow[kLen + k3];
                double b0 = Brow[k3];
                c0 += a0 * b0;
                c1 += a1 * b0;
              }
              Ocol[0*n + 0] += c0;
              Ocol[1*n + 0] += c1;
            }
          }
        }
      }
    }
  }
}

void vm_untiled(int64_t start, int64_t end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;

  for (i = start; i < end; ++i) {
    aOff = i * k;
    oOff = i * n;
    for (j = 0; j < n; ++j) {
      bOff = j * k;
      O[oOff + j] = 0.0;
      for (l = 0; l < k; ++l) {
        O[oOff + j] += A[aOff + l] * B[bOff + l];
      }
    }
  }
}

void vm_just_unrolled(int64_t start, int64_t end, void *args, int *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A = my_args->a;
  double *B = my_args->b;
  double *O = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int64_t i, j, l;
  int64_t aOff, bOff, oOff;
  int64_t ls;

  for (i = start; i < end; ++i) {
    aOff = i * k;
    oOff = i * n;
    for (j = 0; j < n; ++j) {
      bOff = j * k;
      double out = 0.0;
      for (l = 0; l < k-5; l += 6) {
        out += A[aOff + l] * B[bOff + l];
        out += A[aOff + l + 1] * B[bOff + l + 1];
        out += A[aOff + l + 2] * B[bOff + l + 2];
        out += A[aOff + l + 3] * B[bOff + l + 3];
        out += A[aOff + l + 4] * B[bOff + l + 4];
        out += A[aOff + l + 5] * B[bOff + l + 5];
      }
      for (; l < k; ++l) {
        out += A[aOff + l] * B[bOff + l];
      }
      O[oOff + j] = out;
    }
  }
}

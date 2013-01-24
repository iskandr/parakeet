#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define aligned __attribute__((aligned(16)))

typedef struct {
  double *a;
  double *b;
  double *out;
  int     m;
  int     n;
  int     k;
} vm_args_t;

double *make_array(int m, int n) {
  double *array;
  posix_memalign((void**)(&array), 16, m * n * sizeof(double));
  int i;
  for (i = 0; i < m * n; ++i) {
    array[i] = ((double)m) / n;
  }
  return array;
}

void free_array(double *array) {
  free(array);
}

static inline int min(a, b) {
  return a < b ? a : b;
}

void vm(int64_t start, int64_t end, void *args, int *tile_sizes) {
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

void vm_unrolled(int64_t start, int64_t end, void *args, int *tile_sizes) {
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

void vm_tiled(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
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

void vm_tiled_unrolled(int64_t start, int64_t end,
                       void *args, int64_t *tile_sizes) {
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


void vm_a3_b3_k1(int64_t start, int64_t end, void *args, int64_t *tile_sizes) {
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
      // A's reg tile size set to 3.
      for (i2 = start; i2 < end - 2; i2 += 3) {
        double *Arow = A + i2*kLen;
        double *Orow = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 3.
        for (j2 = j; j2 < j2End - 2; j2 += 3) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2, c3, c4, c5, c6, c7, c8;
          c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double a2 = Arow[2*kLen + k3];
            double b0 = Brow[k3];
            double b1 = Brow[kLen + k3];
            double b2 = Brow[2*kLen + k3];
            c0 = c0 + (a0 * b0);
            c1 = c1 + (a0 * b1);
            c2 = c2 + (a0 * b2);
            c3 = c3 + (a1 * b0);
            c4 = c4 + (a1 * b1);
            c5 = c5 + (a1 * b2);
            c6 = c6 + (a2 * b0);
            c7 = c7 + (a2 * b1);
            c8 = c8 + (a2 * b2);
          }
          Ocol[0*n + 0] = Ocol[0*n + 0] + c0;
          Ocol[0*n + 1] = Ocol[0*n + 1] + c1;
          Ocol[0*n + 2] = Ocol[0*n + 2] + c2;
          Ocol[1*n + 0] = Ocol[1*n + 0] + c3;
          Ocol[1*n + 1] = Ocol[1*n + 1] + c4;
          Ocol[1*n + 2] = Ocol[1*n + 2] + c5;
          Ocol[2*n + 0] = Ocol[2*n + 0] + c6;
          Ocol[2*n + 1] = Ocol[2*n + 1] + c7;
          Ocol[2*n + 2] = Ocol[2*n + 2] + c8;
        }
        // Cleanup stragglers of j2 register tile
        for (; j2 < j2End; ++j2) {
          double *Brow = B + j2*kLen;
          double *Ocol = Orow + j2;
          double c0, c1, c2;
          c0 = c1 = c2 = 0.0;
          int64_t k3;
          for (k3 = k; k3 < k3End; ++k3) {
            double a0 = Arow[k3];
            double a1 = Arow[kLen + k3];
            double a2 = Arow[2*kLen + k3];
            double b0 = Brow[k3];
            c0 += a0 * b0;
            c1 += a1 * b0;
            c2 += a2 * b0;
          }
          Ocol[0*n + 0] += c0;
          Ocol[1*n + 0] += c1;
          Ocol[2*n + 0] += c2;          
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
          double *Brow aligned = B + j2*kLen;
          double *Ocol aligned = Orow + j2;
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

void vm_a4_b4_k0_aligned(int64_t start, int64_t end, void *args,
                         int64_t *tile_sizes) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *A aligned = my_args->a;
  double *B aligned = my_args->b;
  double *O aligned = my_args->out;
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
    double *Btile aligned = B + j*kLen;
    int64_t it;
    for (it = start; it < end; ++it) {
      double *Otile aligned = O + it*n;
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
        double *Arow aligned = A + i2*kLen;
        double *Orow aligned = O + i2*n;
        int64_t j2;
        // B's reg tile size set to 4.
        for (j2 = j; j2 < j2End - 3; j2 += 4) {
          double *Brow aligned = B + j2*kLen;
          double *Ocol aligned = Orow + j2;
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
          double *Brow aligned = B + j2*kLen;
          double *Ocol aligned = Orow + j2;
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

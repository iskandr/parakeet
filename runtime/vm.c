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
  double *a = my_args->a;
  double *b = my_args->b;
  double *out = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;

  int i, j, it, jt, iter;
  double sum;
  int ts0, ts1;
  ts0 = tile_sizes[0];
  ts1 = tile_sizes[1];
  for (iter = start; iter <= end; ++iter) {
    for (it = 0; it < n; it += ts0) {
      for (i = it; i < it + ts0 && i < n; ++i) {
        sum = 0.0;
        for (jt = 0; jt < k; jt += ts1) {
          for (j = jt; j < jt + ts1 && j < k; ++j) {
            sum += a[iter*k + j] * b[i*k + j];
          }
        }
        out[iter*n + i] = sum;
      }
    }
  }
}

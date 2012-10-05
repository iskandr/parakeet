#include <assert.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "job.h"
#include "thread_pool.h"

typedef struct {
  double *a;
  double *b;
  double *out;
  int     m;
  int     n;
  int     k;
} vm_args_t;

inline int safe_div(int n, int d) {
  return n / d + (n % d ? 1 : 0);
}

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

void cache_blocking_mm(double *A, double *B, double *O,
                       int m, int n, int k, int oStride,
                       int l3aLen, int l3bLen,
                       int l2aLen, int l2bLen,
                       int l1aLen, int l1bLen) {
  int i, j, l;
  int aOff, bOff, oOff;

  int l3a, l3b, l2a, l2b, l2as, l2bs, l1a, l1b, l1as, l1bs;
  int is, js;
  for (l3a = 0; l3a < m; l3a += l3aLen) {
    for (l3b = 0; l3b < n; l3b += l3bLen) {
      l2as = l3a + l3aLen;
      if (l2as > m) l2as = m;
      for (l2a = l3a; l2a < l2as; l2a += l2aLen) {
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
                oOff = i * oStride;
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
}

void test_mm(void) {
  int max_threads = 8;
  thread_pool_t *thread_pool = create_thread_pool(max_threads);

  int m = 10000;
  int n = 800;
  int k = 800;
  double *a = make_array(m, k);
  double *b = make_array(k, n);
  double *o = make_array(m, n);
  int tile_sizes[2] = {32, 32};

  int num_threads = 8;
  job_t *job = make_job(0, m, num_threads, 128);

  vm_args_t vm_args;
  vm_args.a = a;
  vm_args.b = b;
  vm_args.out = o;
  vm_args.m = m;
  vm_args.n = n;
  vm_args.k = k;

  launch_job(thread_pool, &vm, &vm_args, job, tile_sizes, 0);
  wait_for_job(thread_pool);

  int pass = 1;
  int i, j, l;
  double sum;
  struct timeval start, end, result;
  gettimeofday(&start, NULL);
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      sum = 0.0;
      for (l = 0; l < k; ++l) {
        sum += a[i*k + l] * b[j*k + l];
      }
      pass = pass && (abs(sum - o[i*n + j]) < 1e-4);
    }
  }
  gettimeofday(&end, NULL);
  CU_ASSERT(pass);

//  double naive_time;
//  timersub(&end, &start, &result);
//  naive_time = result.tv_sec + result.tv_usec / 1000000.0;
//  printf("Naive time: %f\n", naive_time);

  destroy_thread_pool(thread_pool);
  free_job(job);
  free_array(a);
  free_array(b);
  free_array(o);
}

int init_suite1(void) {
  return 0;
}

int clean_suite1(void) {
  return 0;
}

int main(int argc, char **argv) {
  CU_pSuite pSuite = NULL;

  if (CUE_SUCCESS != CU_initialize_registry()) {
    return CU_get_error();
  }

  pSuite = CU_add_suite("Runtime Tests", init_suite1, clean_suite1);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  if ((NULL == CU_add_test(pSuite, "MM", test_mm))) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}

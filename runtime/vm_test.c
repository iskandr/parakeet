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
  int    *tile_sizes;
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

void vm(int iter, void *args) {
  vm_args_t *my_args = (vm_args_t*)args;
  double *a = my_args->a;
  double *b = my_args->b;
  double *out = my_args->out;
  int m = my_args->m;
  int n = my_args->n;
  int k = my_args->k;
  int *tile_sizes = my_args->tile_sizes;

  int i, j, it, jt, itend, jtend;
  double sum;
  int ts0, ts1;
  ts0 = tile_sizes[0];
  ts1 = tile_sizes[1];
  itend = safe_div(n, ts0);
  jtend = safe_div(k, ts1);
  for (it = 0; it < itend; it += ts0) {
    for (i = it; i < it + ts0 && i < n; ++i) {
      sum = 0.0;
      for (jt = 0; jt < jtend; jt += ts1) {
        for (j = jt; j < jt + ts1 && j < k; ++j) {
          sum += a[iter*k + j] * b[i*k + j];
        }
        out[iter*n + i] = sum;
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
  int tile_sizes[2] = {1, 1};

  int num_threads = 8;
  job_t *job = make_job(0, m, num_threads);
  
  vm_args_t vm_args;
  vm_args.a = a;
  vm_args.b = b;
  vm_args.out = o;
  vm_args.m = m;
  vm_args.n = n;
  vm_args.k = k;
  vm_args.tile_sizes = tile_sizes;
  
  launch_job(thread_pool, &vm, &vm_args, job, 0);
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
    }
  }
  gettimeofday(&end, NULL);
  CU_ASSERT(pass);

  double naive_time;
  timersub(&end, &start, &result);
  naive_time = result.tv_sec + result.tv_usec / 1000000.0;
  printf("Naive time: %f\n", naive_time);
  
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

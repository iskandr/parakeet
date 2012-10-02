#include <assert.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "job.h"
#include "runtime.h"
#include "thread_pool.h"

typedef struct {
  int *in;
  int *out;
} add1_args_t;

void add1(int iter, void *args) {
  add1_args_t *my_args = (add1_args_t*)args;
  
  my_args->out[iter] = my_args->in[iter] + 1;
}

void test_monitor_job(void) {
  int len = 200000000;
  int *in = (int*)malloc(sizeof(int) * len);
  int *out = (int*)malloc(sizeof(int) * len);
  int i;
  for (i = 0; i < len; ++i) {
    in[i] = i;
  }

  add1_args_t add1_args;
  add1_args.in = in;
  add1_args.out = out;

  int max_threads = 8;
  runtime_t *runtime = create_runtime(max_threads);

  struct timeval start, end, result;
  gettimeofday(&start, NULL);
  run_job(runtime, &add1, &add1_args, len);
  gettimeofday(&end, NULL);
  timersub(&end, &start, &result);
  double rt = result.tv_sec + result.tv_usec / 1000000.0;

  destroy_runtime(runtime);

  int pass = 1;
  for (i = 0; i < len; ++i) {
    pass &= out[i] == in[i] + 1;
  }
  CU_ASSERT(pass);

  printf("Total runtime: %f secs\n", rt);
  free(in);
  free(out);
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
  
  if ((NULL == CU_add_test(pSuite, "Monitor Job", test_monitor_job))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}

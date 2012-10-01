#include <assert.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"
#include "thread_pool.h"

typedef struct {
  int *in;
  int *out;
} add1_args_t;

void add1(int iter, void *args) {
  add1_args_t *my_args = (add1_args_t*)args;
  
  my_args->out[iter] = my_args->in[iter] + 1;
  if (iter > 500000 - 1)
    printf("Too big iter: %d\n", iter);
}

void test_create_destroy(void) {
  int num_threads = 1;
  thread_pool_t *thread_pool = create_thread_pool(num_threads);
  destroy_thread_pool(thread_pool);
  CU_ASSERT(1);
}

void test_run_threads(void) {
  int max_threads = 8;
  thread_pool_t *thread_pool = create_thread_pool(max_threads);
  
  int len = 5000;
  int *in = (int*)malloc(sizeof(int) * len);
  int *out = (int*)malloc(sizeof(int) * len);
  int i;
  for (i = 0; i < len; ++i) {
    in[i] = i;
  }

  int num_threads = 8;
  job_t *job = make_job(len, num_threads);
  
  add1_args_t add1_args;
  add1_args.in = in;
  add1_args.out = out;
  launch_job(thread_pool, &add1, &add1_args, job);
  wait_for_job(thread_pool);

  int pass = 1;
  for (i = 0; (i < len) && pass; ++i) {
    pass &= out[i] == in[i] + 1;
  }
  CU_ASSERT(pass);
  
  destroy_thread_pool(thread_pool);
  free_job(job);
  free(in);
  free(out);
}

void test_pause_threads(void) {
  int max_threads = 8;
  thread_pool_t *thread_pool = create_thread_pool(max_threads);
  int num_threads = max_threads;
  
  int len = 500000;
  int *in = (int*)malloc(sizeof(int) * len);
  int *out = (int*)malloc(sizeof(int) * len);
  int i;
  for (i = 0; i < len; ++i) {
    in[i] = i;
  }

  job_t *job = make_job(len, num_threads);

  add1_args_t add1_args;
  add1_args.in = in;
  add1_args.out = out;
  
  launch_job(thread_pool, &add1, &add1_args, job);
  pause_job(thread_pool);
  launch_job(thread_pool, &add1, &add1_args, job);
  wait_for_job(thread_pool);

  int pass = 1;
  for (i = 0; i < len && pass; ++i) {
    pass &= out[i] == in[i] + 1;
  }
  CU_ASSERT(pass);
  
  destroy_thread_pool(thread_pool);
  free_job(job);
  free(in);
  free(out);
}

void test_reconfigure_threads(void) {
  int max_threads = 8;
  thread_pool_t *thread_pool = create_thread_pool(max_threads);
  int num_threads = max_threads;
  
  int len = 500000;
  int *in = (int*)malloc(sizeof(int) * len);
  int *out = (int*)malloc(sizeof(int) * len);
  int i;
  for (i = 0; i < len; ++i) {
    in[i] = i;
  }

  job_t *job = make_job(len, num_threads);
  
  add1_args_t add1_args;
  add1_args.in = in;
  add1_args.out = out;
  
  launch_job(thread_pool, &add1, &add1_args, job); 
  pause_job(thread_pool);
  num_threads = 3;
  job = reconfigure_job(job, num_threads);
  launch_job(thread_pool, &add1, &add1_args, job);
  wait_for_job(thread_pool);

  int pass = 1;
  for (i = 0; i < len && pass; ++i) {
    pass &= out[i] == in[i] + 1;
  }
  CU_ASSERT(pass);
  
  destroy_thread_pool(thread_pool);
  free_job(job);
  free(in);
  free(out);
}

void test_sequence_of_jobs(void) {
  int max_threads = 8;
  thread_pool_t *thread_pool = create_thread_pool(max_threads);
  int num_threads = max_threads;
  
  int len = 100000;
  int *in = (int*)malloc(sizeof(int) * len);
  int *out = (int*)malloc(sizeof(int) * len);

  job_t *job;
  
  add1_args_t add1_args;
  add1_args.in = in;
  add1_args.out = out;

  int i;
  num_threads = 2;
  for (i = 0; i < 2; ++i) {
    job = make_job(len, num_threads);
    launch_job(thread_pool, &add1, &add1_args, job); 
    pause_job(thread_pool);
    num_threads++;
    job = reconfigure_job(job, num_threads);
    launch_job(thread_pool, &add1, &add1_args, job);
    wait_for_job(thread_pool);

    int pass = 1;
    int i;
    for (i = 0; i < len; ++i) {
      pass &= out[i] == in[i] + 1;
    }
    CU_ASSERT(pass);

    free_job(job);
  }
  
  destroy_thread_pool(thread_pool);
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

  /* initialize the CUnit test registry */
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();
  
  /* add a suite to the registry */
  pSuite = CU_add_suite("Runtime Tests", init_suite1, clean_suite1);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  if ((NULL == CU_add_test(pSuite, "Create & Destroy", test_create_destroy))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  /* add the tests to the suite */
  if ((NULL == CU_add_test(pSuite, "Run add1", test_run_threads))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  
  if ((NULL == CU_add_test(pSuite, "Pause tasks", test_pause_threads))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  
  if ((NULL == CU_add_test(pSuite, "Reconfigure tasks",
                           test_reconfigure_threads))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  
  if ((NULL == CU_add_test(pSuite, "Sequence of jobs",
                           test_sequence_of_jobs))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  
  /* Run all tests using the CUnit Basic interface */
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}

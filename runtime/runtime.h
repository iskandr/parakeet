#ifndef _RUNTIME_H_
#define _RUNTIME_H_

#include "thread_pool.h"

typedef struct {
  thread_pool_t    *thread_pool;
  // TODO: Add statement cache for found settings, other state for managing
  //       dynamic performance settings.
} runtime_t;

runtime_t *create_runtime(int max_threads);

void run_job(runtime_t *runtime,
             work_function_t work_function, void *args, int arg_len);

// Assumes that the runtime isn't running any jobs.
void destroy_runtime(runtime_t *runtime);

static double calibrate_par(runtime_t *runtime,
                            work_function_t work_function, void *args,
                            job_t *job);
static double get_seq_throughput(work_function_t work_function, void *args,
                                 int num_iters);
static double get_par_throughput(runtime_t *runtime,
                                 work_function_t work_function, void *args,
                                 job_t *job, int num_threads);
static int get_npar(int num_threads);

#endif // _RUNTIME_H_

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

#endif // _RUNTIME_H_

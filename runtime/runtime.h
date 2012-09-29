#ifndef _RUNTIME_H_
#define _RUNTIME_H_

#include <pthread.h>

#include "thread_pool.h"

job_t *make_job(int len, int max_threads, int num_threads);
// job_t *reconfigure_job(job_t *old_job, int num_threads);
void free_job(job_t *job);

#endif // _RUNTIME_H_

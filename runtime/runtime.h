#ifndef _RUNTIME_H_
#define _RUNTIME_H_

#include <pthread.h>

#include "thread_pool.h"

job_t *make_job(int len, int num_threads);
job_t *reconfigure_job(job_t *old_job, int num_threads);
int num_unfinished_tasks(job_t *job);
void free_job(job_t *job);

#endif // _RUNTIME_H_

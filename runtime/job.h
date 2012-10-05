#ifndef _JOB_H_
#define _JOB_H_

#include <pthread.h>

typedef struct {
  long            next_iteration;
  long            last_iteration;
} task_t;

typedef struct {
  task_t            *tasks;
  int                num_tasks;
  int                cur_task;
  pthread_barrier_t *barrier;
} task_list_t;

typedef struct {
  task_list_t      *task_lists;
  int               num_lists;
  pthread_barrier_t barrier;
} job_t;

job_t *make_job(int start, int stop, int num_threads, int chunk_len);
job_t *reconfigure_job(job_t *old_job, int num_threads);
int num_unfinished_tasks(job_t *job);
int num_threads(job_t *job);
void free_job(job_t *job);

#endif // _JOB_H_

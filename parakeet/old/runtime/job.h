#ifndef _JOB_H_
#define _JOB_H_

#include <pthread.h>
#include <stdint.h>

typedef struct {
  int64_t next_start;
  int64_t step;
  int64_t end;
} task_t;

typedef struct {
  task_t            *tasks;
  int64_t            num_tasks;
  int64_t            cur_task;
  pthread_barrier_t *barrier;
} task_list_t;

typedef struct {
  task_list_t      *task_lists;
  int               num_lists;
  pthread_barrier_t barrier;
} job_t;

job_t *make_job(int64_t start, int64_t stop, int64_t step, int num_threads,
                int64_t task_len);
job_t *reconfigure_job(job_t *old_job, int64_t step);
int num_threads(job_t *job);
void free_job(job_t *job);

#endif // _JOB_H_

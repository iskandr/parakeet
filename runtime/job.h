#ifndef _JOB_H_
#define _JOB_H_

#include <pthread.h>

typedef struct {
  int next_start;
  int step;
  int end;
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

job_t *make_job(int start, int stop, int step, int num_threads,
                int task_len);
job_t *reconfigure_job(job_t *old_job, int step);
int num_threads(job_t *job);
void free_job(job_t *job);

#endif // _JOB_H_

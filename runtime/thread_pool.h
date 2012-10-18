#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <pthread.h>

#include "job.h"

typedef void (*work_function_t)(int, int, void*, int*);

typedef enum {
  THREAD_RUN = 0,
  THREAD_FINISHED,
  THREAD_PAUSE,
  THREAD_IDLE,
  THREAD_STOP
} thread_status_t;

typedef struct {
  task_list_t       *task_list;
  pthread_mutex_t    mutex;
  pthread_cond_t     cond;
  thread_status_t    status;
  pthread_cond_t    *master_cond;
  int                notify_when_done;
  work_function_t    work_function;
  void              *args;
  int               *tile_sizes;
  int                iters_done;
  unsigned long long timestamp;
} worker_data_t;

typedef struct {
  pthread_t          *workers;
  int                 num_workers;
  int                 num_active;
  pthread_cond_t      master_cond;
  worker_data_t      *worker_data;
  int                *iters_done;
  unsigned long long *timestamps;
  job_t              *job;
} thread_pool_t;

thread_pool_t *create_thread_pool(int max_threads);
void launch_job(thread_pool_t *thread_pool,
                work_function_t work_function, void *args, job_t *job,
                int *tile_sizes, int reset_tps);
void pause_job(thread_pool_t *thread_pool);
int job_finished(thread_pool_t *thread_pool);
int get_iters_done(thread_pool_t *thread_pool);
double get_throughput(thread_pool_t *thread_pool);
void wait_for_job(thread_pool_t *thread_pool);
job_t *get_job(thread_pool_t *thread_pool);
void destroy_thread_pool(thread_pool_t *thread_pool);

#endif // _THREAD_POOL_H_

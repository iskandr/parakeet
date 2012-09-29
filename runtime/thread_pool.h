#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <pthread.h>

typedef void (*work_function_t)(int, void*);

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

typedef enum {
  THREAD_RUN = 0,
  THREAD_FINISHED,
  THREAD_PAUSE,
  THREAD_IDLE,
  THREAD_STOP
} thread_status_t;

typedef struct {
  task_list_t     *task_list;
  pthread_mutex_t  mutex;
  pthread_cond_t   cond;
  thread_status_t  status;
  pthread_cond_t  *master_cond;
  int              notify_when_done;
  work_function_t  work_function;
  void            *args;
} worker_data_t;

typedef struct {
  pthread_t      *workers;
  int             num_workers;
  int             num_active;
  pthread_cond_t  master_cond;
  worker_data_t  *worker_data;
  job_t          *job;
} thread_pool_t;

thread_pool_t *create_thread_pool(int max_threads);
void launch_job(thread_pool_t *thread_pool,
                work_function_t work_function, void *args, job_t *job);
void pause_job(thread_pool_t *thread_pool);
void wait_for_job(thread_pool_t *thread_pool);
job_t *get_job(thread_pool_t *thread_pool);
void destroy_thread_pool(thread_pool_t *thread_pool);

#endif // _THREAD_POOL_H_

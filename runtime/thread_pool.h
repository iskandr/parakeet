#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <pthread.h>

typedef void (*work_function_t)(int, void*);

typedef struct {
  long            next_iteration;
  long            last_iteration;
} task_t;

typedef struct {
  task_t *tasks;
  int     num_tasks;
  int     cur_task;
} task_list_t;

typedef enum {
  THREAD_RUN = 0,
  THREAD_FINISHED,
  THREAD_PAUSE,
  THREAD_IDLE,
  THREAD_STOP
} thread_status_t;

typedef struct {
  task_list_t       *task_list;
  thread_status_t    status;
  pthread_barrier_t *barrier;
  work_function_t    work_function;
  void              *args;
} worker_data_t;

typedef struct {
  pthread_t         *workers;
  int                num_workers;
  int                num_active;
  pthread_barrier_t *paused_barrier;
  pthread_barrier_t *idle_barrier;
  task_list_t       *task_lists;
  worker_data_t     *worker_data;
} thread_pool_t;

thread_pool_t *create_thread_pool(int max_threads);
void launch_tasks(thread_pool_t *thread_pool, int num_threads,
                  work_function_t work_function, void *args, int num_args,
                  task_list_t *task_lists);
void pause_tasks(thread_pool_t *thread_pool);
void wait_for_tasks(thread_pool_t *thread_pool);
task_list_t *get_task_lists(thread_pool_t *thread_pool);
void destroy_thread_pool(thread_pool_t *thread_pool);

#endif // _THREAD_POOL_H_

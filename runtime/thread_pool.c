#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "thread_pool.h"

typedef struct {
  int               id;
  worker_data_t    *worker_data;
} worker_args_t;

static void *worker(void *args) {
  worker_args_t *my_args = (worker_args_t*)args;

  int id = my_args->id;
  worker_data_t *worker_data = my_args->worker_data;
  free(args);

  // Wait for task queue to be ready for me
  pthread_barrier_wait(worker_data->barrier);
  int total_iters_done = 0;

  // Do work forever
  for (;;) {
    thread_status_t status = __sync_fetch_and_add(&worker_data->status, 0);
    if (status == THREAD_RUN) {
      // get_time();
      task_list_t *task_list = worker_data->task_list;
      task_t *task = &task_list->tasks[task_list->cur_task];
      (*worker_data->work_function)(task->next_iteration++, worker_data->args);
      total_iters_done++;
      // get_time();
      // update local time accumulator();
      
      // If I'm done with this task, move to the next one or move to the
      // finished state.  For now, don't do any work stealing.
      if (task->next_iteration > task->last_iteration) {
        if (task_list->cur_task == task_list->num_tasks) {
          // TODO: Have to make sure that the value hasn't changed??
          __sync_val_compare_and_swap(&worker_data->status,
                                      status, THREAD_FINISHED);
        } else {
          task_list->cur_task++;
        }
      }
    } else if (status == THREAD_STOP) {
      pthread_barrier_wait(worker_data->barrier);
      break;
    } else {
      pthread_barrier_wait(worker_data->barrier);
    }
  }
}

thread_pool_t *create_thread_pool(int max_threads) {
  int i, rc;

  thread_pool_t *thread_pool = (thread_pool_t*)malloc(sizeof(thread_pool_t));
  
  thread_pool->workers = (pthread_t*)malloc(max_threads*sizeof(pthread_t));
  thread_pool->num_workers = max_threads;
  thread_pool->num_active = 0;
  thread_pool->paused_barrier =
    (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
  thread_pool->idle_barrier =
    (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(thread_pool->paused_barrier, NULL, 1);
  pthread_barrier_init(thread_pool->idle_barrier, NULL, max_threads + 1);
  thread_pool->worker_data =
    (worker_data_t*)malloc(sizeof(worker_data_t)*max_threads);

  for (i = 0; i < max_threads; ++i) {
    thread_pool->worker_data[i].task_list = NULL;
    thread_pool->worker_data[i].status = THREAD_IDLE;
    thread_pool->worker_data[i].barrier = thread_pool->idle_barrier;
    worker_args_t *args = (worker_args_t*)malloc(sizeof(worker_args_t));
    args->id = i;
    args->worker_data = &thread_pool->worker_data[i];
    rc = pthread_create(&thread_pool->workers[i], NULL, worker, (void*)args);
    if (rc) {
      printf("Couldn't create worker %d (Error code %d). Exiting.\n", i, rc);
      exit(-1);
    }
  }
  
  return thread_pool;
}

// This function should only ever be called when all of the threads are paused
// or idle.
void launch_tasks(thread_pool_t *thread_pool, int num_threads,
                  work_function_t work_function, void *args, int num_args,
                  task_list_t *task_lists) {
  assert(num_threads <= thread_pool->num_workers);

  // Cache out the current barriers to use to wake up the threads waiting on
  // them.
  pthread_barrier_t *old_paused_barrier = thread_pool->paused_barrier;
  pthread_barrier_t *old_idle_barrier = thread_pool->idle_barrier;
  
  int num_threads_changed = thread_pool->num_active != num_threads;

  // If we're changing the number of active threads, we need to adjust the
  // barriers to be for the proper number of threads each.
  if (num_threads_changed) {
    thread_pool->paused_barrier =
      (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
    thread_pool->idle_barrier =
      (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
    pthread_barrier_init(thread_pool->paused_barrier, NULL, num_threads + 1);
    pthread_barrier_init(thread_pool->idle_barrier, NULL,
                         thread_pool->num_workers - num_threads + 1);
  }
  
  thread_pool->task_lists = task_lists;
  thread_pool->num_active = num_threads;

  // Update the threads' data with the current batch of work and parallelism
  // configuration.
  int i;
  for (i = 0; i < num_threads; ++i) {
    assert(thread_pool->worker_data[i].status != THREAD_RUN);
    assert(task_lists[i].num_tasks > 0);

    thread_pool->worker_data[i].status = THREAD_RUN;
    thread_pool->worker_data[i].task_list = &task_lists[i];
    thread_pool->worker_data[i].barrier = thread_pool->paused_barrier;
    thread_pool->worker_data[i].work_function = work_function;
    thread_pool->worker_data[i].args = args;
  }
  for (i = num_threads; i < thread_pool->num_workers; ++i) {
    assert(thread_pool->worker_data[i].status != THREAD_RUN);
    assert(task_lists[i].num_tasks == 0);

    thread_pool->worker_data[i].status = THREAD_IDLE;
    thread_pool->worker_data[i].task_list = NULL;
    thread_pool->worker_data[i].barrier = thread_pool->idle_barrier;
    thread_pool->worker_data[i].work_function = NULL;
    thread_pool->worker_data[i].args = NULL;
  }

  // Wait on each of the old barriers, thus waking up all of the worker threads.
  pthread_barrier_wait(old_paused_barrier);
  pthread_barrier_wait(old_idle_barrier);
  
  // Clean up the old barriers if the number of threads changed.
  if (num_threads_changed) {
    pthread_barrier_destroy(old_paused_barrier);
    pthread_barrier_destroy(old_idle_barrier);
    free(old_paused_barrier);
    free(old_idle_barrier);
  }
}

void pause_tasks(thread_pool_t *thread_pool) {
  int i;
  for (i = 0; i < thread_pool->num_active; ++i) {
    thread_status_t status =
      __sync_val_compare_and_swap(&thread_pool->worker_data[i].status,
                                  THREAD_RUN, THREAD_PAUSE);
  }
  
  // Make sure all threads have seen the message before returning.
  pthread_barrier_wait(thread_pool->paused_barrier);
}

void wait_for_tasks(thread_pool_t *thread_pool) {
  // Barrier wait for all tasks to reach the only point they can: finished.
  pthread_barrier_wait(thread_pool->paused_barrier);
  
  int i, all_done;
  all_done = 1;
  for (i = 0; i < thread_pool->num_active; ++i) {
    all_done &= THREAD_FINISHED == thread_pool->worker_data[i].status;
  }
  
  if (!all_done) {
    printf("Error: waited for tasks to finish but they didn't!\n");
  }
  
  // TODO: At this point, we can free the task lists.  Should we be the ones
  //       with that responsibility?
}

task_list_t *get_task_lists(thread_pool_t *thread_pool) {
  return thread_pool->task_lists;
}

void destroy_thread_pool(thread_pool_t *thread_pool) {
  int i, all_done;
  all_done = 1;
  for (i = 0; i < thread_pool->num_active; ++i) {
    all_done &=
      __sync_bool_compare_and_swap(&thread_pool->worker_data[i].status,
                                   THREAD_FINISHED, THREAD_STOP);
  }
  pthread_barrier_wait(thread_pool->paused_barrier);
  
  if (!all_done) {
    printf("Error: waited for tasks to finish but they didn't!\n");
    exit(-1);
  }
  
  for (i = thread_pool->num_active; i < thread_pool->num_workers; ++i) {
    thread_pool->worker_data[i].status = THREAD_STOP;
  }
  pthread_barrier_wait(thread_pool->idle_barrier);
  
  free(thread_pool->worker_data);
  pthread_barrier_destroy(thread_pool->paused_barrier);
  free(thread_pool->paused_barrier);
  pthread_barrier_destroy(thread_pool->idle_barrier);
  free(thread_pool->idle_barrier);
  free(thread_pool->workers);
}

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

  struct timeval start, end, result;

  // Do work forever
  for (;;) {
    pthread_mutex_lock(&worker_data->mutex);
    thread_status_t status = worker_data->status;
    pthread_mutex_unlock(&worker_data->mutex);
    if (status == THREAD_RUN) {
      pthread_mutex_lock(&worker_data->mutex);
      task_list_t *task_list = worker_data->task_list;

      // Check whether we're done.
      if (task_list->cur_task == task_list->num_tasks ||
          (worker_data->fixed_num_iters > 0 &&
           worker_data->fixed_num_iters == worker_data->iters_done)) {
        worker_data->status = THREAD_FINISHED;
        if (worker_data->notify_when_done) {
          pthread_cond_signal(worker_data->master_cond);
        }
      } else {
        // We know now that we have an iteration to perform for this task, so
        // do it.
        task_t *task = &task_list->tasks[task_list->cur_task];
        gettimeofday(&start, NULL);
        (*worker_data->work_function)(task->next_iteration++,
                                      worker_data->args);
        gettimeofday(&end, NULL);
        timersub(&end, &start, &result);
        worker_data->time_spent += result.tv_sec + result.tv_usec / 1000000.0;
        worker_data->iters_done++;

        // If this was the last iteration of this task, move to the next one.
        if (task->next_iteration > task->last_iteration) {
          task_list->cur_task++;
        }
      }
      pthread_mutex_unlock(&worker_data->mutex);
    } else if (status == THREAD_STOP) {
      break;
    } else {
      pthread_mutex_lock(&worker_data->mutex);
      if (status == THREAD_PAUSE) {
        pthread_barrier_wait(worker_data->task_list->barrier);
      }
      pthread_cond_wait(&worker_data->cond, &worker_data->mutex);
      pthread_mutex_unlock(&worker_data->mutex);
    }
  }
  
  pthread_exit(NULL);
}

thread_pool_t *create_thread_pool(int max_threads) {
  int i, rc;

  thread_pool_t *thread_pool = (thread_pool_t*)malloc(sizeof(thread_pool_t));
  
  thread_pool->workers = (pthread_t*)malloc(max_threads*sizeof(pthread_t));
  thread_pool->num_workers = max_threads;
  thread_pool->num_active = 0;
  pthread_cond_init(&thread_pool->master_cond, NULL);
  thread_pool->worker_data =
    (worker_data_t*)malloc(sizeof(worker_data_t)*max_threads);
  thread_pool->job = NULL;

  for (i = 0; i < max_threads; ++i) {
    thread_pool->worker_data[i].task_list = NULL;
    pthread_mutex_init(&thread_pool->worker_data[i].mutex, NULL);
    pthread_cond_init(&thread_pool->worker_data[i].cond, NULL);
    thread_pool->worker_data[i].status = THREAD_IDLE;
    thread_pool->worker_data[i].master_cond =
      &thread_pool->master_cond;
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

// This function should only ever be called when all of the threads are paused.
void launch_job(thread_pool_t *thread_pool,
                work_function_t work_function, void *args, job_t *job,
                int fixed_num_iters) {
  assert(job->num_lists <= thread_pool->num_workers);

  thread_pool->job = job;
  thread_pool->num_active = job->num_lists;

  // Update the threads' data with the current batch of work and parallelism
  // configuration.
  int i;
  for (i = 0; i < thread_pool->num_active; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    if (job->task_lists[i].num_tasks > 0) {
      thread_pool->worker_data[i].status = THREAD_RUN;
      thread_pool->worker_data[i].task_list = &job->task_lists[i];
    } else {
      thread_pool->worker_data[i].status = THREAD_FINISHED;
      thread_pool->worker_data[i].task_list = NULL;
    }
    thread_pool->worker_data[i].notify_when_done = 0;
    thread_pool->worker_data[i].work_function = work_function;
    thread_pool->worker_data[i].args = args;
    thread_pool->worker_data[i].fixed_num_iters = fixed_num_iters;
    thread_pool->worker_data[i].iters_done = 0;
    thread_pool->worker_data[i].time_spent = 0.0;
    pthread_cond_signal(&thread_pool->worker_data[i].cond);
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
  for (i = thread_pool->num_active; i < thread_pool->num_workers; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    thread_pool->worker_data[i].status = THREAD_IDLE;
    thread_pool->worker_data[i].task_list = NULL;
    thread_pool->worker_data[i].work_function = NULL;
    thread_pool->worker_data[i].args = NULL;
    thread_pool->worker_data[i].fixed_num_iters = fixed_num_iters;
    thread_pool->worker_data[i].iters_done = 0;
    thread_pool->worker_data[i].time_spent = 0.0;
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
}

void pause_job(thread_pool_t *thread_pool) {
  int i;
  for (i = 0; i < thread_pool->num_active; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    thread_pool->worker_data[i].status = THREAD_PAUSE;
    pthread_cond_signal(&thread_pool->worker_data[i].cond);
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
  pthread_barrier_wait(&thread_pool->job->barrier);

  thread_pool->num_active = 0;
}

int job_finished(thread_pool_t *thread_pool) {
  int all_done = 1;
  int i;
  for (i = 0; i < thread_pool->num_active && all_done; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    all_done =
      all_done && THREAD_FINISHED == thread_pool->worker_data[i].status;
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
  return all_done;
}

double get_throughput(thread_pool_t *thread_pool) {
  int total_iters = 0;
  double total_time = 0.0;
  int i;
  for (i = 0; i < thread_pool->num_active; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    total_iters += thread_pool->worker_data[i].iters_done;
    total_time += thread_pool->worker_data[i].time_spent;
    thread_pool->worker_data[i].iters_done = 0;
    thread_pool->worker_data[i].time_spent = 0.0;
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
  return total_iters / total_time;
}

void wait_for_job(thread_pool_t *thread_pool) {
  int i;
  for (i = 0; i < thread_pool->num_active; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    if (THREAD_FINISHED != thread_pool->worker_data[i].status) {
      thread_pool->worker_data[i].notify_when_done = 1;
      pthread_cond_wait(&thread_pool->master_cond,
                        &thread_pool->worker_data[i].mutex);
    }
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
  }
}

job_t *get_job(thread_pool_t *thread_pool) {
  return thread_pool->job;
}

void destroy_thread_pool(thread_pool_t *thread_pool) {
  int i;  
  for (i = 0; i < thread_pool->num_workers; ++i) {
    pthread_mutex_lock(&thread_pool->worker_data[i].mutex);
    thread_pool->worker_data[i].status = THREAD_STOP;
    pthread_cond_signal(&thread_pool->worker_data[i].cond);
    pthread_mutex_unlock(&thread_pool->worker_data[i].mutex);
    pthread_join(thread_pool->workers[i], NULL);
    pthread_mutex_destroy(&thread_pool->worker_data[i].mutex);
    pthread_cond_destroy(&thread_pool->worker_data[i].cond);
  }

  pthread_cond_destroy(&thread_pool->master_cond);
  free(thread_pool->worker_data);
  free(thread_pool->workers);
  free(thread_pool);
}

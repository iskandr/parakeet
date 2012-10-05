#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "thread_pool.h"

#include "runtime.h"

job_t *make_job(int start, int stop, int num_threads, int chunk_len) {
  int num_iters = stop - start;
  int num_chunks = num_iters / chunk_len + (num_iters % chunk_len ? 1 : 0);
  job_t *job = (job_t*)malloc(sizeof(job_t));
  job->task_lists = (task_list_t*)malloc(sizeof(task_list_t) * num_threads);
  job->num_lists = num_threads;
  pthread_barrier_init(&job->barrier, NULL, num_threads + 1);
  int last_thread =
    ((num_chunks % num_threads - 1) + num_threads) % num_threads;
  int i, j;
  for (i = 0; i < num_threads; ++i) {
    int num_tasks = num_chunks / num_threads;
    num_tasks += i < num_chunks % num_threads;
    job->task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
    job->task_lists[i].num_tasks = num_tasks;
    job->task_lists[i].cur_task = 0;
    job->task_lists[i].barrier = &job->barrier;
    int cur_iter = chunk_len * i + start;
    int step = chunk_len * num_threads;
    for (j = 0; j < num_tasks; ++j) {
      job->task_lists[i].tasks[j].next_iteration = cur_iter;
      job->task_lists[i].tasks[j].last_iteration = cur_iter + chunk_len - 1;
      cur_iter += step;
    }
    if (i == last_thread) {
      job->task_lists[i].tasks[num_tasks - 1].last_iteration = stop - 1;
    }
  }

  return job;
}

job_t *reconfigure_job(job_t *job, int num_threads) {
  if (num_threads == job->num_lists) {
    return job;
  }

  job_t *new_job = (job_t*)malloc(sizeof(job_t));
  new_job->task_lists = (task_list_t*)malloc(sizeof(task_list_t) * num_threads);
  new_job->num_lists = num_threads;
  pthread_barrier_init(&new_job->barrier, NULL, num_threads + 1);
  int total_tasks = num_unfinished_tasks(job);

  int cur_list = 0;
  int cur_task = job->task_lists[0].cur_task;
  int i;
  for (i = 0; i < num_threads; ++i) {
    int num_tasks = total_tasks / num_threads;
    num_tasks += i < total_tasks % num_threads;
    new_job->task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
    new_job->task_lists[i].num_tasks = num_tasks;
    new_job->task_lists[i].cur_task = 0;
    new_job->task_lists[i].barrier = &new_job->barrier;

    int tasks_done = 0;
    while(tasks_done < num_tasks) {
      int tasks_left_in_cur_list =
        job->task_lists[cur_list].num_tasks - cur_task;

      // Skip any empty lists
      if (tasks_left_in_cur_list < 1) {
        cur_list++;
        cur_task = job->task_lists[cur_list].cur_task;
        continue;
      }

      int tasks_left_to_do = num_tasks - tasks_done;
      task_t *src = job->task_lists[cur_list].tasks + cur_task;
      int num_to_copy;

      if (tasks_left_in_cur_list > tasks_left_to_do) {
        num_to_copy = tasks_left_to_do;
        cur_task += tasks_left_to_do;
      } else {
        num_to_copy = tasks_left_in_cur_list;
        cur_list++;
        if (cur_list < job->num_lists) {
          cur_task = job->task_lists[cur_list].cur_task;
        }
      }

      memcpy(&new_job->task_lists[i].tasks[tasks_done], src,
             sizeof(task_t) * num_to_copy);
      tasks_done += num_to_copy;
    }
  }
  free_job(job);

  return new_job;
}

int num_unfinished_tasks(job_t *job) {
  int total_tasks = 0;
  int i;
  for (i = 0; i < job->num_lists; ++i) {
    total_tasks += job->task_lists[i].num_tasks - job->task_lists[i].cur_task;
  }
  return total_tasks;
}

int num_threads(job_t *job) {
  return job->num_lists;
}

void free_job(job_t *job) {
  int i;
  for (i = 0; i < job->num_lists; ++i) {
    free(job->task_lists[i].tasks);
  }
  free(job->task_lists);
  pthread_barrier_destroy(&job->barrier);
  free(job);
}

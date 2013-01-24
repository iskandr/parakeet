#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "job.h"
#include "runtime.h"
#include "thread_pool.h"

static inline safe_div(n, d) {
  return n / d + (n % d ? 1 : 0);
}

// This function assigns the tasks in round-robin fashion to threads.
job_t *make_job(int64_t start, int64_t stop, int64_t step, int num_threads,
                int64_t task_len) {
  int64_t num_iters = stop - start;
  int64_t num_total_tasks = safe_div(num_iters, step);
  int64_t num_total_chunks = safe_div(num_total_tasks, task_len);
  job_t *job = (job_t*)malloc(sizeof(job_t));
  job->task_lists = (task_list_t*)malloc(sizeof(task_list_t) * num_threads);
  job->num_lists = num_threads;
  pthread_barrier_init(&job->barrier, NULL, num_threads + 1);
  int i, j;
  int64_t cur_iter = start;
  int64_t chunk_step = task_len * step;
  for (i = 0; i < num_threads; ++i) {
    int64_t num_tasks = num_total_chunks / num_threads;
    num_tasks += i < num_total_chunks % num_threads;
    job->task_lists[i].num_tasks = num_tasks;
    job->task_lists[i].cur_task = 0;
    job->task_lists[i].barrier = &job->barrier;
    if (num_tasks > 0) {
      job->task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
    }
    for (j = 0; j < num_tasks; ++j) {
      job->task_lists[i].tasks[j].next_start = cur_iter;
      job->task_lists[i].tasks[j].step = step;
      cur_iter += chunk_step;
      job->task_lists[i].tasks[j].end = cur_iter;
    }
    if (i == num_threads - 1) {
      job->task_lists[i].tasks[num_tasks - 1].end = stop;
    }
  }

  return job;
}

// TODO: If we ever want to use task_len, this needs to be updated.
job_t *reconfigure_job(job_t *job, int64_t step) {
  int i;
  for (i = 0; i < job->num_lists; ++i) {
    if (job->task_lists[i].cur_task != job->task_lists[i].num_tasks) {
      int64_t start =
          job->task_lists[i].tasks[job->task_lists[i].cur_task].next_start;
      int64_t end =
          job->task_lists[i].tasks[job->task_lists[i].num_tasks - 1].end;
      int64_t num_tasks = safe_div(end - start, step);
      free(job->task_lists[i].tasks);
      job->task_lists[i].num_tasks = num_tasks;
      job->task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
      job->task_lists[i].cur_task = 0;
      int64_t cur_iter = start;
      int j;
      for (j = 0; j < num_tasks; ++j) {
        job->task_lists[i].tasks[j].next_start = cur_iter;
        job->task_lists[i].tasks[j].step = step;
        cur_iter += step;
        job->task_lists[i].tasks[j].end = cur_iter;
      }
      job->task_lists[i].tasks[num_tasks - 1].end = end;
    }
  }

  return job;
}

//job_t *reconfigure_job(job_t *job, int num_threads) {
//  if (num_threads == job->num_lists) {
//    return job;
//  }
//
//  job_t *new_job = (job_t*)malloc(sizeof(job_t));
//  new_job->task_lists = (task_list_t*)malloc(sizeof(task_list_t) * num_threads);
//  new_job->num_lists = num_threads;
//  pthread_barrier_init(&new_job->barrier, NULL, num_threads + 1);
//  int total_tasks = num_unfinished_tasks(job);
//
//  int cur_list = 0;
//  int cur_task = job->task_lists[0].cur_task;
//  int i;
//  for (i = 0; i < num_threads; ++i) {
//    int num_tasks = total_tasks / num_threads;
//    num_tasks += i < total_tasks % num_threads;
//    new_job->task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
//    new_job->task_lists[i].num_tasks = num_tasks;
//    new_job->task_lists[i].cur_task = 0;
//    new_job->task_lists[i].barrier = &new_job->barrier;
//
//    int tasks_done = 0;
//    while(tasks_done < num_tasks) {
//      int tasks_left_in_cur_list =
//        job->task_lists[cur_list].num_tasks - cur_task;
//
//      // Skip any empty lists
//      if (tasks_left_in_cur_list < 1) {
//        cur_list++;
//        cur_task = job->task_lists[cur_list].cur_task;
//        continue;
//      }
//
//      int tasks_left_to_do = num_tasks - tasks_done;
//      task_t *src = job->task_lists[cur_list].tasks + cur_task;
//      int num_to_copy;
//
//      if (tasks_left_in_cur_list > tasks_left_to_do) {
//        num_to_copy = tasks_left_to_do;
//        cur_task += tasks_left_to_do;
//      } else {
//        num_to_copy = tasks_left_in_cur_list;
//        cur_list++;
//        if (cur_list < job->num_lists) {
//          cur_task = job->task_lists[cur_list].cur_task;
//        }
//      }
//
//      memcpy(&new_job->task_lists[i].tasks[tasks_done], src,
//             sizeof(task_t) * num_to_copy);
//      tasks_done += num_to_copy;
//    }
//  }
//  free_job(job);
//
//  return new_job;
//}

int num_threads(job_t *job) {
  return job->num_lists;
}

void free_job(job_t *job) {
  int i;
  for (i = 0; i < job->num_lists; ++i) {
    if (job->task_lists[i].num_tasks > 0) {
      free(job->task_lists[i].tasks);
    }
  }
  free(job->task_lists);
  pthread_barrier_destroy(&job->barrier);
  free(job);
}

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "thread_pool.h"

#include "runtime.h"

job_t *make_job(int len, int max_threads, int num_threads) {
  int chunk_len = 32;
  int num_chunks = len / chunk_len + (len % chunk_len ? 1 : 0);
  job_t *job = (job_t*)malloc(sizeof(job_t));
  job->task_lists = (task_list_t*)malloc(sizeof(task_list_t) * max_threads);
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
    int cur_iter = chunk_len * i;
    int step = chunk_len * num_threads;
    for (j = 0; j < num_tasks; ++j) {
      job->task_lists[i].tasks[j].next_iteration = cur_iter;
      job->task_lists[i].tasks[j].last_iteration = cur_iter + chunk_len - 1;
      cur_iter += step;
    }
    if (i == last_thread) {
      job->task_lists[i].tasks[num_tasks - 1].last_iteration = len - 1;
    }
  }
  for (i = num_threads; i < max_threads; ++i) {
    job->task_lists[i].tasks = NULL;
    job->task_lists[i].num_tasks = 0;
    job->task_lists[i].cur_task = 0;
    job->task_lists[i].barrier = NULL;
  }

  return job;
}
/*
task_list_t *reconfigure_task_lists(task_list_t *task_lists, int num_lists,
                                    int num_threads) {
  task_list_t *new_task_lists =
    (task_list_t*)malloc(sizeof(task_list_t) * num_lists);
  int total_tasks = 0;
  int i;
  for (i = 0; i < num_lists; ++i) {
    if (task_lists[i].tasks == NULL) {
      break;
    }
    
    total_tasks += task_lists[i].num_tasks - task_lists[i].cur_task;
  }

  int cur_list = 0;
  int cur_task = task_lists[0].cur_task;
  for (i = 0; i < num_threads; ++i) {
    int num_tasks = total_tasks / num_threads;
    num_tasks += i < total_tasks % num_threads ? 1 : 0;
    new_task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
    new_task_lists[i].num_tasks = num_tasks;
    new_task_lists[i].cur_task = 0;
    
    int tasks_done = 0;
    while(tasks_done < num_tasks) {
      int tasks_left_in_cur_list = task_lists[cur_list].num_tasks - cur_task;
      int tasks_left_to_do = num_tasks - tasks_done;
      task_t *src = task_lists[cur_list].tasks + cur_task;
      int num_to_copy;
      
      if (tasks_left_in_cur_list > tasks_left_to_do) {
        num_to_copy = tasks_left_to_do;
        cur_task += tasks_left_to_do;
      } else {
        num_to_copy = tasks_left_in_cur_list;
        cur_list++;
        cur_task = task_lists[cur_list].cur_task;
      }
      
      memcpy(&new_task_lists[i].tasks[tasks_done], src,             
             sizeof(task_t) * num_to_copy);
      tasks_done += num_to_copy;
    }
  }
  for (i = num_threads; i < num_lists; ++i) {
    new_task_lists[i].tasks = NULL;
    new_task_lists[i].num_tasks = 0;
    new_task_lists[i].cur_task = 0;
  }
  
  for (i = 0; i < num_lists; ++i) {
    if (task_lists[i].tasks) {
      free(task_lists[i].tasks);
    }
  }
  free(task_lists);

  return new_task_lists;
}*/

void free_job(job_t *job) {
  int i;
  for (i = 0; i < job->num_lists; ++i) {
    free(job->task_lists[i].tasks);
  }
  free(job->task_lists);
  pthread_barrier_destroy(&job->barrier);
  free(job);
}

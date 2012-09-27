#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "thread_pool.h"

#include "runtime.h"

task_list_t *make_task_lists(int len, int max_threads, int num_threads) {
  int chunk_len = 32;
  int num_chunks = len / chunk_len + (len % chunk_len ? 1 : 0);
  task_list_t *task_lists =
    (task_list_t*)malloc(sizeof(task_list_t) * max_threads);
  int i, j;
  for (i = 0; i < num_threads; ++i) {
    int num_tasks = num_chunks / num_threads;
    num_tasks += i < num_chunks % num_threads ? 1 : 0;
    task_lists[i].tasks = (task_t*)malloc(sizeof(task_t) * num_tasks);
    task_lists[i].num_tasks = num_tasks;
    task_lists[i].cur_task = 0;
    int cur_iter = chunk_len * i;
    int step = chunk_len * num_threads;
    for (j = 0; j < num_tasks; ++j) {
      task_lists[i].tasks[j].next_iteration = cur_iter;
      task_lists[i].tasks[j].last_iteration = cur_iter + chunk_len - 1;
      cur_iter += step;
    }
  }
  for (i = num_threads; i < max_threads; ++i) {
    task_lists[i].tasks = NULL;
    task_lists[i].num_tasks = 0;
    task_lists[i].cur_task = 0;
  }
  
  int last = num_threads - 1;
  task_lists[last].tasks[task_lists[last].num_tasks - 1].last_iteration =
    len - 1;

  return task_lists;
}

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
      int old_cur_task = cur_task;
      int old_cur_list = cur_list;
      
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
}

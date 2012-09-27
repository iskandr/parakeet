#ifndef _RUNTIME_H_
#define _RUNTIME_H_

#include <pthread.h>

#include "thread_pool.h"

task_list_t *make_task_lists(int len, int max_threads, int num_threads);
task_list_t *reconfigure_task_lists(task_list_t *task_lists, int num_lists,
                                    int num_threads);

#endif // _RUNTIME_H_

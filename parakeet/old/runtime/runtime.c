#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "thread_pool.h"

#include "runtime.h"

const int n_seq = 1680;
int d_par = 4; // TODO: Explore other ways of setting this.

runtime_t *create_runtime(int max_threads) {
  runtime_t *runtime = (runtime_t*)malloc(sizeof(runtime_t));
  runtime->thread_pool = create_thread_pool(max_threads);
  return runtime;
}

void run_job(runtime_t *runtime,
             work_function_t work_function, void *args, int arg_len) {
  // Calibrate the parallel version
  printf("Calibrating parallel version\n");
  job_t *job = make_job(0, arg_len, d_par);
  double par_tp = calibrate_par(runtime, work_function, args, job);
  job = get_job(runtime->thread_pool);
  int num_threads = job->num_lists;
  printf("Parallel version to have %d threads\n", num_threads);

  // TODO: For now, we only consider parallel versions.
  while (!job_finished(runtime->thread_pool)) {
    usleep(20000);
    double tp = get_throughput(runtime->thread_pool);
    if (abs(tp - par_tp) / par_tp > 0.2) {
      printf("Throughput changed.\n");
      printf("Old throughput: %f\n", par_tp);
      printf("New throughput: %f\n", tp);
      pause_job(runtime->thread_pool);
      par_tp = calibrate_par(runtime, work_function, args, job);
      job = get_job(runtime->thread_pool);
    }
  }
  free_job(job);
}

void destroy_runtime(runtime_t *runtime) {
  destroy_thread_pool(runtime->thread_pool);
  free(runtime);
}

static double calibrate_par(runtime_t *runtime,
                            work_function_t work_function, void *args,
                            job_t *job) {
  int dop = job->num_lists;
  double dop_tp, dop_m1_tp, dop_p1_tp;

  // Get throughput for DOP - 1 # of threads
  if (dop > 1) {
    job = reconfigure_job(job, dop - 1);
    dop_m1_tp = get_par_throughput(runtime, work_function, args, job, dop - 1);
    printf("TP with %d threads: %f\n", dop - 1, dop_m1_tp);
  } else {
    dop_m1_tp = dop_tp + 1;
  }

  // Get throughput for DOP + 1 # of threads
  if (dop < runtime->thread_pool->num_workers) {
    job = reconfigure_job(job, dop + 1);
    dop_p1_tp = get_par_throughput(runtime, work_function, args, job, dop + 1);
    printf("TP with %d threads: %f\n", dop + 1, dop_p1_tp);
  } else {
    dop_p1_tp = dop_tp + 1;
  }

  // Get throughput for DOP # of threads
  job = reconfigure_job(job, dop);
  dop_tp = get_par_throughput(runtime, work_function, args, job, dop);
  printf("TP with %d threads: %f\n", dop, dop_tp);

  int increasing = 1;
  if (dop_m1_tp > dop_tp && dop_m1_tp > dop_p1_tp) {
    increasing = 0;
  } else if (dop_p1_tp <= dop_tp) {
    job = reconfigure_job(job, dop);
    launch_job(runtime->thread_pool, work_function, args, job, 0);
    return dop_tp;
  }

  int not_done = 1;
  while (not_done) {
    if (increasing) {
      job = reconfigure_job(job, dop + 1);
      dop_p1_tp = get_par_throughput(runtime, work_function, args, job, dop+1);
      printf("TP with %d threads: %f\n", dop + 1, dop_p1_tp);
      if (dop_p1_tp <= dop_tp) {
        job = reconfigure_job(job, dop);
        not_done = 0;
      } else {
        dop += 1;
        dop_tp = dop_p1_tp;
      }
    } else {
      job = reconfigure_job(job, dop - 1);
      dop_m1_tp = get_par_throughput(runtime, work_function, args, job, dop-1);
      printf("TP with %d threads: %f\n", dop - 1, dop_m1_tp);
      if (dop_m1_tp < dop_tp) {
        job = reconfigure_job(job, dop);
        not_done = 0;
      } else {
        dop -= 1;
        dop_tp = dop_m1_tp;
      }
    }
    not_done = not_done && dop > 1 && dop < runtime->thread_pool->num_workers;
  }
  launch_job(runtime->thread_pool, work_function, args, job, 0);

  return dop_tp;
}

static double get_seq_throughput(work_function_t work_function, void *args,
                                 int num_iters) {
  struct timeval start, stop, result;
  int i;
  gettimeofday(&start, NULL);
  for (i = 0; i < num_iters; ++i) {
    (*work_function)(i, args);
  }
  gettimeofday(&stop, NULL);
  timersub(&stop, &start, &result);
  return num_iters / (result.tv_sec + result.tv_usec / 1000000.0);
}

static double get_par_throughput(runtime_t *runtime,
                                 work_function_t work_function, void *args,
                                 job_t *job, int num_threads) {
  launch_job(runtime->thread_pool, work_function, args, job,
             get_npar(num_threads));
  wait_for_job(runtime->thread_pool);
  return get_throughput(runtime->thread_pool);
}

static int get_npar(int num_threads) {
  //return n_seq > 2 * num_threads ? n_seq : 2 * num_threads;
  return n_seq / num_threads;
}

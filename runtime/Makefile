IFLAGS = -g
LFLAGS = -pthread
FLAGS = $(IFLAGS) $(LFLAGS)
OBJS = runtime.o thread_pool.o job.o

all: runtime

job:
	gcc -fPIC -c job.c $(FLAGS) -o job.o

threadpool: job
	gcc -fPIC -c thread_pool.c $(FLAGS) -o thread_pool.o

runtime: threadpool job
	gcc -fPIC -c runtime.c $(FLAGS) -o runtime.o

tests: runtime threadpool
	gcc -o runtime_test $(FLAGS) -lcunit runtime_test.c $(OBJS)
	gcc -o job_test $(FLAGS) -lcunit job_test.c $(OBJS)

clean:
	rm *_test *.o *~

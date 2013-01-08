IFLAGS = -O3
LFLAGS = -pthread -lm -lprofiler
FLAGS = $(IFLAGS) $(LFLAGS)
OBJS = thread_pool.o job.o

all: runtime

job:
	gcc -fPIC -c job.c $(FLAGS) -o job.o

threadpool: job
	gcc -fPIC -c thread_pool.c $(FLAGS) -o thread_pool.o

runtime: threadpool job
	gcc -shared -Wl -Bsymbolic -o libparakeetruntime.so $(OBJS)

tests: runtime threadpool
	#gcc -o job_test $(FLAGS) -lcunit job_test.c $(OBJS)
	gcc -fPIC -c vm.c $(FLAGS) -o vm.o
	gcc -shared -o vm.so vm.o
	#gcc -o vm_test $(FLAGS) -lcunit vm_test.c $(OBJS)

clean:
	rm -f *_test *.o *.so *.pyc *~

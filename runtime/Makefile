CC = gcc
IFLAGS = -O3 -ffast-math -ftree-vectorize -msse2
#IFLAGS = -O3 -ffast-math -msse2 -mllvm -vectorize-loops
LFLAGS = -pthread -lm -lprofiler
FLAGS = $(IFLAGS) $(LFLAGS)
OBJS = thread_pool.o job.o

all: runtime

job:
	$(CC) -fPIC -c job.c $(FLAGS) -o job.o

threadpool: job
	$(CC) -fPIC -c thread_pool.c $(FLAGS) -o thread_pool.o

runtime: threadpool job
	$(CC) -shared -Wl -Bsymbolic -o libparakeetruntime.so $(OBJS)

tests: runtime threadpool
	$(CC) -fPIC -c vm.c $(FLAGS) -o vm.o
	$(CC) -shared -o vm.so vm.o

clean:
	rm -f *_test *.o *.so *.pyc *~

IFLAGS = -g
LFLAGS = -pthread
FLAGS = $(IFLAGS) $(LFLAGS)

all: runtime

threadpool:
	gcc -fPIC -c thread_pool.c $(FLAGS) -o thread_pool.o

runtime: threadpool
	gcc -fPIC -c runtime.c $(FLAGS) -o runtime.o

tests: runtime threadpool
	gcc -o runtime_test $(FLAGS) -lcunit runtime_test.c runtime.o thread_pool.o

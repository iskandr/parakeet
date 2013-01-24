CC = gcc
IFLAGS = -O3 -ffast-math -msse2 -funroll-loops
CLFLAGS = -O3 -ffast-math -msse2 -mllvm -vectorize-loops
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
	$(CC) $(FLAGS) -fPIC -c vm.c -o vm.o
	clang $(CLFLAGS) -fPIC -c vm_tiled_clang.c -o vm_tiled_clang.o
	llc -O3 -relocation-model=pic mm_par_unrolled.ll
	llc -O3 -relocation-model=pic mm_par_tiled.ll
	llc -O3 -relocation-model=pic mm_par_tiled_unrolled.ll
	llc -O3 -relocation-model=pic mm_par_reg_tiled.ll
	as -o mm_par_unrolled.o mm_par_unrolled.s
	as -o mm_par_tiled.o mm_par_tiled.s
	as -o mm_par_tiled_unrolled.o mm_par_tiled_unrolled.s
	as -o mm_par_reg_tiled.o mm_par_reg_tiled.s
	$(CC) -shared -o vm.so vm.o mm_par_unrolled.o mm_par_tiled.o \
				mm_par_tiled_unrolled.o mm_par_reg_tiled.o vm_tiled_clang.o

clean:
	rm -f *_test *.o *.so *.pyc *~

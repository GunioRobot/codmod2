import os
os.chdir('/home/j/project/causes of death/codmod/codmod2')
import threadpool
import numpy

def test_function(i):
    numpy.linalg.eig(numpy.random.random((250,250)))

def run_test(n):
    os.system('export OMP_NUM_THREADS='+str(n))
    threadpool.set_threadpool_size(n)
    threadpool.map_noreturn(test_function, [[i] for i in range(18)])

%timeit run_test(18)

%timeit run_test(1)



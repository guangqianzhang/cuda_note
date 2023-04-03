import numpy as np
from timeit import default_timer as timer
from numba import cuda, jit

number=10000000.00
def FillArrayWithoutGPU(a):
    for k in range(a.size):
        a[k] += 1

'''indicates that Numba should compile the decorated function to run on NVIDIA GPUs using CUDA'''
@jit(target_backend='cuda')
def FillArrayWithGPU(a):
    for k in range(a.size):
        a[k] += 1


a = np.ones(100000000, dtype=np.float64)

start = timer()
FillArrayWithoutGPU(a)
witoutGPU_time = timer() - start
print('withoutgpu:{}'.format(witoutGPU_time))  # 0.025049600  32.8244902
start = timer()
FillArrayWithGPU(a)
withGPU_time = timer() - start
print('withgpu:{}'.format(withGPU_time))  # 1.0484267  1.178032200000004 大数据cuda才有优势


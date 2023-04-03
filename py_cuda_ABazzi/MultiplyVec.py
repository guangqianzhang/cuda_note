import numpy as np
from timeit import default_timer as timer
from numba import vectorize

number=1e7
'''NumPy库中的vectorize装饰器允许该函数对数组进行逐元素操作，并且可以自动生成在NVIDIA GPU上运行的CUDA内核。'''
@vectorize(["float32(float32,float32)"],target='cuda')
def MultiplyMyVectors(a,b):
    # for i in range(a.size):
    #     c[i]=a[i]*b[i]  # 0.218

    return a*b  # 0.924966

def main():
    N=640000
    A=np.ones(N,dtype=np.float32)
    B=np.ones(N,dtype=np.float32)
    C=np.ones(N,dtype=np.float32)

    start=timer()
    C=MultiplyMyVectors(A,B)
    vectormultiplt_time=timer()-start

    print('C[:6]='+str(C[:6]))
    print("C[-6:]=" +str(C[-6:]))

    print('this multiplication took %f seconds'%vectormultiplt_time)

main()
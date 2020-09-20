import pycuda
import time
import numpy
import pycuda.autoinit
import pycuda.driver as drv
import matplotlib as mpl
from matplotlib import pyplot

N=int(input("Enter the input in Millions: "))

N=N*1000000

x=[1]*N
y=[2]*N
t1=time.time()
for r in range(N):
     y[r]=y[r]+x[r]
t2=time.time()     
tc=t2-t1  
print("CPU execution Time: ",tc)

from pycuda.compiler import SourceModule
mod=SourceModule("""
     __global__ void add(int *z,int *a,int *b)
     {
         const int i = threadIdx.x;
         z[i] = b[i] + a[i];
     }
     
     """)
add=mod.get_function("add")
a = numpy.full(N,1).astype(numpy.int32)
z = numpy.full(N,0).astype(numpy.int32)
b = numpy.full(N,2).astype(numpy.int32)
t11=time.time()
add(drv.Out(z),drv.In(a), drv.In(b),block=(400,1,1), grid=(1,1))
t22=time.time()
print(t22-t11)
tz=t22-t11
ex_time=list()
ex_time.append(tc)
ex_time.append(tz)
ex_time=[r*100 for r in ex_time]
print(ex_time)
pyplot.ylabel("Time Taken in milliseconds")
pyplot.title("For two million values")
mpl.pyplot.bar(["cpu","gpu"],ex_time,0.4)
pyplot.show()
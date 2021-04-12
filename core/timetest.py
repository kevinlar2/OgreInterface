import numpy as np
from numba import njit
import time

a = 2 * np.ones((50,50,50), dtype=np.int16)

b = 2.1 * np.ones((50,50,50), dtype=np.float64)

@njit(parallel=True)
def jit_square():
    for i in range(1000):
        a**2

def no_jit_square():
    for i in range(1000):
        a**2


i = 100

s1 = time.time()
for _ in range(i):
    jit_square()
e1 = time.time()
print('Jit =', (e1 - s1) / i)

s2 = time.time()
for _ in range(i):
    no_jit_square()
e2 = time.time()
print('No Jit =', (e2 - s2) / i)

print('dif =', (e2-s2)/(e1-s1))

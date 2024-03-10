import cupy as cp
import numpy as np
import time

def gpu_calc(size):
    a_gpu = cp.random.rand(size)
    b_gpu = cp.random.rand(size)

    start_time = time.time()
    _ = a_gpu + b_gpu
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start_time

    print(f"GPU计算时间: {gpu_time}秒")

def cpu_calc(size):
    a = np.random.rand(size)
    b = np.random.rand(size)

    start_time = time.time()
    _ = a + b
    cpu_time = time.time() - start_time

    print(f"CPU计算时间: {cpu_time}秒")

if __name__ == "__main__":
    size = 1000000000
    gpu_calc(size)
    cpu_calc(size)
import pycuda.autoinit
import pycuda.driver as cuda

from numpy import array, allclose
from numpy import float32

from numpy.linalg import inv, norm

from time import time

from def_prob import *

from kernels import *

from GPU_4_debug import *
from CPU_4_debug import *


A, B, g, n = dynamic()

x0 = array( [50, 100, -1500, -10, 20, 50] )

u_min = float32(  5.0 )
u_max = float32( 13.0 )
rho   = float32(  1.0 )
theta = 20.0

params = {
    'rho'  : rho,
    'u_min': u_min,
    'u_max': u_max,
    'ct'   : float32( cos( deg2rad( theta ) ) )
}

H, P, q = constraints( A, B, g, n, x0, theta )

inv_P = inv( rho * P.T @ P + H )

CPU = setup_CPU_memory( n, P, q, inv_P )
GPU = setup_GPU_memory( n, P, q, inv_P )

update_y_GPU0 = y_kernel0()
update_y_GPU1 = y_kernel1()
update_w_GPU0 = w_kernel0()
update_w_GPU1 = w_kernel1()
update_w_GPU_ = w_kernel_()
update_r_GPU0 = r_kernel0()

func = {
    'update_y_GPU0': update_y_GPU0,
    'update_y_GPU1': update_y_GPU1,
    'update_w_GPU_': update_w_GPU_,
    'update_w_GPU0': update_w_GPU0,
    'update_w_GPU1': update_w_GPU1,
    'update_r_GPU0': update_r_GPU0
}



if __name__ == "__main__":

    runtime_CPU = update_CPU( CPU, params, n )
    runtime_GPU = update_GPU( GPU, params, n, func )

    result_cpu_y = CPU['y_cpu']
    result_cpu_w = CPU['w_cpu']
    result_cpu_r = CPU['r_cpu']

    t1 = time()

    result_gpu_y = GPU['y_gpu'].to_host()
    result_gpu_w = GPU['w_gpu'].to_host()
    result_gpu_r = GPU['r_gpu'].to_host()

    t2 = time()

    memcall_GPU = t2 - t1

    print( 'time to call function:', runtime_GPU )
    print( 'time to call memory  :', memcall_GPU )

    diff_y = result_gpu_y - result_cpu_y
    diff_w = result_gpu_w - result_cpu_w
    diff_r = result_gpu_r - result_cpu_r

    print( 'residual y:', norm( diff_y ) )
    print( 'residual w:', norm( diff_w ) )
    print( 'residual r:', norm( diff_r ) )

    assert allclose( result_gpu_y, result_cpu_y )
    assert allclose( result_gpu_w, result_cpu_w )
    assert allclose( result_gpu_r, result_cpu_r )

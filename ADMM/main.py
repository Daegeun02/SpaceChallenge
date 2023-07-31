from cudg import gpuarray

from cudg.linalg import dot

import pycuda.autoinit
import pycuda.driver as cuda

from numpy import array
from numpy import float32, int32

from numpy.linalg import inv, norm

from time import time

from def_prob import *

from kernels import *

update_y_GPU0 = y_kernel0()
update_y_GPU1 = y_kernel1()
update_w_GPU0 = w_kernel0()
update_w_GPU1 = w_kernel1()
update_r_GPU0 = r_kernel0()

t1 = time()

A, B, g, n = dynamic()

x0 = array( [50, 100, -1500, -10, 20, 50] )

u_min = float32(  5.0 )
u_max = float32( 13.0 )
rho   = float32(  1.0 )
theta = 20.0

H, P, q = constraints( A, B, g, n, x0, theta )

inv_P = inv( rho * P.T @ P + H )

y = zeros((4*n  , 1 ))
w = zeros((5*n+6, 1 ))
r = zeros((5*n+6, 1 ))

# t1 = time()

d1 = 4*n
d2 = 5*n+6

## memory setup
y_gpu = gpuarray( (d1,1), y[:d1,:] )
w_gpu = gpuarray( (d2,1), w[:d2,:] )
r_gpu = gpuarray( (d2,1), r[:d2,:] )
q_gpu = gpuarray( q.shape, q )
P_gpu = gpuarray( P.shape, P )

shape = ( 4*n ,5*n+6)
buff1 = gpuarray( shape, inv_P @ P.T )

shape = (5*n+6,  1  )
buff2 = gpuarray( shape, zeros(shape) )


if __name__ == "__main__":

    for _ in range( 500 ):
        update_y_GPU0(
            buff2.gpudata, buff2.n_row, buff2.n_col,
            w_gpu.gpudata, r_gpu.gpudata, q_gpu.gpudata,
            rho,
            block=buff2._block, grid=buff2._grid
        )

        dot( y_gpu, buff1, buff2 )

        update_y_GPU1(
            y_gpu.gpudata, y_gpu.n_row, y_gpu.n_col,
            u_min, u_max,
            block=y_gpu._block, grid=y_gpu._grid
        )

        dot( buff2, P_gpu, y_gpu )

        update_w_GPU0(
            w_gpu.gpudata, w_gpu.n_row, w_gpu.n_col,
            r_gpu.gpudata, q_gpu.gpudata, buff2.gpudata,
            int32(6),
            rho, u_min, u_max,
            block=w_gpu._block, grid=w_gpu._grid
        )

        _block=(  1  ,1,1)
        _grid =(2*n+6,1,1)

        update_w_GPU1(
            w_gpu.gpudata, w_gpu.n_row, w_gpu.n_col,
            int32(6),
            u_min, u_max,
            block=_block, grid=_grid
        )

        update_r_GPU0(
            r_gpu.gpudata, r_gpu.n_row, r_gpu.n_col,
            q_gpu.gpudata, w_gpu.gpudata, buff2.gpudata,
            rho,
            block=w_gpu._block, grid=w_gpu._grid
        )

    t2 = time()

    w_pre = w_gpu.to_host()

    update_y_GPU0(
        buff2.gpudata, buff2.n_row, buff2.n_col,
        w_gpu.gpudata, r_gpu.gpudata, q_gpu.gpudata,
        rho,
        block=buff2._block, grid=buff2._grid
    )

    dot( y_gpu, buff1, buff2 )

    update_y_GPU1(
        y_gpu.gpudata, y_gpu.n_row, y_gpu.n_col,
        u_min, u_max,
        block=y_gpu._block, grid=y_gpu._grid
    )

    dot( buff2, P_gpu, y_gpu )

    update_w_GPU0(
        w_gpu.gpudata, w_gpu.n_row, w_gpu.n_col,
        r_gpu.gpudata, q_gpu.gpudata, buff2.gpudata,
        int32(6),
        rho, u_min, u_max,
        block=w_gpu._block, grid=w_gpu._grid
    )

    _block=(  1  ,1,1)
    _grid =(2*n+6,1,1)

    update_w_GPU1(
        w_gpu.gpudata, w_gpu.n_row, w_gpu.n_col,
        int32(6),
        u_min, u_max,
        block=_block, grid=_grid
    )

    update_r_GPU0(
        r_gpu.gpudata, r_gpu.n_row, r_gpu.n_col,
        q_gpu.gpudata, w_gpu.gpudata, buff2.gpudata,
        rho,
        block=w_gpu._block, grid=w_gpu._grid
    )

    y_opt = y_gpu.to_host()
    w_opt = w_gpu.to_host()
    r_opt = r_gpu.to_host()

    print( 'runtime: ', t2 - t1 )

    ## primal feasibility
    primal_res = P @ y_opt - q - w_opt

    print( 'primal feasibility: \n', norm( primal_res ) )

    print( '...: \n', norm( r_opt ) )


    ## dual feasibility
    dual_res = (-rho) * P.T @ ( w_opt - w_pre )

    print( 'dual feasibility: \n', norm( dual_res ) )

    ## lossless convexification
    u = y_opt[:3*n].reshape(-1,3)
    r = y_opt[3*n:].reshape(-1)

    loss_res = norm( u, axis=1 ) - r

    print( 'lossless convexification: \n', norm( loss_res ) )
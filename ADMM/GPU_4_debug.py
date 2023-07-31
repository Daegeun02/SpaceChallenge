from cudg import gpuarray

from cudg.linalg import dot

from numpy import zeros
from numpy import int32

from time import time



def setup_GPU_memory( n, P, q, inv_P ):

    y = zeros((4*n  , 1 ))
    w = zeros((5*n+6, 1 ))
    r = zeros((5*n+6, 1 ))

    ## memory setup
    y_gpu = gpuarray( y.shape, y )
    w_gpu = gpuarray( w.shape, w )
    r_gpu = gpuarray( r.shape, r )
    q_gpu = gpuarray( q.shape, q )
    P_gpu = gpuarray( P.shape, P )

    shape = ( 4*n ,5*n+6)
    buff1 = gpuarray( shape, inv_P @ P.T )

    shape = (5*n+6,  1  )
    buff2 = gpuarray( shape, zeros(shape) )

    GPU = {
        'y_gpu': y_gpu,
        'w_gpu': w_gpu,
        'r_gpu': r_gpu,
        'q_gpu': q_gpu,
        'P_gpu': P_gpu,
        'buff1': buff1,
        'buff2': buff2
    }

    return GPU

def update_GPU( GPU, params, n, func ):

    y_gpu = GPU['y_gpu']
    w_gpu = GPU['w_gpu']
    r_gpu = GPU['r_gpu']
    q_gpu = GPU['q_gpu']
    P_gpu = GPU['P_gpu']
    buff1 = GPU['buff1']
    buff2 = GPU['buff2']

    update_y_GPU0 = func['update_y_GPU0']
    update_y_GPU1 = func['update_y_GPU1']
    update_w_GPU0 = func['update_w_GPU0']
    update_w_GPU1 = func['update_w_GPU1']
    update_r_GPU0 = func['update_r_GPU0']

    rho   = params['rho']
    u_min = params['u_min']
    u_max = params['u_max']

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
        block=r_gpu._block, grid=r_gpu._grid
    )


def solve_GPU( GPU, params, n, func ):

    y_gpu = GPU['y_gpu']
    w_gpu = GPU['w_gpu']
    r_gpu = GPU['r_gpu']
    q_gpu = GPU['q_gpu']
    P_gpu = GPU['P_gpu']
    buff1 = GPU['buff1']
    buff2 = GPU['buff2']

    update_y_GPU0 = func['update_y_GPU0']
    update_y_GPU1 = func['update_y_GPU1']
    update_w_GPU0 = func['update_w_GPU0']
    update_w_GPU1 = func['update_w_GPU1']
    update_r_GPU0 = func['update_r_GPU0']

    rho   = params['rho']
    u_min = params['u_min']
    u_max = params['u_max']

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
            block=r_gpu._block, grid=r_gpu._grid
        )

    y_opt = y_gpu.to_host()

    return y_opt
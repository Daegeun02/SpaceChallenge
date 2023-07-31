from numpy import zeros

from numpy.linalg import norm

from time import time



def setup_CPU_memory( n, P, q, inv_P ):

    y = zeros((4*n  , 1 ))
    w = zeros((5*n+6, 1 ))
    r = zeros((5*n+6, 1 ))

    buff1 = inv_P @ P.T
    
    CPU = {
        'y_cpu': y,
        'w_cpu': w,
        'r_cpu': r,
        'q_cpu': q,
        'P_cpu': P,
        'buff1': buff1
    }

    return CPU


def update_CPU( CPU, params, n ):

    y_cpu = CPU['y_cpu']
    w_cpu = CPU['w_cpu']
    r_cpu = CPU['r_cpu']
    q_cpu = CPU['q_cpu']
    P_cpu = CPU['P_cpu']
    buff1 = CPU['buff1']

    rho   = params['rho']
    u_min = params['u_min']
    u_max = params['u_max']

    t1 = time()

    y_cpu[:,:] = buff1 @ ( rho * ( q_cpu + w_cpu ) - r_cpu )

    y_cpu[3*n:,:][ y_cpu[3*n:,:] < u_min ] = u_min
    y_cpu[3*n:,:][ y_cpu[3*n:,:] > u_max ] = u_max

    w_cpu[:,:] = (1/rho) * r_cpu + P_cpu @ y_cpu - q_cpu

    buff_w = zeros(4)

    w_cpu[:6,:] = 0

    ## projection w1
    w_cpu[ 6 :n+6,:][ w_cpu[ 6 :n+6,:] < 0 ] = 0
    
    ## projection w2
    for i in range( n ):
        buff_w[:] = w_cpu[4*i+n+6:4*i+n+10,0]
        
        mngw = norm( buff_w[0:3] )
        
        if ( mngw > buff_w[3] ):
            
            p = 0.5 * ( mngw + buff_w[3] )
            
            if ( p > u_max ):
                p = u_max
            elif ( p < u_min ):
                p = u_min
        
            buff_w[0:3] *= ( p / mngw )
            buff_w[ 3 ] = p
        
        w_cpu[4*i+n+6:4*i+n+10,0] = buff_w[:]

    r_cpu[:,:] += rho * ( P_cpu @ y_cpu - q_cpu - w_cpu )

    t2 = time()

    return t2 - t1
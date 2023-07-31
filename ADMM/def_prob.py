from numpy import zeros, array
from numpy import eye
from numpy import cos, deg2rad

from numpy.linalg import matrix_power



def dynamic():

    T  = 35
    dt = 0.35
    n  = int( T / dt )

    A = array([
        [ 1, 0, 0, dt,  0,  0 ],
        [ 0, 1, 0,  0, dt,  0 ],
        [ 0, 0, 1,  0,  0, dt ],
        [ 0, 0, 0,  1,  0,  0 ],
        [ 0, 0, 0,  0,  1,  0 ],
        [ 0, 0, 0,  0,  0,  1 ]
    ])

    b = 0.5 * dt * dt
    B = array([
        [  b,  0,  0 ],
        [  0,  b,  0 ],
        [  0,  0,  b ],
        [ dt,  0,  0 ],
        [  0, dt,  0 ],
        [  0,  0, dt ]
    ])

    g = array([
        [ 0 ],
        [ 0 ],
        [ 9.81 * b ],
        [ 0 ],
        [ 0 ],
        [ 9.81 * dt ]
    ])

    return A, B, g, n


def constraints( A, B, g, n, x0, theta ):

    G = zeros((6,3*n))
    C = ( matrix_power( A, n ) @ x0 ).reshape(-1,1)

    for i in range( n ):

        Ai = matrix_power( A, n-i-1 )

        G[:,3*i:3*i+3] = Ai @ B
        C[:,:] += Ai @ g

    C[:,:] *= (-1)

    U3 = zeros(( n ,3*n))
    TM = zeros(( n , n ))
    D1 = zeros((4*n,3*n))
    D2 = zeros((4*n, n ))

    ct = cos( deg2rad( theta ) ) * (-1)

    for i in range( n ):
        U3[  i  ,3*i+2] = -1

        TM[  i  ,  i  ] = ct

        D1[4*i+0,3*i+0] = 1
        D1[4*i+1,3*i+1] = 1
        D1[4*i+2,3*i+2] = 1

        D2[4*i+3,  i  ] = 1

    H = zeros(( 4*n ,4*n))
    P = zeros((5*n+6,4*n))
    q = zeros((5*n+6, 1 ))

    H[:3*n,:3*n] = eye( 3*n )

    P[   : 6 ,:3*n] = G
    P[ 6 :n+6,:3*n] = U3
    P[n+6:   ,:3*n] = D1

    P[ 6 :n+6,3*n:] = TM
    P[n+6:   ,3*n:] = D2

    q[:6,:] = C

    return H, P, q
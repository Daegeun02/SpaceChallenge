from numpy import zeros, array
from numpy import inf

from numpy.linalg import norm



def optimize( _func, x0, bound=[-inf,inf], lr=1e4, itr=500, tol=1e-8 ):

    xK = array( x0 )
    N  = len( x0 )

    fK, gK = _func( x0 )

    footp = zeros((itr,len(xK)))
    trace = zeros(itr)

    for i in range( itr ):
        
        xK[:] -= lr * gK

        for k in range( N ):
            if ( xK[k] < bound[0] ):
                xK[k] = bound[0]
            elif ( xK[k] > bound[1] ):
                xK[k] = bound[1]
            else:
                pass

        fP     = fK
        fK, gK = _func( xK )

        footp[i,:] = xK
        trace[i]   = norm( fK )

        print( xK, fK )

        # if ( abs( norm( fK ) - norm( fP ) ) < tol ):
        #     print( 'gradient at optimal =>', gK )
        #     return xK, trace, i

        if ( norm( fK ) < norm( fP ) ):
            lr *= 1.2
        else:
            lr *= 0.5

    print( 'gradient at optimal =>', gK )

    return xK, trace, i
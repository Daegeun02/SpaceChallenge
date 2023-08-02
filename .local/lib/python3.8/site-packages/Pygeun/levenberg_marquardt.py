## Levenberg-Marquardt Algorithm
from numpy import eye 
from numpy import zeros_like

from numpy.linalg import norm, inv



class LevenbergMarquardt:


    def __init__(self, lam, tol, itr):

        self.lam = lam
        self.tol = tol
        self.itr = itr


    def solve(self, func, jacb, x0, lam=-1, root=False, force_return=False):
        '''
        root = levenbergMarquardt( 
            func, jacb, x0, args
        )

        Finds a root of f(x) = 0
        func: f(x), vector variable -> vector valued
        jacb: jacobian matrix of f(x)
        x0  : initial condition
        lam : lambda for regularizer

        *** x is vector ***
        '''
        if ( lam == -1 ):
            lam = self.lam
        else:
            lam = lam
        tol = self.tol
        itr = self.itr

        xK = x0
        uxK = zeros_like( x0 )
        fK = func( xK )         ## object function
        fP = zeros_like( fK )   ## previous object

        ## initialize lambda
        dim = len( fK )
        lam = lam * eye( dim )

        for _ in range( itr ):
            Df = jacb( xK )
            ## 1. check optimality condition
            opt_cond = Df.T @ fK

            # if ( norm( opt_cond ) < tol ):
            #     print( fK )
            #     return xK
            if ( norm( fK ) < tol ):
                return xK

            ## 2. update xK
            dx = inv( Df.T @ Df + lam ) @ opt_cond

            uxK[:] = xK - dx

            ## remember previous objects
            fP[:] = fK

            ## re evaluate objects
            fK[:] = func( uxK )

            ## 3. check tentative iterate
            if ( norm( fK ) < norm( fP ) ):
                ## update xK
                xK[:] = uxK
                lam *= 0.8
            else:
                ## do not update xK
                fK[:] = fP
                lam *= 1.2

        if force_return:
            # print( f'force returned careful for numerical error => {norm( opt_cond )}' )
            return xK

        print( 'optimal condition =>', opt_cond )
        print( 'lambda =>', lam[0,0] )
        print('Levenberg-Marquardt fail to find root of given function')
    

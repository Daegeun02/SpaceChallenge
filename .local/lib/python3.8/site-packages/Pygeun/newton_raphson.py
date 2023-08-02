from numpy import abs



class NewtonRaphson:


    def __init__(self, tol, itr):

        self.tol = tol
        self.itr = itr


    def solve(self, func, grad, x0, force_return=False):
        '''
        root = newtonRaphson(
            func, grad, x0, args
        )

        Finds a root of f(x) = 0 by combining the Newton - Raphson method.
        func: f(x)
        grad: gradient of f(x)
        x0  : initial condition
        args: arguments of func, grad

        *** x is scalar ***
        '''

        tol = self.tol
        itr = self.itr

        xK = x0

        if ( func( xK ) == 0.0 ):
            return xK

        for _ in range( itr ):
            f = func( xK ) 
            g = grad( xK )

            if ( abs( f ) < tol ):
                return xK

            if ( g == 0.0 ):
                print('zero division error')
                break

            dx = f / g

            xK -= dx

        if force_return:
            # print( f'force returned careful for numerical error => {abs( f )}' )
            return xK

        print('NewtonRaphson fail to find root of given function')
        print( f )
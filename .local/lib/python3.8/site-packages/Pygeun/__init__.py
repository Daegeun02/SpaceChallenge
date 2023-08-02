__version__ = '0.0.4'



## optimization algorithm's are implemented in this package
from .newton_raphson       import NewtonRaphson
from .levenberg_marquardt  import LevenbergMarquardt
from .gradient_descent     import optimize
 


tol = 1.0e-8
itr = 500
NR_solver = NewtonRaphson( tol, itr )
NR_solver = NR_solver.solve


lam = 1
tol = 1.0e-8
itr = 500
LM_solver = LevenbergMarquardt( lam, tol, itr )
LM_solver = LM_solver.solve
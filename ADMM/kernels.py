import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule



def y_kernel0():
    kernel_function = \
    '''
    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_y_GPU0(
        float* result, int n_row, int n_col,
        float* W, float* L, float* q,
        float rho
    ) {
        const int _n = 32;

        __shared__ float sM[1024];
        
        int thread_id = tx * _n + ty;
        sM[thread_id] = 0;

        int _row = tx + bx * _n;
        int _col = ty + by * _n;
        
        if ( (_row < n_row) && (_col < n_col) )
        {
            sM[thread_id] += W[_row];
            sM[thread_id] += q[_row];
            sM[thread_id] *= rho;
            sM[thread_id] -= L[_row];

            result[_row] = sM[thread_id];
        }

        __syncthreads();
    }
    '''

    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_y_GPU0' )

def y_kernel1():
    kernel_function = \
    '''
    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_y_GPU1(
        float* Y, int n_row, int n_col,
        float u_min, float u_max
    ) {
        const int _n = 32;

        __shared__ float sM[1024];

        int thread_id = tx * _n + ty;

        int _row  = tx + bx * _n;
        int _col  = ty + by * _n;
        int m_row = (int)( (float)(n_row) * 0.75 ) - 1;

        if ( (_row > m_row) && (_row < n_row) && (_col < n_col) )
        {
            sM[thread_id] = Y[_row];

            if ( sM[thread_id] < u_min )
            {
                sM[thread_id] = u_min;
            }
            else if ( sM[thread_id] > u_max )
            {
                sM[thread_id] = u_max;
            }

            Y[_row] = sM[thread_id];
        }

        __syncthreads();
    }
    '''
    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_y_GPU1' )


def w_kernel_():
    kernel_function = \
    '''
    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_w_GPU_(
        float* buff2, int n_row, int n_col,
        float* P, float* Y,
        int DOF,
        float ct
    ) {
        const int _n = 32;

        __shared__ float sM[1024];

        int thread_id = tx * _n + ty;
        sM[thread_id] = 0;

        int _row = tx + bx * _n;
        int _col = ty + by * _n;
        int n    = (int)( (float)(n_row - DOF) * 0.2 );

        if ( _row < n_row )
        {
            if ( _row < DOF )
            {

            }
            else if ( (_row < n + DOF) && (_col < n_col) )
            {
                int idxy = _row - DOF;
                int idxu = 3 * idxy + 2;
                int idxr = idxy + 3 * n;

                sM[thread_id] -= Y[idxr];
                sM[thread_id] *= ct;
                sM[thread_id] -= Y[idxu];
            }
            else if ( (_row % 4 == 0) && (_col < n_col) )
            {
                int idxy = _row - DOF - n;
                int idxu = 3 * ( idxy / 4 ) + 0;

                sM[thread_id] = Y[idxu];
            }
            else if ( (_row % 4 == 1) && (_col < n_col) )
            {
                int idxy = _row - DOF - n;
                int idxu = 3 * ( idxy / 4 ) + 1;

                sM[thread_id] = Y[idxu];
            }
            else if ( (_row % 4 == 2) && (_col < n_col) )
            {
                int idxy = _row - DOF - n;
                int idxu = 3 * ( idxy / 4 ) + 2;

                sM[thread_id] = Y[idxu];
            }
            else if ( (_row % 4 == 3) && (_col < n_col) )
            {
                int idxy = _row - DOF - n;
                int idxr = idxy / 4 + 3 * n;

                sM[thread_id] = Y[idxr];
            }
        }

        if ( (_row < n_row) && (_col < n_col) )
        {
            buff2[_row] = sM[thread_id];
            //buff2[_row] = _row - DOF - n;
        }
    }
    '''
    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_w_GPU_' )


def w_kernel0():
    kernel_function = \
    '''
    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_w_GPU0(
        float* W, int n_row, int n_col,
        float* L, float* q, float* buff2, 
        int DOF,
        float rho, float u_min, float u_max
    ) {
        const int _n = 32;

        __shared__ float sM[1024];

        int thread_id = tx * _n + ty;
        sM[thread_id] = 0;

        int _row = tx + bx * _n;
        int _col = ty + by * _n;
        int n    = (int)( (float)(n_row - DOF) * 0.2 );

        if ( (_row < n_row) && (_col < n_col) )
        {
            sM[thread_id] += L[_row];
            sM[thread_id] /= rho;
            sM[thread_id] += buff2[_row];
            sM[thread_id] -= q[_row];

            W[_row] = sM[thread_id];
        }

        __syncthreads();
    }
    '''
    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_w_GPU0' )


def w_kernel1():
    kernel_function = \
    '''
    #include <math.h>

    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_w_GPU1(
        float* W, int n_row, int n_col,
        int DOF, 
        float u_min, float u_max
    ) {
        __shared__ float sM[7];

        int _row = bx;
        int _col = by;
        int n    = (int)( (float)(n_row - DOF) * 0.2 );

        if ( (tx == 0) && (ty == 0) )
        {
            if ( (_row < DOF) && (_col < n_col) )
            {
                W[_row] = 0;
            }

            else if ( (_row < n + DOF) && (_col < n_col) )
            {
                if ( W[_row] < 0 )
                {
                    W[_row] = 0;
                }
            }

            else if ( _col < n_col )
            {
                int idxn = 4 * ( _row - DOF - n ) + DOF + n;

                sM[0] = W[idxn+0];
                sM[1] = W[idxn+1];
                sM[2] = W[idxn+2];
                sM[3] = W[idxn+3];

                sM[4] = 0;

                sM[4] += sM[0] * sM[0];
                sM[4] += sM[1] * sM[1];
                sM[4] += sM[2] * sM[2];

                sM[5] = sqrt( sM[4] );

                if ( sM[5] > sM[3] )
                {
                    sM[6] = 0.5 * ( sM[5] + sM[3] );

                    if ( sM[6] < u_min )
                    {
                        sM[6] = u_min;
                    }
                    else if ( sM[6] > u_max )
                    {
                        sM[6] = u_max;
                    }

                    W[idxn+0] = sM[0] * sM[6] / sM[5];
                    W[idxn+1] = sM[1] * sM[6] / sM[5];
                    W[idxn+2] = sM[2] * sM[6] / sM[5];
                    W[idxn+3] = sM[6];
                }
            }
        }

        __syncthreads();
    }
    '''
    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_w_GPU1' )


def r_kernel0():
    kernel_function = \
    '''
    #define tx ( threadIdx.x )
    #define ty ( threadIdx.y )
    #define bx ( blockIdx.x )
    #define by ( blockIdx.y )

    __global__ void update_r_GPU0(
        float* L, int n_row, int n_col,
        float* q, float* W, float* buff2,
        float rho
    ) {
        const int _n = 32;

        __shared__ float sM[1024];

        int thread_id = tx * _n + ty;
        sM[thread_id] = 0;

        int _row = tx + bx * _n;
        int _col = ty + by * _n;

        if ( (_row < n_row) && (_col < n_col) )
        {
            sM[thread_id] += buff2[_row];
            sM[thread_id] -= q[_row];
            sM[thread_id] -= W[_row];
            sM[thread_id] *= rho;

            L[_row] += sM[thread_id];
        }
    }
    '''
    kernel = SourceModule( kernel_function )

    return kernel.get_function( 'update_r_GPU0' )
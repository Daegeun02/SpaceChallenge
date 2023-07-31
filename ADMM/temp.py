'''
        if ( (_row < n_row) && (_col < n_col) )
        {
            if ( _row < DOF )
            {
                W[_row] = 0;
            }

            else if ( _row < n + DOF )
            {
                if ( sM[thread_id] < 0 )
                {
                    W[_row] = 0;
                }
            }

            else
            {
                if ( (_row - DOF - n) % 4 == 0 )
                {
                    sM[thread_id+0] = W[_row+0];
                    sM[thread_id+1] = W[_row+1];
                    sM[thread_id+2] = W[_row+2];
                    sM[thread_id+3] = W[_row+3];

                    sM[thread_id+4] = 0;

                    sM[thread_id+4] += sM[thread_id+0] * sM[thread_id+0];
                    sM[thread_id+4] += sM[thread_id+1] * sM[thread_id+1];
                    sM[thread_id+4] += sM[thread_id+2] * sM[thread_id+2];

                    sM[thread_id+5] = _sqrt( sM[thread_id+4] );

                    if ( sM[thread_id+5] > sM[thread_id+3] )
                    {
                        sM[thread_id+6] = 0.5 * ( sM[thread_id+5] + sM[thread_id+3] );

                        if ( sM[thread_id+6] < u_min )
                        {
                            sM[thread_id+6] = u_min;
                        }
                        else if ( sM[thread_id+6] > u_max )
                        {
                            sM[thread_id+6] = u_max;
                        }

                        W[_row+0] = sM[thread_id+0] * sM[thread_id+6] / sM[thread_id+5];
                        W[_row+1] = sM[thread_id+1] * sM[thread_id+6] / sM[thread_id+5];
                        W[_row+2] = sM[thread_id+2] * sM[thread_id+6] / sM[thread_id+5];
                        W[_row+3] = sM[thread_id+6];
                    }
                }
            }
        }

        __syncthreads();
'''
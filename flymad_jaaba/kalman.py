
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

def kalman_filter(series, estimate, Q=1e-5, R=0.5):
    """
    pass a pandas series and estimated value, with optional kwargs for parameters.
    """
    z = series.notnull()
    n_iter = len(z)
    sz = (n_iter,) # size of array
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    # intial guesses
    xhat[0] = estimate
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat[-1]


def smooth_by_kalman(series, width, estimate=None, Q=1e-5, R=0.5):
    """
    pass a pandas series and integer for width of 'moving window'
    returns the mean of kalman filtering in both directions
    """ 
    if estimate == None:
        foo = series.mean()
    else:
        foo = estimate
    data_fwd = []
    data_rev = []   
    for a in range(1, len(series)):
        if a < width:
            start = 0
        else:
            start = a-width
        foo = kalman_filter(series[start:a], foo, Q, R)
        data_rev.append(foo)
    for a in range(width+1, width+len(series)):
        foo = kalman_filter(series[a-width:a], foo, Q, R)
        data_fwd.append(foo)
    return [sum(x)/2.0 for x in zip(data_fwd,data_rev)]    
    


def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P



def kalman_series(df):
    ndim = len(df.dtypes)
    x = np.matrix(np.zeros(ndim)).T 
    P = np.matrix(np.eye(ndim))*1000    
    R = 0.01**2
    motion = np.matrix(np.zeros(ndim)).T #np.matrix('0. 0. 0. 0.').T
    Q = np.matrix(np.eye(ndim))
    F = np.matrix('''
      1. 0. 1. 0.;
      0. 1. 0. 1.;
      0. 0. 1. 0.;
      0. 0. 0. 1.
      '''),
    H = np.matrix('''
      1. 0. 0. 0.;
      0. 1. 0. 0.''')
    result = []
    for meas in zip(df[x] for x in df.columns):
        x, P = kalman(x, P, meas, R, motion, Q,F,H)
        r = x[0][0][0]
        result.append(float(r))
    return pd.Series(result, index=series.index)

















###################################################################################################3



def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))


def kalman_1d(x, P, measurement, R,
              motion = np.matrix('0. 0. ').T,
              Q = np.matrix(np.eye(1))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''1.  '''),
                  H = np.matrix('''1.'''))



def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def demo_kalman_1d():
    x = np.matrix('0. 0. ').T 
    P = np.matrix(np.eye(1))*1000 # initial uncertainty

    N = 20
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x**2
    observed_y = true_y + 0.05*np.random.random(N)*true_y
    plt.plot(true_x, observed_y, 'ro')
    result = []
    R = 0.01**2
    for meas in observed_y:
        x, P = kalman_1d(x, P, meas, R)
        print 'meas: \t', meas, '\tx: \t', x
        result.append((x).tolist())
    print result
    kalman_y = result
    plt.plot(true_x, kalman_y, 'g-')
    plt.show()
    


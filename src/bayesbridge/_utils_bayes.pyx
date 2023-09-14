
cimport cython
from cython cimport cdivision, boundscheck, wraparound
from libcpp cimport bool as bool_cpp
from libc.math cimport fabs
import numpy as np
cimport numpy as np


@boundscheck(False)
@wraparound(False) 
cpdef bool_cpp _is_sorted(long[:] argsort_indices):
    cdef long i
    cdef ssize_t size = argsort_indices.shape[0]
    for i in range(size):
        if argsort_indices[i] != i:
            return False
    return True


@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cpdef _get_thickness(double[:] depth):
    cdef ssize_t size = depth.shape[0]
    cdef int i
    cdef double d1, d2
    cdef double[:] thickness = np.zeros(size, dtype=np.double)
    d1 = 0.
    for i in range(size - 1):
        d2 = (depth[i] + depth[i+1]) / 2.
        thickness[i] = d2 - d1
        d1 = d2
    return np.asarray(thickness)


@boundscheck(False)
@wraparound(False) 
cpdef (int, int) _closest_and_final_index(double[:] ndarray, double value):
    cdef int closest_idx = 0 
    cdef int i
    cdef double vmin = fabs(ndarray[0] - value)
    cdef double v
    cdef ssize_t size = ndarray.shape[0]
    
    for i in range(1, size):
        v = fabs(ndarray[i] - value)
        if v < vmin:
            vmin = v
            closest_idx = i
    if ndarray[closest_idx] > value:
        return closest_idx, closest_idx
    return closest_idx, closest_idx+1
        

@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cdef double _interpolate_linear_1d(double xp, double[:] x, double[:] y):
    """ Linear interpolation in 1-D
    Parameters
    ----------
    xp : float
        Interpolation point
    x, y : ndarray of floats
        Discrete set of known data points
        
    Returns
    -------
    yp : float
        Interpolated value
    """
    
    cdef ssize_t xlen = x.shape[0]
    cdef int i
    cdef double x0, x1, y0, y1
    
    if xp <= x[0]:
        return y[0]
    elif xp >= x[xlen-1]:
        return y[xlen-1]
    i = floor_index(xp, x, xlen)
    x0 = x[i]
    x1 = x[i+1]
    y0 = y[i]
    y1 = y[i+1]
    return y0 + (xp - x0) * (y1 - y0) / (x1 - x0)


@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cdef double _interpolate_nearest_1d(double xp, double[:] x, double[:] y):
    """ Nearest neighbour interpolation in 1-D
    Parameters
    ----------
    xp : float
        Interpolation point
    x, y : ndarray of floats
        Discrete set of known data points
        
    Returns
    -------
    yp : float
        Interpolated value, corresponding to a discrete value yk such that
        k = argmin(abs(xp - x))
    """
    cdef ssize_t xlen = x.shape[0]
    cdef int i
    cdef double x0, x1, y0, y1
    
    if xp <= x[0]:
        return y[0]
    elif xp >= x[xlen-1]:
        return y[xlen-1]
    i = nearest_index(xp, x, xlen)
    return y[i]


@boundscheck(False)
@wraparound(False) 
cpdef int nearest_index(double xp, double[:] x, ssize_t xlen):
    cdef int i
    cdef double x0, x1
    for i in range(xlen):
        x0 = x[i]
        x1 = x[i + 1]
        if x0 <= xp <= x1:
            if fabs(x0-xp) < fabs(x1-xp):
                return i
            return i + 1



@boundscheck(False)
@wraparound(False) 
cdef int floor_index(double xp, double[:] x, ssize_t xlen):
    cdef int i
    for i in range(xlen):
        if x[i] <= xp <= x[i+1]:
            return i


#@boundscheck(False)
#@wraparound(False) 
#def interpolate_1d(xp, double[:] x, double[:] y, bool_cpp nearest=False):
#    """ Linear or nearest neighbour interpolation in 1-D
#    Parameters
#    ----------
#    xp : float, ndarray of floats
#        Interpolation point(s)
#        
#    x, y : ndarray of floats
#        Discrete set of known data points
#    
#    nearest : bool
#        If True, nearest neighbour interpolation is performed. Default is False
#    
#    Returns
#    -------
#    yp : float
#        Interpolated value(s)
#    """
#    func = _interpolate_nearest_1d if nearest else _interpolate_linear_1d
#    if np.isscalar(xp):
#        return func(xp, x, y)
#    cdef int i, size = xp.shape[0]
#    cdef double[:] yp = np.zeros(size, dtype=np.double)
#    for i in range(size):
#        yp[i] = func(xp[i], x, y)
#    return np.asarray(yp)


@boundscheck(False)
@wraparound(False) 
cpdef interpolate_linear_1d(xp, double[:] x, double[:] y):
    """ Linear or nearest neighbour interpolation in 1-D
    Parameters
    ----------
    xp : float, ndarray of floats
        Interpolation point(s)
        
    x, y : ndarray of floats
        Discrete set of known data points
    
    nearest : bool
        If True, nearest neighbour interpolation is performed. Default is False
    
    Returns
    -------
    yp : float
        Interpolated value(s)
    """
    if np.isscalar(xp):
        return _interpolate_linear_1d(xp, x, y)
    cdef int i, size = xp.shape[0]
    cdef double[:] yp = np.zeros(size, dtype=np.double)
    for i in range(size):
        yp[i] = _interpolate_linear_1d(xp[i], x, y)
    return np.asarray(yp)


@boundscheck(False)
@wraparound(False) 
cpdef interpolate_nearest_1d(xp, double[:] x, double[:] y):
    """ Linear or nearest neighbour interpolation in 1-D
    Parameters
    ----------
    xp : float, ndarray of floats
        Interpolation point(s)
        
    x, y : ndarray of floats
        Discrete set of known data points
    
    nearest : bool
        If True, nearest neighbour interpolation is performed. Default is False
    
    Returns
    -------
    yp : float
        Interpolated value(s)
    """
    if np.isscalar(xp):
        return _interpolate_nearest_1d(xp, x, y)
    cdef int i, size = xp.shape[0]
    cdef double[:] yp = np.zeros(size, dtype=np.double)
    for i in range(size):
        yp[i] = _interpolate_nearest_1d(xp[i], x, y)
    return np.asarray(yp)


@boundscheck(False)
@wraparound(False) 
@cdivision(True)
cpdef inverse_covariance(double sigma, 
                                        double r, 
                                        Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double factor = 1 / (sigma**2 * (1 - r**2))
    cdef double[:, ::1] matrix = np.zeros((n, n), dtype=np.double)
    
    for i in range(1, n-1):
        matrix[i, i] = (1 + r**2) * factor
        matrix[i, i-1] = -r * factor
        matrix[i-1, i] = -r * factor
    matrix[0, 0] = factor
    matrix[n-1, n-1] = factor
    matrix[i+1, i] = -r * factor
    matrix[i, i+1] = -r * factor
    return np.asarray(matrix)
    



